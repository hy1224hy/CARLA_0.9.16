"""Frozen protocol and offline evidence helpers for P0-BASE C0 evaluation.

This module deliberately has no CARLA dependency.  Contract, checkpoint and
result invariants therefore remain testable on machines without a simulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from models.actor_critic import Actor
from training.action_protocol import (
    ACTION_MAPPING_VERSION,
    ACTION_PROTOCOL_VERSION,
    LEGACY_ACTION_MAPPING_VERSION,
    LEGACY_ACTION_PROTOCOL_VERSION,
    LONGITUDINAL_DEADZONE,
    action_protocol_spec,
)
from training.checkpoint_protocol import (
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_TRAINING_PROTOCOL_VERSION,
    LIFECYCLE_PROTOCOL_VERSION,
    OBSERVATION_PROTOCOL_VERSION,
    TRAINING_PROTOCOL_VERSION,
    atomic_write_json,
    load_checkpoint_file,
    scenario_pool_fingerprint,
    sha256_file,
    validate_checkpoint_header,
)
from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    LEGACY_METRICS_PROTOCOL_VERSION,
    METRICS_PROTOCOL_VERSION,
    POLICY_ACTION_RATE_NAMES,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
)


LEGACY_C0_PROTOCOL_VERSION = "p0_base_c0_eval_v1"
C0_PROTOCOL_VERSION = "p0_base_c0_eval_v2"
LEGACY_C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v2"
C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v3"
C0_SUMMARY_VERSION = "p0_base_c0_summary_v1"
C0_MANIFEST_VERSION = "p0_base_c0_manifest_v1"
C0_RESOLVED_CONFIG_VERSION = "p0_base_c0_resolved_v1"
C0_VALIDATOR_REPORT_VERSION = "p0_base_c0_validator_v1"
AGENT_IDS = (0, 1)
OUTCOMES = ("success", "collision", "timeout", "other")
REPEATABILITY_FIELDS = (
    "case_id",
    "outcome",
    "episode_end_reason",
    "episode_length",
    "agent_0_reached",
    "agent_1_reached",
)
RUN_ID_RE = re.compile(
    r"^P0_BASE_C0_(smoke|gate|validation)_u([0-9]{6})_([0-9]{8}_[0-9]{6})$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SUMMARY_CSV_FIELDS = (
    "scope",
    "name",
    "case_count",
    "success_count",
    "collision_count",
    "timeout_count",
    "other_count",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "other_rate",
    "mean_episode_length",
    "mean_team_return",
)


class C0ProtocolError(RuntimeError):
    """Raised when a C0 input or artifact violates the frozen protocol."""


def canonical_json_sha256(payload: Any) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C0ProtocolError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_strict_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise C0ProtocolError(f"JSON file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    C0ProtocolError(f"non-finite JSON constant is forbidden: {token}")
                ),
            )
    except json.JSONDecodeError as error:
        raise C0ProtocolError(f"invalid JSON at {resolved}: {error}") from error


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if not isinstance(payload, Mapping):
        raise C0ProtocolError(f"{context} must be an object")
    actual = set(payload)
    if actual != expected:
        raise C0ProtocolError(
            f"{context} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _inside(project_dir: Path, relative: str, *, context: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise C0ProtocolError(f"{context} must be a non-empty relative path")
    candidate = (project_dir.resolve() / relative).resolve()
    try:
        candidate.relative_to(project_dir.resolve())
    except ValueError as error:
        raise C0ProtocolError(f"{context} escapes project directory: {relative}") from error
    return candidate


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise C0ProtocolError(f"{context} must be a positive integer")
    return value


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise C0ProtocolError(f"{context} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise C0ProtocolError(f"{context} must be finite numeric data") from error
    if not math.isfinite(result):
        raise C0ProtocolError(f"{context} must be finite numeric data")
    return result


def _validate_scenario(payload: Any, *, path: Path) -> None:
    required = {
        "schema_version",
        "town",
        "difficulty",
        "family",
        "description",
        "tags",
        "goal_zone",
        "agents",
    }
    _require_exact_fields(payload, required, context=f"scenario {path.name}")
    if payload["schema_version"] != 3 or payload["town"] != "Town02":
        raise C0ProtocolError(f"scenario {path.name} must use schema 3 and Town02")
    if payload["difficulty"] not in (1, 2, 3, 4):
        raise C0ProtocolError(f"scenario {path.name} has invalid difficulty")
    if not str(payload["family"]).strip():
        raise C0ProtocolError(f"scenario {path.name} has empty family")
    if not isinstance(payload["tags"], list) or not payload["tags"]:
        raise C0ProtocolError(f"scenario {path.name} must have tags")
    if set(payload["agents"]) != {"agent0", "agent1"}:
        raise C0ProtocolError(f"scenario {path.name} must define agent0 and agent1")


def validate_c0_contract(payload: Any, *, project_dir: Path) -> None:
    top_fields = {
        "protocol_version",
        "contract_id",
        "status",
        "purpose",
        "environment",
        "action",
        "checkpoint",
        "profiles",
        "outcomes",
        "outputs",
        "repeatability",
        "run_id_prefix",
        "critical_source_paths",
    }
    _require_exact_fields(payload, top_fields, context="C0 contract")
    protocol_version = payload["protocol_version"]
    if protocol_version not in {
        LEGACY_C0_PROTOCOL_VERSION,
        C0_PROTOCOL_VERSION,
    }:
        raise C0ProtocolError("C0 protocol version mismatch")
    legacy = protocol_version == LEGACY_C0_PROTOCOL_VERSION
    expected_contract_id = (
        "P0_BASE_C0_TOWN02_V1" if legacy else "P0_BASE_C0_TOWN02_V2"
    )
    if payload["contract_id"] != expected_contract_id:
        raise C0ProtocolError("C0 contract_id mismatch")
    if payload["status"] != "frozen" or payload["run_id_prefix"] != "P0_BASE_C0_":
        raise C0ProtocolError("C0 contract must be frozen with the registered prefix")

    env = payload["environment"]
    _require_exact_fields(
        env,
        {
            "carla_version",
            "town",
            "num_agents",
            "straight_road_length",
            "timestep",
            "max_steps",
            "reward_version",
            "reward_gamma",
            "adversarial_agent_ids",
            "allow_extra_traffic",
        },
        context="environment",
    )
    if (
        env["carla_version"] != "0.9.16"
        or env["town"] != "Town02"
        or env["num_agents"] != 2
        or env["reward_version"] != "v2"
        or env["adversarial_agent_ids"] != []
        or env["allow_extra_traffic"] is not False
    ):
        raise C0ProtocolError("C0 environment must be two-agent Town02 v2 without traffic/adversary")
    _positive_int(env["max_steps"], context="environment.max_steps")
    for name in ("straight_road_length", "timestep", "reward_gamma"):
        if _finite(env[name], context=f"environment.{name}") <= 0:
            raise C0ProtocolError(f"environment.{name} must be positive")

    action = payload["action"]
    if legacy:
        expected_action = {
            "mode": "deterministic_actor_mean_tanh",
            "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION,
            "action_protocol_version": LEGACY_ACTION_PROTOCOL_VERSION,
            "throttle_brake_mapping": LEGACY_ACTION_MAPPING_VERSION,
        }
    else:
        expected_action = {
            "mode": "deterministic_actor_mean_tanh",
            "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION,
            "action_protocol_version": ACTION_PROTOCOL_VERSION,
            "mapping_version": ACTION_MAPPING_VERSION,
            "policy_action_dim": 2,
            "environment_action_dim": 3,
            "longitudinal_deadzone": LONGITUDINAL_DEADZONE,
        }
    _require_exact_fields(action, set(expected_action), context="action")
    if dict(action) != expected_action:
        raise C0ProtocolError("C0 action policy differs from the frozen deterministic protocol")

    checkpoint = payload["checkpoint"]
    _require_exact_fields(
        checkpoint,
        {
            "schema_version",
            "training_protocol_version",
            "required_training_contract_id",
            "required_training_contract_sha256",
            "allowed_checkpoint_kinds",
            "allowed_checkpoint_updates",
            "required_agent_ids",
            "obs_dim",
            "action_dim",
            "hidden_dim",
        },
        context="checkpoint",
    )
    expected_checkpoint_schema = (
        LEGACY_CHECKPOINT_SCHEMA_VERSION if legacy else CHECKPOINT_SCHEMA_VERSION
    )
    expected_training_protocol = (
        LEGACY_TRAINING_PROTOCOL_VERSION if legacy else TRAINING_PROTOCOL_VERSION
    )
    expected_action_dim = 3 if legacy else 2
    expected_training_contract_id = (
        "P0_BASE_MAPPO_TOWN02_PILOT_V1"
        if legacy
        else "P0_BASE_MAPPO_TOWN02_ACTION_V2"
    )
    if (
        checkpoint["schema_version"] != expected_checkpoint_schema
        or checkpoint["training_protocol_version"] != expected_training_protocol
        or checkpoint["allowed_checkpoint_kinds"] != ["periodic"]
        or checkpoint["allowed_checkpoint_updates"] != [5, 10, 20, 30, 40, 60, 80, 100]
        or checkpoint["required_agent_ids"] != [0, 1]
        or checkpoint["obs_dim"] != 19
        or checkpoint["action_dim"] != expected_action_dim
        or checkpoint["hidden_dim"] != 128
        or checkpoint["required_training_contract_id"]
        != expected_training_contract_id
        or not SHA256_RE.fullmatch(str(checkpoint["required_training_contract_sha256"]))
    ):
        raise C0ProtocolError("checkpoint compatibility policy differs from the frozen baseline")

    profiles = payload["profiles"]
    if set(profiles) != {"smoke", "gate", "validation"}:
        raise C0ProtocolError("profiles must be exactly smoke, gate and validation")
    expected_counts = {"smoke": (1, 1), "gate": (16, 1), "validation": (8, 3)}
    seen_case_ids: set[tuple[str, int]] = set()
    profile_hashes: Dict[str, set[str]] = {}
    for profile_name, profile in profiles.items():
        _require_exact_fields(
            profile,
            {
                "purpose",
                "scenario_dir",
                "scenario_pool_sha256",
                "scenario_names",
                "episode_seeds",
                "formal_use",
            },
            context=f"profiles.{profile_name}",
        )
        scenario_dir = _inside(project_dir, profile["scenario_dir"], context=f"{profile_name}.scenario_dir")
        if not scenario_dir.is_dir():
            raise C0ProtocolError(f"scenario directory not found: {scenario_dir}")
        actual_pool_fingerprint = scenario_pool_fingerprint(scenario_dir)
        if profile["scenario_pool_sha256"] != actual_pool_fingerprint:
            raise C0ProtocolError(
                f"{profile_name} scenario pool fingerprint drift: "
                f"actual={actual_pool_fingerprint}"
            )
        names = profile["scenario_names"]
        seeds = profile["episode_seeds"]
        if not isinstance(names, list) or not isinstance(seeds, list):
            raise C0ProtocolError(f"{profile_name} names/seeds must be lists")
        if (len(names), len(seeds)) != expected_counts[profile_name]:
            raise C0ProtocolError(f"{profile_name} scenario/seed cardinality drift")
        if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
            raise C0ProtocolError(f"{profile_name} scenarios/seeds must be unique")
        hashes: set[str] = set()
        for name in names:
            if Path(name).name != name or not name.endswith(".json"):
                raise C0ProtocolError(f"invalid scenario name in {profile_name}: {name!r}")
            path = (scenario_dir / name).resolve()
            try:
                path.relative_to(scenario_dir.resolve())
            except ValueError as error:
                raise C0ProtocolError(f"scenario path escapes directory: {name}") from error
            scenario = load_strict_json(path)
            _validate_scenario(scenario, path=path)
            hashes.add(sha256_file(path))
            for seed in seeds:
                _positive_int(seed, context=f"{profile_name}.episode_seed")
                identity = (str(path), seed)
                if identity in seen_case_ids:
                    raise C0ProtocolError("scenario/seed pair is duplicated across profiles")
                seen_case_ids.add(identity)
        profile_hashes[profile_name] = hashes
        actual_names = sorted(path.name for path in scenario_dir.glob("*.json"))
        if profile_name in {"gate", "validation"} and sorted(names) != actual_names:
            raise C0ProtocolError(f"{profile_name} must enumerate its complete frozen scenario directory")
    if profile_hashes["gate"] & profile_hashes["validation"]:
        raise C0ProtocolError("validation scenarios must not duplicate training scenarios by content hash")

    outcomes = payload["outcomes"]
    _require_exact_fields(
        outcomes,
        {
            "success_reasons",
            "collision_reasons",
            "timeout_reasons",
            "other_reasons",
            "require_mutual_exclusivity",
            "allow_other_for_go",
        },
        context="outcomes",
    )
    reason_lists = [outcomes[f"{name}_reasons"] for name in OUTCOMES]
    if any(not isinstance(values, list) or not values for values in reason_lists):
        raise C0ProtocolError("each outcome must have at least one terminal reason")
    flat = [reason for values in reason_lists for reason in values]
    if len(flat) != len(set(flat)) or outcomes["require_mutual_exclusivity"] is not True:
        raise C0ProtocolError("outcome terminal reasons must be mutually exclusive")
    if outcomes["allow_other_for_go"] is not False:
        raise C0ProtocolError("other outcome cannot qualify for GO")

    outputs = payload["outputs"]
    required_outputs = {
        "evaluation_root",
        "manifest",
        "resolved_config",
        "episodes_jsonl",
        "summary_json",
        "summary_csv",
        "validator_report",
    }
    _require_exact_fields(outputs, required_outputs, context="outputs")
    for name, relative in outputs.items():
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts or not str(relative).strip():
            raise C0ProtocolError(f"outputs.{name} must be a safe relative path")

    repeatability = payload["repeatability"]
    _require_exact_fields(repeatability, {"required_repeats", "discrete_fields"}, context="repeatability")
    if _positive_int(repeatability["required_repeats"], context="repeatability.required_repeats") < 2:
        raise C0ProtocolError("repeatability requires at least two runs")
    if repeatability["discrete_fields"] != list(REPEATABILITY_FIELDS):
        raise C0ProtocolError("repeatability.discrete_fields differ from the frozen signature")

    critical = payload["critical_source_paths"]
    if not isinstance(critical, list) or not critical or len(critical) != len(set(critical)):
        raise C0ProtocolError("critical_source_paths must be a non-empty unique list")
    for relative in critical:
        if not _inside(project_dir, relative, context="critical_source_paths").exists():
            raise C0ProtocolError(f"critical source not found: {relative}")


def load_c0_contract(path: str | Path, *, project_dir: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(project_dir.resolve()).as_posix()
    except ValueError as error:
        raise C0ProtocolError("C0 contract must be inside the project directory") from error
    payload = load_strict_json(resolved)
    validate_c0_contract(payload, project_dir=project_dir)
    if relative not in payload["critical_source_paths"]:
        raise C0ProtocolError("active C0 contract must be listed in critical_source_paths")
    return {
        "payload": payload,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "canonical_sha256": canonical_json_sha256(payload),
    }


def expand_profile_cases(
    contract: Mapping[str, Any], profile_name: str, *, project_dir: Path
) -> list[Dict[str, Any]]:
    if profile_name not in contract["profiles"]:
        raise C0ProtocolError(f"unknown C0 profile: {profile_name}")
    profile = contract["profiles"][profile_name]
    scenario_dir = _inside(project_dir, profile["scenario_dir"], context="scenario_dir")
    cases: list[Dict[str, Any]] = []
    for scenario_name in profile["scenario_names"]:
        scenario_path = (scenario_dir / scenario_name).resolve()
        scenario = load_strict_json(scenario_path)
        for seed in profile["episode_seeds"]:
            ordinal = len(cases) + 1
            cases.append(
                {
                    "case_id": f"{profile_name}:{Path(scenario_name).stem}:s{seed}",
                    "case_ordinal": ordinal,
                    "scenario_name": scenario_name,
                    "scenario_path": str(scenario_path),
                    "scenario_sha256": sha256_file(scenario_path),
                    "scenario_family": scenario["family"],
                    "scenario_difficulty": int(scenario["difficulty"]),
                    "episode_seed": int(seed),
                    "scenario_config": scenario,
                }
            )
    return cases


def classify_outcome(episode_end_reason: str, outcomes: Mapping[str, Any]) -> str:
    matches = [name for name in OUTCOMES if episode_end_reason in outcomes[f"{name}_reasons"]]
    if len(matches) != 1:
        raise C0ProtocolError(
            f"episode_end_reason must map to exactly one outcome: {episode_end_reason!r}"
        )
    return matches[0]


def parse_eval_id(eval_id: str, *, profile: str, checkpoint_update: int) -> None:
    match = RUN_ID_RE.fullmatch(str(eval_id))
    if not match:
        raise C0ProtocolError(
            "eval_id must match P0_BASE_C0_<profile>_u<6digits>_YYYYMMDD_HHMMSS"
        )
    if match.group(1) != profile or int(match.group(2)) != int(checkpoint_update):
        raise C0ProtocolError("eval_id profile/update do not match the selected run inputs")


def load_and_validate_c0_checkpoint(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    allow_diagnostic_checkpoint: bool = False,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    checkpoint, digest = load_checkpoint_file(path, map_location=map_location)
    legacy = contract["protocol_version"] == LEGACY_C0_PROTOCOL_VERSION
    validate_checkpoint_header(checkpoint, allow_legacy=legacy)
    policy = contract["checkpoint"]
    mismatches: list[str] = []
    if checkpoint.get("checkpoint_schema_version") != policy["schema_version"]:
        mismatches.append("checkpoint schema")
    if checkpoint.get("training_protocol_version") != policy["training_protocol_version"]:
        mismatches.append("training protocol")
    expected_protocols = {
        "observation": OBSERVATION_PROTOCOL_VERSION,
        "action": (
            LEGACY_ACTION_PROTOCOL_VERSION
            if legacy
            else ACTION_PROTOCOL_VERSION
        ),
        "lifecycle": LIFECYCLE_PROTOCOL_VERSION,
        "reward": contract["environment"]["reward_version"],
        "metrics": (
            LEGACY_METRICS_PROTOCOL_VERSION
            if legacy
            else METRICS_PROTOCOL_VERSION
        ),
    }
    if checkpoint.get("protocol_versions") != expected_protocols:
        mismatches.append("protocol_versions")
    if not legacy and checkpoint.get("action_spec") != action_protocol_spec():
        mismatches.append("action_spec")
    model_spec = checkpoint.get("model_spec")
    expected_model = {
        "obs_dim": policy["obs_dim"],
        "action_dim": policy["action_dim"],
        "num_agents": len(policy["required_agent_ids"]),
        "agent_ids": policy["required_agent_ids"],
        "hidden_dim": policy["hidden_dim"],
        "critic_type": "AgentConditionedCritic",
    }
    if not isinstance(model_spec, Mapping) or any(model_spec.get(key) != value for key, value in expected_model.items()):
        mismatches.append("model_spec")
    actors = checkpoint.get("actors")
    if not isinstance(actors, Mapping) or {int(key) for key in actors} != set(policy["required_agent_ids"]):
        mismatches.append("actors")
    if mismatches:
        raise C0ProtocolError("checkpoint is incompatible with C0: " + ", ".join(mismatches))
    for agent_id in policy["required_agent_ids"]:
        state = actors.get(agent_id, actors.get(str(agent_id)))
        try:
            actor = Actor(
                obs_dim=policy["obs_dim"],
                act_dim=policy["action_dim"],
                hidden=policy["hidden_dim"],
            )
            actor.load_state_dict(state, strict=True)
        except Exception as error:
            raise C0ProtocolError(
                f"checkpoint actor state is incompatible for agent={agent_id}: {error}"
            ) from error

    formal_reasons: list[str] = []
    if checkpoint.get("checkpoint_kind") not in policy["allowed_checkpoint_kinds"]:
        formal_reasons.append("checkpoint_kind is not periodic")
    if int(checkpoint.get("update", -1)) not in policy["allowed_checkpoint_updates"]:
        formal_reasons.append("checkpoint update is not a registered P0-BASE milestone")
    metadata = checkpoint.get("run_metadata", {})
    if metadata.get("experiment_contract_id") != policy["required_training_contract_id"]:
        formal_reasons.append("training experiment contract id mismatch")
    if metadata.get("experiment_contract_sha256") != policy["required_training_contract_sha256"]:
        formal_reasons.append("training experiment contract SHA256 mismatch")
    if formal_reasons and not allow_diagnostic_checkpoint:
        raise C0ProtocolError(
            "checkpoint is not eligible for formal C0 evaluation: " + "; ".join(formal_reasons)
        )
    return {
        "checkpoint": checkpoint,
        "sha256": digest,
        "path": str(Path(path).expanduser().resolve()),
        "update": int(checkpoint["update"]),
        "formal_eligible": not formal_reasons,
        "formal_ineligibility_reasons": formal_reasons,
    }


class ScalarStats:
    """Small deterministic accumulator used by the simulator-facing runner."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.last = 0.0

    def add(self, value: Any, *, context: str) -> None:
        number = _finite(value, context=context)
        self.count += 1
        self.total += number
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.last = number

    def as_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "last": None}
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count,
            "last": self.last,
        }


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise C0ProtocolError(f"JSONL file not found: {source}")
    records: list[Dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise C0ProtocolError(f"blank JSONL line at {source}:{line_number}")
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    C0ProtocolError(f"non-finite JSON constant at line {line_number}: {token}")
                ),
            )
        except json.JSONDecodeError as error:
            raise C0ProtocolError(f"invalid JSONL at {source}:{line_number}: {error}") from error
        if not isinstance(payload, dict):
            raise C0ProtocolError(f"JSONL record at {source}:{line_number} is not an object")
        records.append(payload)
    return records


def _validate_stats(payload: Any, *, expected_count: int, context: str) -> None:
    _require_exact_fields(payload, {"count", "min", "max", "mean", "last"}, context=context)
    if payload["count"] != expected_count:
        raise C0ProtocolError(f"{context}.count differs from active decisions")
    if expected_count == 0:
        if any(payload[name] is not None for name in ("min", "max", "mean", "last")):
            raise C0ProtocolError(f"{context} empty stats must use null values")
    else:
        values = [_finite(payload[name], context=f"{context}.{name}") for name in ("min", "max", "mean", "last")]
        if values[0] > values[1] or not values[0] - 1e-12 <= values[2] <= values[1] + 1e-12:
            raise C0ProtocolError(f"{context} min/mean/max are inconsistent")


def validate_episode_record(
    record: Any,
    *,
    eval_id: str,
    profile: str,
    checkpoint_sha256: str,
    expected_case: Mapping[str, Any] | None = None,
) -> None:
    top = {
        "record_version",
        "eval_id",
        "profile",
        "case_id",
        "case_ordinal",
        "scenario_name",
        "scenario_sha256",
        "scenario_family",
        "scenario_difficulty",
        "episode_seed",
        "checkpoint_sha256",
        "started_at",
        "finished_at",
        "episode_length",
        "episode_end_reason",
        "outcome",
        "agents",
        "team",
    }
    _require_exact_fields(record, top, context="episode record")
    record_version = record["record_version"]
    if record_version not in {
        LEGACY_C0_EPISODE_RECORD_VERSION,
        C0_EPISODE_RECORD_VERSION,
    }:
        raise C0ProtocolError("episode record version mismatch")
    current_action = record_version == C0_EPISODE_RECORD_VERSION
    if record["eval_id"] != eval_id or record["profile"] != profile:
        raise C0ProtocolError("episode record evaluation identity mismatch")
    if record["checkpoint_sha256"] != checkpoint_sha256:
        raise C0ProtocolError("episode checkpoint SHA256 mismatch")
    if not SHA256_RE.fullmatch(str(record["scenario_sha256"])):
        raise C0ProtocolError("episode scenario SHA256 is invalid")
    episode_length = _positive_int(record["episode_length"], context="episode_length")
    if record["outcome"] not in OUTCOMES or not str(record["episode_end_reason"]).strip():
        raise C0ProtocolError("episode outcome/end reason is invalid")
    if not str(record["started_at"]).strip() or not str(record["finished_at"]).strip():
        raise C0ProtocolError("episode timestamps cannot be empty")
    if expected_case is not None:
        for key in (
            "case_id",
            "case_ordinal",
            "scenario_name",
            "scenario_sha256",
            "scenario_family",
            "scenario_difficulty",
            "episode_seed",
        ):
            if record[key] != expected_case[key]:
                raise C0ProtocolError(f"episode case identity mismatch for {key}")

    agents = record["agents"]
    if not isinstance(agents, Mapping) or set(agents) != {"0", "1"}:
        raise C0ProtocolError("episode agents must be exactly 0 and 1")
    expected_agent_fields = {
        "episode_return",
        "active_decision_steps",
        "reached_target",
        "collision",
        "final_done",
        "final_terminated",
        "final_truncated",
        "reward_component_sums",
        "reward_residual_max_abs",
        "diagnostics",
        "actions",
    }
    returns = []
    active_total = 0
    for agent_id in ("0", "1"):
        agent = agents[agent_id]
        _require_exact_fields(agent, expected_agent_fields, context=f"agent {agent_id}")
        active_steps = agent["active_decision_steps"]
        if isinstance(active_steps, bool) or not isinstance(active_steps, int) or not 0 <= active_steps <= episode_length:
            raise C0ProtocolError(f"agent {agent_id} active_decision_steps is invalid")
        if not all(isinstance(agent[name], bool) for name in ("reached_target", "collision", "final_done", "final_terminated", "final_truncated")):
            raise C0ProtocolError(f"agent {agent_id} lifecycle fields must be booleans")
        if agent["final_done"] != (agent["final_terminated"] or agent["final_truncated"]):
            raise C0ProtocolError(f"agent {agent_id} final done semantics are invalid")
        if agent["final_terminated"] and agent["final_truncated"]:
            raise C0ProtocolError(f"agent {agent_id} cannot terminate and truncate")
        episode_return = _finite(agent["episode_return"], context=f"agent {agent_id}.episode_return")
        residual = _finite(agent["reward_residual_max_abs"], context=f"agent {agent_id}.reward_residual")
        if residual > 1e-5:
            raise C0ProtocolError(f"agent {agent_id} reward residual exceeds tolerance")
        components = agent["reward_component_sums"]
        if not isinstance(components, Mapping) or set(components) != set(REWARD_COMPONENTS["v2"]):
            raise C0ProtocolError(f"agent {agent_id} reward component schema mismatch")
        component_sum = sum(_finite(value, context=f"agent {agent_id}.reward") for value in components.values())
        if not math.isclose(component_sum, episode_return, rel_tol=0.0, abs_tol=1e-5):
            raise C0ProtocolError(f"agent {agent_id} component sums do not conserve episode return")
        diagnostics = agent["diagnostics"]
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(REWARD_DIAGNOSTICS["v2"]):
            raise C0ProtocolError(f"agent {agent_id} diagnostic schema mismatch")
        for name, stats in diagnostics.items():
            _validate_stats(stats, expected_count=active_steps, context=f"agent {agent_id}.diagnostics.{name}")
        for name in (
            "collision_event",
            "lane_invasion_sensor_event",
            "success_event",
        ):
            stats = diagnostics[name]
            if stats["min"] is not None and (
                stats["min"] not in (0.0, 1.0) or stats["max"] not in (0.0, 1.0)
            ):
                raise C0ProtocolError(f"agent {agent_id} diagnostic {name} must be binary")
        collision_from_diagnostics = (
            diagnostics["collision_event"]["max"] is not None
            and diagnostics["collision_event"]["max"] > 0.0
        )
        if agent["collision"] != collision_from_diagnostics:
            raise C0ProtocolError(f"agent {agent_id} collision flag differs from diagnostics")
        success_from_diagnostics = (
            diagnostics["success_event"]["max"] is not None
            and diagnostics["success_event"]["max"] > 0.0
        )
        if agent["reached_target"] != success_from_diagnostics:
            raise C0ProtocolError(f"agent {agent_id} reached flag differs from success diagnostics")
        actions = agent["actions"]
        action_rate_fields = {f"{name}_rate" for name in ACTION_RATE_NAMES}
        policy_action_rate_fields = (
            {f"{name}_rate" for name in POLICY_ACTION_RATE_NAMES}
            if current_action
            else set()
        )
        policy_action_mean_fields = (
            {"policy_signed_longitudinal_mean", "policy_steer_mean"}
            if current_action
            else set()
        )
        _require_exact_fields(
            actions,
            {
                "count",
                "throttle_mean",
                "steer_mean",
                "brake_mean",
                *action_rate_fields,
                *policy_action_rate_fields,
                *policy_action_mean_fields,
            },
            context=f"agent {agent_id}.actions",
        )
        if actions["count"] != active_steps:
            raise C0ProtocolError(f"agent {agent_id} action count differs from active decisions")
        for name in (
            "throttle_mean",
            "steer_mean",
            "brake_mean",
            *sorted(action_rate_fields),
            *sorted(policy_action_rate_fields),
            *sorted(policy_action_mean_fields),
        ):
            value = actions[name]
            if active_steps == 0:
                if value is not None:
                    raise C0ProtocolError(f"agent {agent_id} empty action stats must be null")
            elif (
                name not in {"steer_mean", *policy_action_mean_fields}
                and not 0.0
                <= _finite(value, context=f"agent {agent_id}.actions.{name}")
                <= 1.0
            ):
                raise C0ProtocolError(f"agent {agent_id} action statistic outside [0,1]")
            elif (
                active_steps
                and name in {"steer_mean", *policy_action_mean_fields}
                and not -1.0
                <= _finite(value, context=f"agent {agent_id}.actions.{name}")
                <= 1.0
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} signed action mean outside [-1,1]"
                )
        if active_steps:
            def rate(name: str) -> float:
                return float(actions[f"{name}_rate"])

            if not math.isclose(
                rate("throttle_boundary"),
                rate("throttle_zero") + rate("throttle_saturation"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} throttle boundary decomposition is invalid"
                )
            if not math.isclose(
                rate("brake_boundary"),
                rate("brake_zero") + rate("brake_saturation"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} brake boundary decomposition is invalid"
                )
            if not math.isclose(
                rate("steer_boundary"),
                rate("steer_saturation"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} steer boundary decomposition is invalid"
                )
            mode_total = sum(
                rate(name)
                for name in (
                    "throttle_mode",
                    "brake_mode",
                    "coast_mode",
                    "longitudinal_conflict",
                )
            )
            if not math.isclose(mode_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise C0ProtocolError(
                    f"agent {agent_id} longitudinal mode rates do not conserve"
                )
            if rate("longitudinal_conflict") != 0.0:
                raise C0ProtocolError(
                    f"agent {agent_id} violates mutually exclusive longitudinal control"
                )
        returns.append(episode_return)
        active_total += active_steps

    team = record["team"]
    _require_exact_fields(team, {"mean_episode_return", "total_active_decision_steps"}, context="team")
    if not math.isclose(_finite(team["mean_episode_return"], context="team.mean_return"), sum(returns) / 2.0, rel_tol=0.0, abs_tol=1e-9):
        raise C0ProtocolError("team mean return is inconsistent")
    if team["total_active_decision_steps"] != active_total:
        raise C0ProtocolError("team active decision count is inconsistent")
    if record["outcome"] == "success" and not all(agent["reached_target"] for agent in agents.values()):
        raise C0ProtocolError("success outcome requires both agents to reach the goal")
    collision_flags = [agent["collision"] for agent in agents.values()]
    if record["outcome"] == "collision" and not any(collision_flags):
        raise C0ProtocolError("collision outcome requires at least one per-agent collision")
    if record["outcome"] != "collision" and any(collision_flags):
        raise C0ProtocolError("per-agent collision cannot appear under a non-collision outcome")


def _aggregate_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(records)
    outcome_counts = Counter(record["outcome"] for record in records)
    return {
        "case_count": count,
        "outcome_counts": {name: outcome_counts.get(name, 0) for name in OUTCOMES},
        "outcome_rates": {
            name: outcome_counts.get(name, 0) / count if count else 0.0 for name in OUTCOMES
        },
        "episode_length": {
            "min": min(record["episode_length"] for record in records) if count else None,
            "max": max(record["episode_length"] for record in records) if count else None,
            "mean": sum(record["episode_length"] for record in records) / count if count else None,
        },
        "mean_team_return": (
            sum(record["team"]["mean_episode_return"] for record in records) / count
            if count
            else None
        ),
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    eval_id: str,
    profile: str,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    if not records:
        raise C0ProtocolError("cannot summarize an empty C0 evaluation")
    record_versions = {record.get("record_version") for record in records}
    if len(record_versions) != 1:
        raise C0ProtocolError("cannot mix legacy and current C0 episode records")
    record_version = next(iter(record_versions))
    if record_version == LEGACY_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = LEGACY_C0_PROTOCOL_VERSION
    elif record_version == C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = C0_PROTOCOL_VERSION
    else:
        raise C0ProtocolError("cannot summarize an unknown C0 episode version")
    overall = _aggregate_group(records)
    by_family: Dict[str, Any] = {}
    for family in sorted({str(record["scenario_family"]) for record in records}):
        by_family[family] = _aggregate_group([record for record in records if record["scenario_family"] == family])
    by_scenario: Dict[str, Any] = {}
    for scenario in sorted({str(record["scenario_name"]) for record in records}):
        by_scenario[scenario] = _aggregate_group([record for record in records if record["scenario_name"] == scenario])
    agents: Dict[str, Any] = {}
    for agent_id in ("0", "1"):
        agents[agent_id] = {
            "reached_count": sum(bool(record["agents"][agent_id]["reached_target"]) for record in records),
            "collision_episode_count": sum(
                record["agents"][agent_id]["diagnostics"]["collision_event"]["max"] not in (None, 0.0)
                for record in records
            ),
            "mean_episode_return": sum(record["agents"][agent_id]["episode_return"] for record in records) / len(records),
            "mean_active_decision_steps": sum(record["agents"][agent_id]["active_decision_steps"] for record in records) / len(records),
            "mean_reward_component_sum": {
                component: sum(record["agents"][agent_id]["reward_component_sums"][component] for record in records) / len(records)
                for component in REWARD_COMPONENTS["v2"]
            },
        }
    return {
        "summary_version": C0_SUMMARY_VERSION,
        "eval_protocol_version": eval_protocol_version,
        "eval_id": eval_id,
        "profile": profile,
        "checkpoint_sha256": checkpoint_sha256,
        "case_count": len(records),
        "overall": overall,
        "agents": agents,
        "by_family": by_family,
        "by_scenario": by_scenario,
    }


def summary_csv_rows(summary: Mapping[str, Any]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    groups = [("overall", "all", summary["overall"])]
    groups.extend(("family", name, value) for name, value in summary["by_family"].items())
    groups.extend(("scenario", name, value) for name, value in summary["by_scenario"].items())
    for scope, name, group in groups:
        counts = group["outcome_counts"]
        rates = group["outcome_rates"]
        rows.append(
            {
                "scope": scope,
                "name": name,
                "case_count": str(group["case_count"]),
                "success_count": str(counts["success"]),
                "collision_count": str(counts["collision"]),
                "timeout_count": str(counts["timeout"]),
                "other_count": str(counts["other"]),
                "success_rate": format(rates["success"], ".12g"),
                "collision_rate": format(rates["collision"], ".12g"),
                "timeout_rate": format(rates["timeout"], ".12g"),
                "other_rate": format(rates["other"], ".12g"),
                "mean_episode_length": format(group["episode_length"]["mean"], ".12g"),
                "mean_team_return": format(group["mean_team_return"], ".12g"),
            }
        )
    return rows


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_summary_csv(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUMMARY_CSV_FIELDS:
            raise C0ProtocolError("summary CSV schema mismatch")
        return [dict(row) for row in reader]


def ensure_completed_prefix(records: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> None:
    if len(records) > len(cases):
        raise C0ProtocolError("resume evidence contains more episodes than the frozen case set")
    seen: set[str] = set()
    for index, record in enumerate(records):
        case_id = str(record.get("case_id", ""))
        if case_id in seen or case_id != cases[index]["case_id"]:
            raise C0ProtocolError("resume episodes must be a unique exact prefix of the frozen case order")
        seen.add(case_id)


def repeatability_signature(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the frozen discrete signature used for same-input rerun checks."""
    return {
        "case_id": record["case_id"],
        "outcome": record["outcome"],
        "episode_end_reason": record["episode_end_reason"],
        "episode_length": record["episode_length"],
        "agent_0_reached": record["agents"]["0"]["reached_target"],
        "agent_1_reached": record["agents"]["1"]["reached_target"],
    }


def compare_repeatability_records(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    configured_fields = contract["repeatability"]["discrete_fields"]
    if configured_fields != list(REPEATABILITY_FIELDS):
        raise C0ProtocolError("repeatability fields differ from the implemented discrete signature")
    if len(left) != len(right):
        raise C0ProtocolError("repeatability runs have different case counts")
    mismatches = []
    for index, (left_record, right_record) in enumerate(zip(left, right)):
        if left_record.get("checkpoint_sha256") != right_record.get("checkpoint_sha256"):
            raise C0ProtocolError("repeatability runs use different checkpoints")
        if left_record.get("profile") != right_record.get("profile"):
            raise C0ProtocolError("repeatability runs use different profiles")
        left_signature = repeatability_signature(left_record)
        right_signature = repeatability_signature(right_record)
        if left_signature != right_signature:
            mismatches.append(
                {
                    "case_ordinal": index + 1,
                    "case_id": left_record.get("case_id", ""),
                    "left": left_signature,
                    "right": right_signature,
                }
            )
    return {
        "status": "PASS" if not mismatches else "HOLD",
        "case_count": len(left),
        "discrete_fields": configured_fields,
        "mismatches": mismatches,
    }


def atomic_update_manifest(path: str | Path, updates: Mapping[str, Any]) -> Dict[str, Any]:
    destination = Path(path)
    payload: MutableMapping[str, Any] = {}
    if destination.is_file():
        loaded = load_strict_json(destination)
        if not isinstance(loaded, dict):
            raise C0ProtocolError("evaluation manifest must be an object")
        payload.update(loaded)
    payload.update(updates)
    atomic_write_json(dict(payload), destination)
    return dict(payload)
