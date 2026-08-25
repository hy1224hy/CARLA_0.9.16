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
    ACTION_CHECKPOINT_SCHEMA_VERSION,
    ACTION_TRAINING_PROTOCOL_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    OFFROAD_CHECKPOINT_SCHEMA_VERSION,
    OFFROAD_TRAINING_PROTOCOL_VERSION,
    VALUENORM_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_TRAINING_PROTOCOL_VERSION,
    LIFECYCLE_PROTOCOL_VERSION,
    PRE_OFFROAD_LIFECYCLE_PROTOCOL_VERSION,
    OBSERVATION_PROTOCOL_VERSION,
    R6_TIME_AWARE_OBSERVATION_PROTOCOL_VERSION,
    R7_CHECKPOINT_SCHEMA_VERSION,
    R7_TRAINING_PROTOCOL_VERSION,
    TOWN05_BASE_OBSERVATION_PROTOCOL_VERSION,
    TRAINING_PROTOCOL_VERSION,
    VALUENORM_TRAINING_PROTOCOL_VERSION,
    atomic_write_json,
    load_checkpoint_file,
    scenario_pool_fingerprint,
    sha256_file,
    validate_checkpoint_header,
)
from training.metrics_protocol import (
    ACTION_METRICS_PROTOCOL_VERSION,
    ACTION_RATE_NAMES,
    LEGACY_METRICS_PROTOCOL_VERSION,
    METRICS_PROTOCOL_VERSION,
    OFFROAD_METRICS_PROTOCOL_VERSION,
    PRE_OFFROAD_METRICS_PROTOCOL_VERSION,
    POLICY_ACTION_RATE_NAMES,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
    R7_METRICS_PROTOCOL_VERSION,
)
from environment.offroad_protocol import (
    OFFROAD_PROTOCOL_VERSION,
    R6_OFFROAD_PROTOCOL_VERSION,
    offroad_protocol_spec,
    r6_offroad_protocol_spec,
)
from training.value_normalization import (
    VALUE_LOSS_PROTOCOL_VERSION,
    VALUE_NORMALIZATION_PROTOCOL_VERSION,
)
from training.mastery_curriculum import (
    CURRICULUM_PROTOCOL_VERSION,
    R7_CURRICULUM_PROTOCOL_VERSION,
)
from training.policy_distribution_protocol import (
    POLICY_DISTRIBUTION_PROTOCOL_VERSION,
    R7_POLICY_DISTRIBUTION_PROTOCOL_VERSION,
)
from environment.stall_protocol import STALL_PROTOCOL_VERSION
from environment.road_legality_protocol import (
    ROAD_LEGALITY_PROTOCOL_VERSION,
    road_legality_protocol_spec,
)
from training.critic_diagnostics import (
    CRITIC_DIAGNOSTICS_PROTOCOL_VERSION,
    validate_critic_probe_batch,
)
from training.learning_rate_protocol import LEARNING_RATE_PROTOCOL_VERSION


LEGACY_C0_PROTOCOL_VERSION = "p0_base_c0_eval_v1"
ACTION_C0_PROTOCOL_VERSION = "p0_base_c0_eval_v2"
REWARD_C0_PROTOCOL_VERSION = "p0_base_c0_eval_v3"
C0_PROTOCOL_VERSION = "p0_base_c0_eval_v4"
TOWN05_C0_PROTOCOL_VERSION = "p0_base_c0_eval_town05_v1"
TOWN05_OFFROAD_C0_PROTOCOL_VERSION = "p0_base_c0_eval_town05_v2"
TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION = (
    "p0_base_c0_eval_town05_v3"
)
TOWN05_R6_C0_PROTOCOL_VERSION = "p0_base_c0_eval_town05_r6_v4"
TOWN05_R7_C0_PROTOCOL_VERSION = "p0_base_c0_eval_town05_r7_v5"
LEGACY_C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v2"
ACTION_C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v3"
REWARD_C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v4"
C0_EPISODE_RECORD_VERSION = "p0_base_c0_episode_v5"
TOWN05_C0_EPISODE_RECORD_VERSION = "p0_base_c0_town05_episode_v1"
TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION = (
    "p0_base_c0_town05_offroad_episode_v2"
)
TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION = (
    "p0_base_c0_town05_offroad_episode_v3"
)
TOWN05_R6_C0_EPISODE_RECORD_VERSION = "p0_base_c0_town05_r6_episode_v4"
TOWN05_R7_C0_EPISODE_RECORD_VERSION = "p0_base_c0_town05_r7_episode_v5"
C0_SUMMARY_VERSION = "p0_base_c0_summary_v1"
TOWN05_R7_C0_SUMMARY_VERSION = "p0_base_c0_town05_r7_summary_v2"
C0_MANIFEST_VERSION = "p0_base_c0_manifest_v1"
C0_RESOLVED_CONFIG_VERSION = "p0_base_c0_resolved_v1"
C0_VALIDATOR_REPORT_VERSION = "p0_base_c0_validator_v1"
AGENT_IDS = (0, 1)
OUTCOMES = ("success", "collision", "timeout", "other")
R7_OUTCOMES = (
    "success",
    "collision",
    "offroad",
    "road_violation",
    "timeout",
)
REPEATABILITY_FIELDS = (
    "case_id",
    "outcome",
    "episode_end_reason",
    "episode_length",
    "agent_0_reached",
    "agent_1_reached",
)
OUTCOME_REPEATABILITY_MODE = "outcome_strict_tick_tolerant_v1"
OUTCOME_REPEATABILITY_HARD_FIELDS = (
    "case_id",
    "outcome",
    "episode_end_reason",
    "agent_0_reached",
    "agent_0_collision",
    "agent_0_offroad_terminated",
    "agent_0_final_done",
    "agent_0_final_terminated",
    "agent_0_final_truncated",
    "agent_0_first_offroad_evidence_present",
    "agent_0_lifecycle_transition",
    "agent_1_reached",
    "agent_1_collision",
    "agent_1_offroad_terminated",
    "agent_1_final_done",
    "agent_1_final_terminated",
    "agent_1_final_truncated",
    "agent_1_first_offroad_evidence_present",
    "agent_1_lifecycle_transition",
)
OUTCOME_REPEATABILITY_TOLERANT_FIELDS = (
    "episode_length",
    "agent_0_active_decision_steps",
    "agent_1_active_decision_steps",
)
OUTCOME_REPEATABILITY_TOLERANCES = {
    "episode_length": 2,
    "agent_0_active_decision_steps": 2,
    "agent_1_active_decision_steps": 2,
}
OUTCOME_REPEATABILITY_CONTINUOUS_FIELDS = (
    "agent_0_episode_return",
    "agent_1_episode_return",
)
OUTCOME_REPEATABILITY_CONTINUOUS_PASS_ABS_TOLERANCE = {
    "agent_0_episode_return": 1e-5,
    "agent_1_episode_return": 1e-5,
}
OUTCOME_REPEATABILITY_CONTINUOUS_HOLD_ABS_TOLERANCE = {
    "agent_0_episode_return": 0.1,
    "agent_1_episode_return": 0.1,
}
OUTCOME_REPEATABILITY_MAX_WARNING_CASE_FRACTION = 0.5
OUTCOME_REPEATABILITY_SYSTEMATIC_MIN_CASE_COUNT = 8
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
R7_SUMMARY_CSV_FIELDS = (
    "scope",
    "name",
    "case_count",
    "success_count",
    "collision_count",
    "offroad_count",
    "road_violation_count",
    "timeout_count",
    "success_rate",
    "collision_rate",
    "offroad_rate",
    "road_violation_rate",
    "timeout_rate",
    "mean_episode_length",
    "mean_team_return",
)

ROAD_LEGALITY_EVIDENCE_FIELDS = {
    "classification",
    "reference_lane",
    "legal_lane_change",
    "lane_deviation_exempted",
    "violation_reason",
    "violation_raw",
    "consecutive_violation_ticks",
    "violation_confirmed",
    "legal_transition_grace_remaining",
    "success_eligible",
    "update_reference_lane",
    "same_reference_lane",
    "adjacent_to_reference",
    "lane_change_permitted",
    "current_to_reference_cosine",
    "vehicle_to_current_cosine",
    "in_junction",
}


class C0ProtocolError(RuntimeError):
    """Raised when a C0 input or artifact violates the frozen protocol."""


def outcome_names_for_protocol(protocol_version: str) -> tuple[str, ...]:
    """Select the frozen outcome vocabulary without changing historical runs."""

    return (
        R7_OUTCOMES
        if str(protocol_version) == TOWN05_R7_C0_PROTOCOL_VERSION
        else OUTCOMES
    )


def outcome_names_for_record(record_version: str) -> tuple[str, ...]:
    return (
        R7_OUTCOMES
        if str(record_version) == TOWN05_R7_C0_EPISODE_RECORD_VERSION
        else OUTCOMES
    )


def outcome_names_for_mapping(outcomes: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        R7_OUTCOMES
        if "road_violation_reasons" in outcomes
        else OUTCOMES
    )


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


def _validate_scenario(payload: Any, *, path: Path, expected_town: str) -> None:
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
    optional = {"goal_mode", "agent_goal_zones"}
    if not isinstance(payload, Mapping):
        raise C0ProtocolError(f"scenario {path.name} must be a mapping")
    missing = required - set(payload)
    extra = set(payload) - required - optional
    if missing or extra:
        raise C0ProtocolError(
            f"scenario {path.name} fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if payload["schema_version"] != 3 or payload["town"] != expected_town:
        raise C0ProtocolError(
            f"scenario {path.name} must use schema 3 and {expected_town}"
        )
    if payload["difficulty"] not in (1, 2, 3, 4):
        raise C0ProtocolError(f"scenario {path.name} has invalid difficulty")
    if not str(payload["family"]).strip():
        raise C0ProtocolError(f"scenario {path.name} has empty family")
    if not isinstance(payload["tags"], list) or not payload["tags"]:
        raise C0ProtocolError(f"scenario {path.name} must have tags")
    if set(payload["agents"]) != {"agent0", "agent1"}:
        raise C0ProtocolError(f"scenario {path.name} must define agent0 and agent1")
    if "agent_goal_zones" in payload and set(payload["agent_goal_zones"]) != {
        "agent0",
        "agent1",
    }:
        raise C0ProtocolError(
            f"scenario {path.name} agent_goal_zones must define agent0 and agent1"
        )
    goal_mode = payload.get("goal_mode")
    if goal_mode not in (None, "preserve_start_lane"):
        raise C0ProtocolError(f"scenario {path.name} has unsupported goal_mode")
    if expected_town == "Town05":
        if payload["family"] == "required_merge" and goal_mode is not None:
            raise C0ProtocolError(
                f"scenario {path.name} merge task must use the shared target lane"
            )
        if payload["family"] != "required_merge" and goal_mode != "preserve_start_lane":
            raise C0ProtocolError(
                f"scenario {path.name} lane-holding task needs preserve_start_lane"
            )


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
        ACTION_C0_PROTOCOL_VERSION,
        REWARD_C0_PROTOCOL_VERSION,
        C0_PROTOCOL_VERSION,
        TOWN05_C0_PROTOCOL_VERSION,
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }:
        raise C0ProtocolError("C0 protocol version mismatch")
    legacy = protocol_version == LEGACY_C0_PROTOCOL_VERSION
    reward_v2 = protocol_version in {
        LEGACY_C0_PROTOCOL_VERSION,
        ACTION_C0_PROTOCOL_VERSION,
    }
    expected_contract_id = {
        LEGACY_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN02_V1",
        ACTION_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN02_V2",
        REWARD_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN02_REWARD_V3",
        C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN02_CRITIC_V4",
        TOWN05_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN05_V1",
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN05_V2",
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION: (
            "P0_BASE_C0_TOWN05_V3"
        ),
        TOWN05_R6_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN05_R6_V4",
        TOWN05_R7_C0_PROTOCOL_VERSION: "P0_BASE_C0_TOWN05_R7_V5",
    }[protocol_version]
    if payload["contract_id"] != expected_contract_id:
        raise C0ProtocolError("C0 contract_id mismatch")
    if payload["status"] != "frozen" or payload["run_id_prefix"] != "P0_BASE_C0_":
        raise C0ProtocolError("C0 contract must be frozen with the registered prefix")

    env = payload["environment"]
    new_offroad_protocol = protocol_version in {
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }
    r6_protocol = protocol_version == TOWN05_R6_C0_PROTOCOL_VERSION
    r7_protocol = protocol_version == TOWN05_R7_C0_PROTOCOL_VERSION
    environment_fields = {
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
    }
    if new_offroad_protocol:
        environment_fields.update(
            {"lidar_range_m", "ttc_max_range_m", "offroad_protocol"}
        )
    if r7_protocol:
        environment_fields.add("road_legality_protocol")
    _require_exact_fields(
        env,
        environment_fields,
        context="environment",
    )
    if (
        env["carla_version"] != "0.9.16"
        or env["town"] not in {"Town02", "Town05"}
        or (
            protocol_version in {
                TOWN05_C0_PROTOCOL_VERSION,
                TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
                TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
                TOWN05_R6_C0_PROTOCOL_VERSION,
                TOWN05_R7_C0_PROTOCOL_VERSION,
            }
            and env["town"] != "Town05"
        )
        or (
            protocol_version
            not in {
                TOWN05_C0_PROTOCOL_VERSION,
                TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
                TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
                TOWN05_R6_C0_PROTOCOL_VERSION,
                TOWN05_R7_C0_PROTOCOL_VERSION,
            }
            and env["town"] != "Town02"
        )
        or env["num_agents"] != 2
        or env["reward_version"] != (
            "v5"
            if r7_protocol
            else "v4"
            if r6_protocol
            else ("v2" if reward_v2 else "v3")
        )
        or env["adversarial_agent_ids"] != []
        or env["allow_extra_traffic"] is not False
    ):
        raise C0ProtocolError(
            "C0 environment must match the registered two-agent town without traffic/adversary"
        )
    if new_offroad_protocol:
        expected_offroad_spec = (
            r6_offroad_protocol_spec()
            if (r6_protocol or r7_protocol)
            else offroad_protocol_spec()
        )
        if env["offroad_protocol"] != expected_offroad_spec:
            raise C0ProtocolError("C0 offroad protocol differs from training")
        for name in ("lidar_range_m", "ttc_max_range_m"):
            if _finite(env[name], context=f"environment.{name}") <= 0.0:
                raise C0ProtocolError(f"environment.{name} must be positive")
    if r7_protocol and env["road_legality_protocol"] != road_legality_protocol_spec():
        raise C0ProtocolError("C0 road-legality protocol differs from R7 training")
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
            "observation_protocol_version": (
                R6_TIME_AWARE_OBSERVATION_PROTOCOL_VERSION
                if (r6_protocol or r7_protocol)
                else TOWN05_BASE_OBSERVATION_PROTOCOL_VERSION
                if new_offroad_protocol
                else OBSERVATION_PROTOCOL_VERSION
            ),
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
    checkpoint_fields = {
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
    }
    if r7_protocol:
        checkpoint_fields.add("state_dim")
    _require_exact_fields(
        checkpoint,
        checkpoint_fields,
        context="checkpoint",
    )
    if legacy:
        expected_checkpoint_schema = LEGACY_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = LEGACY_TRAINING_PROTOCOL_VERSION
    elif protocol_version in {ACTION_C0_PROTOCOL_VERSION, REWARD_C0_PROTOCOL_VERSION}:
        expected_checkpoint_schema = ACTION_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = ACTION_TRAINING_PROTOCOL_VERSION
    elif r7_protocol:
        expected_checkpoint_schema = R7_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = R7_TRAINING_PROTOCOL_VERSION
    elif r6_protocol:
        expected_checkpoint_schema = CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = TRAINING_PROTOCOL_VERSION
    elif new_offroad_protocol:
        expected_checkpoint_schema = OFFROAD_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = OFFROAD_TRAINING_PROTOCOL_VERSION
    else:
        expected_checkpoint_schema = VALUENORM_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = VALUENORM_TRAINING_PROTOCOL_VERSION
    expected_action_dim = 3 if legacy else 2
    expected_training_contract_id = {
        LEGACY_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN02_PILOT_V1",
        ACTION_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN02_ACTION_V2",
        REWARD_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN02_REWARD_V3",
        C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN02_CRITIC_V4",
        TOWN05_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN05_V1",
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN05_V2",
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION: (
            "P0_BASE_MAPPO_TOWN05_V3"
        ),
        TOWN05_R6_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN05_R6_V4",
        TOWN05_R7_C0_PROTOCOL_VERSION: "P0_BASE_MAPPO_TOWN05_R7_V5",
    }[protocol_version]
    if (
        checkpoint["schema_version"] != expected_checkpoint_schema
        or checkpoint["training_protocol_version"] != expected_training_protocol
        or checkpoint["allowed_checkpoint_kinds"] != ["periodic"]
        or checkpoint["allowed_checkpoint_updates"]
        != (
            [5, 10, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500, 600]
            if r7_protocol
            else [5, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160]
            if new_offroad_protocol
            else [5, 10, 20, 30, 40, 60, 80, 100]
        )
        or checkpoint["required_agent_ids"] != [0, 1]
        or checkpoint["obs_dim"] != (20 if (r6_protocol or r7_protocol) else 19)
        or checkpoint["action_dim"] != expected_action_dim
        or checkpoint["hidden_dim"] != 128
        or checkpoint["required_training_contract_id"]
        != expected_training_contract_id
        or not SHA256_RE.fullmatch(str(checkpoint["required_training_contract_sha256"]))
        or (r7_protocol and checkpoint["state_dim"] != 12)
    ):
        raise C0ProtocolError("checkpoint compatibility policy differs from the frozen baseline")

    profiles = payload["profiles"]
    if set(profiles) != {"smoke", "gate", "validation"}:
        raise C0ProtocolError("profiles must be exactly smoke, gate and validation")
    expected_counts = {
        "smoke": (1, 1),
        "gate": (32 if new_offroad_protocol else 16, 1),
        "validation": (8, 3),
    }
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
            _validate_scenario(
                scenario, path=path, expected_town=str(env["town"])
            )
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
    outcome_names = outcome_names_for_protocol(protocol_version)
    expected_outcome_fields = {
        *(f"{name}_reasons" for name in outcome_names),
        "require_mutual_exclusivity",
        "allow_unclassified" if r7_protocol else "allow_other_for_go",
    }
    _require_exact_fields(
        outcomes,
        expected_outcome_fields,
        context="outcomes",
    )
    reason_lists = [outcomes[f"{name}_reasons"] for name in outcome_names]
    if any(not isinstance(values, list) or not values for values in reason_lists):
        raise C0ProtocolError("each outcome must have at least one terminal reason")
    flat = [reason for values in reason_lists for reason in values]
    if len(flat) != len(set(flat)) or outcomes["require_mutual_exclusivity"] is not True:
        raise C0ProtocolError("outcome terminal reasons must be mutually exclusive")
    if r7_protocol and outcomes["allow_unclassified"] is not False:
        raise C0ProtocolError("R7 formal outcomes cannot leave a terminal reason unclassified")
    if not r7_protocol and outcomes["allow_other_for_go"] is not False:
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
    outcome_repeatability = (
        protocol_version in {
            TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
            TOWN05_R6_C0_PROTOCOL_VERSION,
            TOWN05_R7_C0_PROTOCOL_VERSION,
        }
    )
    repeatability_fields = (
        {
            "required_repeats",
            "comparison_mode",
            "hard_fields",
            "tolerant_fields",
            "absolute_tolerance_ticks",
            "continuous_fields",
            "continuous_pass_abs_tolerance",
            "continuous_hold_abs_tolerance",
            "max_warning_case_fraction",
            "systematic_drift_min_case_count",
        }
        if outcome_repeatability
        else {"required_repeats", "discrete_fields"}
    )
    _require_exact_fields(
        repeatability,
        repeatability_fields,
        context="repeatability",
    )
    if _positive_int(repeatability["required_repeats"], context="repeatability.required_repeats") < 2:
        raise C0ProtocolError("repeatability requires at least two runs")
    if outcome_repeatability:
        if repeatability["comparison_mode"] != OUTCOME_REPEATABILITY_MODE:
            raise C0ProtocolError("repeatability.comparison_mode differs")
        if repeatability["hard_fields"] != list(
            OUTCOME_REPEATABILITY_HARD_FIELDS
        ):
            raise C0ProtocolError("repeatability.hard_fields differ")
        if repeatability["tolerant_fields"] != list(
            OUTCOME_REPEATABILITY_TOLERANT_FIELDS
        ):
            raise C0ProtocolError("repeatability.tolerant_fields differ")
        if repeatability["absolute_tolerance_ticks"] != (
            OUTCOME_REPEATABILITY_TOLERANCES
        ):
            raise C0ProtocolError(
                "repeatability.absolute_tolerance_ticks differ"
            )
        if repeatability["continuous_fields"] != list(
            OUTCOME_REPEATABILITY_CONTINUOUS_FIELDS
        ):
            raise C0ProtocolError("repeatability.continuous_fields differ")
        if repeatability["continuous_pass_abs_tolerance"] != (
            OUTCOME_REPEATABILITY_CONTINUOUS_PASS_ABS_TOLERANCE
        ):
            raise C0ProtocolError(
                "repeatability.continuous_pass_abs_tolerance differ"
            )
        if repeatability["continuous_hold_abs_tolerance"] != (
            OUTCOME_REPEATABILITY_CONTINUOUS_HOLD_ABS_TOLERANCE
        ):
            raise C0ProtocolError(
                "repeatability.continuous_hold_abs_tolerance differ"
            )
        if repeatability["max_warning_case_fraction"] != (
            OUTCOME_REPEATABILITY_MAX_WARNING_CASE_FRACTION
        ):
            raise C0ProtocolError(
                "repeatability.max_warning_case_fraction differ"
            )
        if repeatability["systematic_drift_min_case_count"] != (
            OUTCOME_REPEATABILITY_SYSTEMATIC_MIN_CASE_COUNT
        ):
            raise C0ProtocolError(
                "repeatability.systematic_drift_min_case_count differ"
            )
    elif repeatability["discrete_fields"] != list(REPEATABILITY_FIELDS):
        raise C0ProtocolError(
            "repeatability.discrete_fields differ from the frozen signature"
        )

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
    matches = [
        name
        for name in outcome_names_for_mapping(outcomes)
        if episode_end_reason in outcomes[f"{name}_reasons"]
    ]
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
    protocol_version = contract["protocol_version"]
    legacy = protocol_version == LEGACY_C0_PROTOCOL_VERSION
    new_offroad_protocol = protocol_version in {
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }
    r6_protocol = protocol_version == TOWN05_R6_C0_PROTOCOL_VERSION
    r7_protocol = protocol_version == TOWN05_R7_C0_PROTOCOL_VERSION
    historical = not (r6_protocol or r7_protocol)
    validate_checkpoint_header(checkpoint, allow_legacy=historical)
    policy = contract["checkpoint"]
    mismatches: list[str] = []
    if checkpoint.get("checkpoint_schema_version") != policy["schema_version"]:
        mismatches.append("checkpoint schema")
    if checkpoint.get("training_protocol_version") != policy["training_protocol_version"]:
        mismatches.append("training protocol")
    expected_protocols = {
        "observation": (
            R6_TIME_AWARE_OBSERVATION_PROTOCOL_VERSION
            if (r6_protocol or r7_protocol)
            else TOWN05_BASE_OBSERVATION_PROTOCOL_VERSION
            if new_offroad_protocol
            else OBSERVATION_PROTOCOL_VERSION
        ),
        "action": (
            LEGACY_ACTION_PROTOCOL_VERSION
            if legacy
            else ACTION_PROTOCOL_VERSION
        ),
        "lifecycle": (
            LIFECYCLE_PROTOCOL_VERSION
            if new_offroad_protocol
            else PRE_OFFROAD_LIFECYCLE_PROTOCOL_VERSION
        ),
        "reward": contract["environment"]["reward_version"],
        "metrics": (
            LEGACY_METRICS_PROTOCOL_VERSION
            if legacy
            else (
                R7_METRICS_PROTOCOL_VERSION
                if r7_protocol
                else METRICS_PROTOCOL_VERSION
                if r6_protocol
                else OFFROAD_METRICS_PROTOCOL_VERSION
                if new_offroad_protocol
                else (
                    PRE_OFFROAD_METRICS_PROTOCOL_VERSION
                    if protocol_version
                    in {C0_PROTOCOL_VERSION, TOWN05_C0_PROTOCOL_VERSION}
                    else ACTION_METRICS_PROTOCOL_VERSION
                )
            )
        ),
    }
    if new_offroad_protocol:
        expected_protocols["offroad"] = (
            R6_OFFROAD_PROTOCOL_VERSION
            if (r6_protocol or r7_protocol) else OFFROAD_PROTOCOL_VERSION
        )
    if r6_protocol:
        expected_protocols.update({
            "policy_distribution": POLICY_DISTRIBUTION_PROTOCOL_VERSION,
            "curriculum": CURRICULUM_PROTOCOL_VERSION,
            "stall": STALL_PROTOCOL_VERSION,
        })
    if r7_protocol:
        expected_protocols.update({
            "policy_distribution": R7_POLICY_DISTRIBUTION_PROTOCOL_VERSION,
            "curriculum": R7_CURRICULUM_PROTOCOL_VERSION,
            "stall": STALL_PROTOCOL_VERSION,
            "road_legality": ROAD_LEGALITY_PROTOCOL_VERSION,
            "critic_diagnostics": CRITIC_DIAGNOSTICS_PROTOCOL_VERSION,
            "learning_rate": LEARNING_RATE_PROTOCOL_VERSION,
        })
    if protocol_version in {
        C0_PROTOCOL_VERSION,
        TOWN05_C0_PROTOCOL_VERSION,
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }:
        expected_protocols.update(
            {
                "value_normalization": VALUE_NORMALIZATION_PROTOCOL_VERSION,
                "value_loss": VALUE_LOSS_PROTOCOL_VERSION,
            }
        )
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
    if not historical:
        expected_model["critic_output_semantics"] = "normalized_value"
    if r7_protocol:
        expected_model["state_dim"] = int(policy["state_dim"])
    if not isinstance(model_spec, Mapping) or any(model_spec.get(key) != value for key, value in expected_model.items()):
        mismatches.append("model_spec")
    if protocol_version in {
        TOWN05_C0_PROTOCOL_VERSION,
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }:
        expected_gamma = float(contract["environment"]["reward_gamma"])
        training_config = checkpoint.get("training_config", {})
        metadata = checkpoint.get("run_metadata", {})
        runtime = metadata.get("runtime", {}) if isinstance(metadata, Mapping) else {}
        if (
            not isinstance(training_config, Mapping)
            or training_config.get("gamma") != expected_gamma
        ):
            mismatches.append("training_config.gamma")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("environment_reward_gamma") != expected_gamma
        ):
            mismatches.append("run_metadata.runtime.environment_reward_gamma")
        if new_offroad_protocol and runtime.get(
            "offroad_protocol"
        ) != (
            r6_offroad_protocol_spec()
            if (r6_protocol or r7_protocol)
            else offroad_protocol_spec()
        ):
            mismatches.append("run_metadata.runtime.offroad_protocol")
        if r7_protocol and runtime.get(
            "road_legality_protocol"
        ) != road_legality_protocol_spec():
            mismatches.append("run_metadata.runtime.road_legality_protocol")
    actors = checkpoint.get("actors")
    if not isinstance(actors, Mapping) or {int(key) for key in actors} != set(policy["required_agent_ids"]):
        mismatches.append("actors")
    if mismatches:
        raise C0ProtocolError("checkpoint is incompatible with C0: " + ", ".join(mismatches))
    if r7_protocol:
        probe = checkpoint.get("critic_probe_batch")
        if not isinstance(probe, Mapping):
            raise C0ProtocolError(
                "checkpoint is incompatible with C0: critic_probe_batch"
            )
        try:
            validate_critic_probe_batch(
                probe,
                expected_state_dim=int(policy["state_dim"]),
            )
        except ValueError as error:
            raise C0ProtocolError(
                f"checkpoint is incompatible with C0: critic_probe_batch: {error}"
            ) from error
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


def _validate_road_legality_evidence(payload: Any, *, context: str) -> None:
    _require_exact_fields(payload, ROAD_LEGALITY_EVIDENCE_FIELDS, context=context)
    if not str(payload["classification"]).strip():
        raise C0ProtocolError(f"{context}.classification cannot be empty")
    if not isinstance(payload["violation_reason"], str):
        raise C0ProtocolError(f"{context}.violation_reason must be a string")
    lane = payload["reference_lane"]
    if lane is not None and (
        not isinstance(lane, list)
        or len(lane) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lane)
    ):
        raise C0ProtocolError(
            f"{context}.reference_lane must be null or [road_id, lane_id]"
        )
    bool_fields = ROAD_LEGALITY_EVIDENCE_FIELDS - {
        "classification",
        "reference_lane",
        "violation_reason",
        "consecutive_violation_ticks",
        "legal_transition_grace_remaining",
        "current_to_reference_cosine",
        "vehicle_to_current_cosine",
    }
    if any(not isinstance(payload[name], bool) for name in bool_fields):
        raise C0ProtocolError(f"{context} boolean evidence is invalid")
    for name in (
        "consecutive_violation_ticks",
        "legal_transition_grace_remaining",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise C0ProtocolError(f"{context}.{name} must be a non-negative integer")
    for name in (
        "current_to_reference_cosine",
        "vehicle_to_current_cosine",
    ):
        value = _finite(payload[name], context=f"{context}.{name}")
        if not -1.0 <= value <= 1.0:
            raise C0ProtocolError(f"{context}.{name} must be in [-1,1]")
    if payload["violation_raw"] != bool(payload["violation_reason"]):
        raise C0ProtocolError(f"{context} violation reason/raw flag mismatch")
    if payload["violation_confirmed"] and not payload["violation_raw"]:
        raise C0ProtocolError(f"{context} confirmed violation must still be raw")
    if payload["success_eligible"] and payload["violation_raw"]:
        raise C0ProtocolError(f"{context} raw violation cannot be success-eligible")


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
        ACTION_C0_EPISODE_RECORD_VERSION,
        REWARD_C0_EPISODE_RECORD_VERSION,
        C0_EPISODE_RECORD_VERSION,
        TOWN05_C0_EPISODE_RECORD_VERSION,
        TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
        TOWN05_R6_C0_EPISODE_RECORD_VERSION,
        TOWN05_R7_C0_EPISODE_RECORD_VERSION,
    }:
        raise C0ProtocolError("episode record version mismatch")
    current_action = record_version != LEGACY_C0_EPISODE_RECORD_VERSION
    reward_version = (
        "v5"
        if record_version == TOWN05_R7_C0_EPISODE_RECORD_VERSION
        else "v4"
        if record_version == TOWN05_R6_C0_EPISODE_RECORD_VERSION
        else "v3"
        if record_version in {
            REWARD_C0_EPISODE_RECORD_VERSION,
            C0_EPISODE_RECORD_VERSION,
            TOWN05_C0_EPISODE_RECORD_VERSION,
            TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
            TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
        }
        else "v2"
    )
    if record["eval_id"] != eval_id or record["profile"] != profile:
        raise C0ProtocolError("episode record evaluation identity mismatch")
    if record["checkpoint_sha256"] != checkpoint_sha256:
        raise C0ProtocolError("episode checkpoint SHA256 mismatch")
    if not SHA256_RE.fullmatch(str(record["scenario_sha256"])):
        raise C0ProtocolError("episode scenario SHA256 is invalid")
    episode_length = _positive_int(record["episode_length"], context="episode_length")
    if (
        record["outcome"] not in outcome_names_for_record(record_version)
        or not str(record["episode_end_reason"]).strip()
    ):
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
    offroad_record = record_version in {
        TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
        TOWN05_R6_C0_EPISODE_RECORD_VERSION,
        TOWN05_R7_C0_EPISODE_RECORD_VERSION,
    }
    if offroad_record:
        expected_agent_fields.update(
            {
                "offroad_terminated",
                "first_offroad_evidence",
                "max_offroad_consecutive_ticks",
                "max_footprint_outside_fraction",
            }
        )
    road_record = record_version == TOWN05_R7_C0_EPISODE_RECORD_VERSION
    if road_record:
        expected_agent_fields.update(
            {
                "road_rule_terminated",
                "road_violation_confirmed",
                "road_legality",
            }
        )
    if record_version in {
        TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
        TOWN05_R6_C0_EPISODE_RECORD_VERSION,
        TOWN05_R7_C0_EPISODE_RECORD_VERSION,
    }:
        expected_agent_fields.add("lifecycle_transition")
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
        if record_version in {
            TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
            TOWN05_R6_C0_EPISODE_RECORD_VERSION,
            TOWN05_R7_C0_EPISODE_RECORD_VERSION,
        }:
            transition = agent["lifecycle_transition"]
            if (
                not isinstance(transition, Mapping)
                or set(transition)
                != {"pre_active", "post_active", "done", "terminated", "truncated"}
                or not all(isinstance(value, bool) for value in transition.values())
                or transition
                != {
                    "pre_active": True,
                    "post_active": False,
                    "done": agent["final_done"],
                    "terminated": agent["final_terminated"],
                    "truncated": agent["final_truncated"],
                }
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} lifecycle transition evidence is invalid"
                )
        episode_return = _finite(agent["episode_return"], context=f"agent {agent_id}.episode_return")
        residual = _finite(agent["reward_residual_max_abs"], context=f"agent {agent_id}.reward_residual")
        if residual > 1e-5:
            raise C0ProtocolError(f"agent {agent_id} reward residual exceeds tolerance")
        components = agent["reward_component_sums"]
        if not isinstance(components, Mapping) or set(components) != set(
            REWARD_COMPONENTS[reward_version]
        ):
            raise C0ProtocolError(f"agent {agent_id} reward component schema mismatch")
        component_sum = sum(_finite(value, context=f"agent {agent_id}.reward") for value in components.values())
        if not math.isclose(component_sum, episode_return, rel_tol=0.0, abs_tol=1e-5):
            raise C0ProtocolError(f"agent {agent_id} component sums do not conserve episode return")
        diagnostics = agent["diagnostics"]
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(
            REWARD_DIAGNOSTICS[reward_version]
        ):
            raise C0ProtocolError(f"agent {agent_id} diagnostic schema mismatch")
        for name, stats in diagnostics.items():
            _validate_stats(stats, expected_count=active_steps, context=f"agent {agent_id}.diagnostics.{name}")
        for name in (
            "collision_event",
            "lane_invasion_sensor_event",
            "success_event",
            *(
                (
                    "road_same_reference_lane",
                    "road_adjacent_to_reference",
                    "road_lane_change_permitted",
                    "road_in_junction",
                    "road_violation_raw",
                    "road_violation_terminated",
                    "road_success_eligible",
                )
                if road_record
                else ()
            ),
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

        if offroad_record:
            if not isinstance(agent["offroad_terminated"], bool):
                raise C0ProtocolError("offroad_terminated must be boolean")
            if agent["offroad_terminated"] and not agent["final_terminated"]:
                raise C0ProtocolError(
                    "offroad termination requires final_terminated=True"
                )
            if (
                isinstance(agent["max_offroad_consecutive_ticks"], bool)
                or not isinstance(agent["max_offroad_consecutive_ticks"], int)
                or agent["max_offroad_consecutive_ticks"] < 0
            ):
                raise C0ProtocolError(
                    "max_offroad_consecutive_ticks must be non-negative integer"
                )
            outside_fraction = _finite(
                agent["max_footprint_outside_fraction"],
                context="max_footprint_outside_fraction",
            )
            if not 0.0 <= outside_fraction <= 1.0:
                raise C0ProtocolError(
                    "max_footprint_outside_fraction must be in [0,1]"
                )
            first_evidence = agent["first_offroad_evidence"]
            if first_evidence is not None:
                _require_exact_fields(
                    first_evidence,
                    {
                        "episode_step",
                        "world_frame",
                        "location",
                        "yaw",
                        "footprint_outside_count",
                        "footprint_outside_fraction",
                    },
                    context="first_offroad_evidence",
                )
        if road_record:
            if not isinstance(agent["road_rule_terminated"], bool):
                raise C0ProtocolError("road_rule_terminated must be boolean")
            if not isinstance(agent["road_violation_confirmed"], bool):
                raise C0ProtocolError("road_violation_confirmed must be boolean")
            _validate_road_legality_evidence(
                agent["road_legality"],
                context=f"agent {agent_id}.road_legality",
            )
            evidence = agent["road_legality"]
            terminal_from_diagnostics = (
                diagnostics["road_violation_terminated"]["max"] is not None
                and diagnostics["road_violation_terminated"]["max"] > 0.0
            )
            if agent["road_rule_terminated"] != terminal_from_diagnostics:
                raise C0ProtocolError(
                    f"agent {agent_id} road-rule flag differs from diagnostics"
                )
            if agent["road_violation_confirmed"] != evidence["violation_confirmed"]:
                raise C0ProtocolError(
                    f"agent {agent_id} confirmed road-violation evidence differs"
                )
            if agent["road_rule_terminated"] and (
                not agent["final_terminated"]
                or not agent["road_violation_confirmed"]
            ):
                raise C0ProtocolError(
                    f"agent {agent_id} road-rule termination lacks confirmed terminal evidence"
                )

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
    if offroad_record:
        offroad_flags = [agent["offroad_terminated"] for agent in agents.values()]
        end_is_offroad = record["episode_end_reason"] in {
            "cooperative_offroad",
            "adversary_offroad",
        }
        if end_is_offroad != any(offroad_flags):
            raise C0ProtocolError(
                "offroad end reason must match per-agent offroad termination"
            )
        if road_record and record["outcome"] == "offroad" and not any(offroad_flags):
            raise C0ProtocolError("offroad outcome requires an offroad termination")
        if road_record and record["outcome"] != "offroad" and any(offroad_flags):
            raise C0ProtocolError(
                "R7 offroad termination cannot appear under another outcome"
            )
    if road_record:
        road_flags = [agent["road_rule_terminated"] for agent in agents.values()]
        end_is_road = record["episode_end_reason"] in {
            "cooperative_road_violation",
            "adversary_road_violation",
        }
        if end_is_road != any(road_flags):
            raise C0ProtocolError(
                "road-violation end reason must match per-agent road-rule termination"
            )
        if record["outcome"] == "road_violation" and not any(road_flags):
            raise C0ProtocolError(
                "road_violation outcome requires a road-rule termination"
            )
        if record["outcome"] != "road_violation" and any(road_flags):
            raise C0ProtocolError(
                "road-rule termination cannot appear under another R7 outcome"
            )
        if record["outcome"] == "success" and not all(
            agent["road_legality"]["success_eligible"]
            for agent in agents.values()
        ):
            raise C0ProtocolError(
                "R7 success requires final road-legality eligibility for both agents"
            )


def _aggregate_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(records)
    outcome_counts = Counter(record["outcome"] for record in records)
    outcome_names = (
        outcome_names_for_record(str(records[0].get("record_version", "")))
        if records
        else OUTCOMES
    )
    result = {
        "case_count": count,
        "outcome_counts": {
            name: outcome_counts.get(name, 0) for name in outcome_names
        },
        "outcome_rates": {
            name: outcome_counts.get(name, 0) / count if count else 0.0
            for name in outcome_names
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
    if records and records[0].get("record_version") in {
        TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
        TOWN05_R6_C0_EPISODE_RECORD_VERSION,
        TOWN05_R7_C0_EPISODE_RECORD_VERSION,
    }:
        result["offroad_count"] = sum(
            any(
                record["agents"][agent_id]["offroad_terminated"]
                for agent_id in ("0", "1")
            )
            for record in records
        )
        result["offroad_rate"] = result["offroad_count"] / count if count else 0.0
    return result


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
        reward_version = "v2"
    elif record_version == ACTION_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = ACTION_C0_PROTOCOL_VERSION
        reward_version = "v2"
    elif record_version == REWARD_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = REWARD_C0_PROTOCOL_VERSION
        reward_version = "v3"
    elif record_version == C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = C0_PROTOCOL_VERSION
        reward_version = "v3"
    elif record_version == TOWN05_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = TOWN05_C0_PROTOCOL_VERSION
        reward_version = "v3"
    elif record_version == TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = TOWN05_OFFROAD_C0_PROTOCOL_VERSION
        reward_version = "v3"
    elif record_version == TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION
        reward_version = "v3"
    elif record_version == TOWN05_R6_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = TOWN05_R6_C0_PROTOCOL_VERSION
        reward_version = "v4"
    elif record_version == TOWN05_R7_C0_EPISODE_RECORD_VERSION:
        eval_protocol_version = TOWN05_R7_C0_PROTOCOL_VERSION
        reward_version = "v5"
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
                for component in REWARD_COMPONENTS[reward_version]
            },
        }
        if record_version in {
            TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
            TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
            TOWN05_R6_C0_EPISODE_RECORD_VERSION,
            TOWN05_R7_C0_EPISODE_RECORD_VERSION,
        }:
            agents[agent_id]["offroad_episode_count"] = sum(
                bool(record["agents"][agent_id]["offroad_terminated"])
                for record in records
            )
        if record_version == TOWN05_R7_C0_EPISODE_RECORD_VERSION:
            agents[agent_id]["road_violation_episode_count"] = sum(
                bool(record["agents"][agent_id]["road_rule_terminated"])
                for record in records
            )
    return {
        "summary_version": (
            TOWN05_R7_C0_SUMMARY_VERSION
            if record_version == TOWN05_R7_C0_EPISODE_RECORD_VERSION
            else C0_SUMMARY_VERSION
        ),
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
        base = {
                "scope": scope,
                "name": name,
                "case_count": str(group["case_count"]),
                "success_count": str(counts["success"]),
                "collision_count": str(counts["collision"]),
                "timeout_count": str(counts["timeout"]),
                "success_rate": format(rates["success"], ".12g"),
                "collision_rate": format(rates["collision"], ".12g"),
                "timeout_rate": format(rates["timeout"], ".12g"),
                "mean_episode_length": format(group["episode_length"]["mean"], ".12g"),
                "mean_team_return": format(group["mean_team_return"], ".12g"),
            }
        if summary["eval_protocol_version"] == TOWN05_R7_C0_PROTOCOL_VERSION:
            base.update({
                "offroad_count": str(counts["offroad"]),
                "road_violation_count": str(counts["road_violation"]),
                "offroad_rate": format(rates["offroad"], ".12g"),
                "road_violation_rate": format(rates["road_violation"], ".12g"),
            })
            rows.append({name: base[name] for name in R7_SUMMARY_CSV_FIELDS})
        else:
            base.update({
                "other_count": str(counts["other"]),
                "other_rate": format(rates["other"], ".12g"),
            })
            rows.append({name: base[name] for name in SUMMARY_CSV_FIELDS})
    return rows


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                R7_SUMMARY_CSV_FIELDS
                if rows and set(rows[0]) == set(R7_SUMMARY_CSV_FIELDS)
                else SUMMARY_CSV_FIELDS
            )
            if any(set(row) != set(fieldnames) for row in rows):
                raise C0ProtocolError("summary CSV row schema mismatch")
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        if tuple(reader.fieldnames or ()) not in {
            SUMMARY_CSV_FIELDS,
            R7_SUMMARY_CSV_FIELDS,
        }:
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


def outcome_repeatability_signature(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return safety/lifecycle identity plus registered tolerant diagnostics."""
    signature: Dict[str, Any] = {
        "case_id": record["case_id"],
        "outcome": record["outcome"],
        "episode_end_reason": record["episode_end_reason"],
        "episode_length": record["episode_length"],
    }
    for agent_id in ("0", "1"):
        agent = record["agents"][agent_id]
        prefix = f"agent_{agent_id}"
        signature.update(
            {
                f"{prefix}_reached": agent["reached_target"],
                f"{prefix}_collision": agent["collision"],
                f"{prefix}_offroad_terminated": agent[
                    "offroad_terminated"
                ],
                f"{prefix}_final_done": agent["final_done"],
                f"{prefix}_final_terminated": agent["final_terminated"],
                f"{prefix}_final_truncated": agent["final_truncated"],
                f"{prefix}_first_offroad_evidence_present": (
                    agent["first_offroad_evidence"] is not None
                ),
                f"{prefix}_lifecycle_transition": agent[
                    "lifecycle_transition"
                ],
                f"{prefix}_active_decision_steps": agent[
                    "active_decision_steps"
                ],
                f"{prefix}_episode_return": agent["episode_return"],
            }
        )
    return signature


def compare_repeatability_records(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    repeatability = contract["repeatability"]
    outcome_mode = contract["protocol_version"] in {
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
        TOWN05_R6_C0_PROTOCOL_VERSION,
        TOWN05_R7_C0_PROTOCOL_VERSION,
    }
    if outcome_mode:
        if repeatability["comparison_mode"] != OUTCOME_REPEATABILITY_MODE:
            raise C0ProtocolError("repeatability comparison mode differs")
        hard_fields = repeatability["hard_fields"]
        tolerant_fields = repeatability["tolerant_fields"]
        tick_tolerances = repeatability["absolute_tolerance_ticks"]
        continuous_fields = repeatability["continuous_fields"]
        continuous_pass = repeatability["continuous_pass_abs_tolerance"]
        continuous_hold = repeatability["continuous_hold_abs_tolerance"]
        max_warning_case_fraction = float(
            repeatability["max_warning_case_fraction"]
        )
        systematic_min_case_count = int(
            repeatability["systematic_drift_min_case_count"]
        )
    else:
        configured_fields = repeatability["discrete_fields"]
        if configured_fields != list(REPEATABILITY_FIELDS):
            raise C0ProtocolError(
                "repeatability fields differ from the implemented discrete signature"
            )
    if len(left) != len(right):
        raise C0ProtocolError("repeatability runs have different case counts")
    mismatches = []
    warnings = []
    for index, (left_record, right_record) in enumerate(zip(left, right)):
        if left_record.get("checkpoint_sha256") != right_record.get("checkpoint_sha256"):
            raise C0ProtocolError("repeatability runs use different checkpoints")
        if left_record.get("profile") != right_record.get("profile"):
            raise C0ProtocolError("repeatability runs use different profiles")
        left_signature = (
            outcome_repeatability_signature(left_record)
            if outcome_mode
            else repeatability_signature(left_record)
        )
        right_signature = (
            outcome_repeatability_signature(right_record)
            if outcome_mode
            else repeatability_signature(right_record)
        )
        if not outcome_mode and left_signature != right_signature:
            mismatches.append(
                {
                    "case_ordinal": index + 1,
                    "case_id": left_record.get("case_id", ""),
                    "left": left_signature,
                    "right": right_signature,
                }
            )
        elif outcome_mode:
            hard_differences = {
                field: {
                    "left": left_signature[field],
                    "right": right_signature[field],
                }
                for field in hard_fields
                if left_signature[field] != right_signature[field]
            }
            tolerance_excesses = {}
            tolerance_warnings = {}
            for field in tolerant_fields:
                delta = abs(
                    int(left_signature[field])
                    - int(right_signature[field])
                )
                tolerance = int(tick_tolerances[field])
                if delta > tolerance:
                    tolerance_excesses[field] = {
                        "left": left_signature[field],
                        "right": right_signature[field],
                        "absolute_delta": delta,
                        "allowed": tolerance,
                    }
                elif delta > 0:
                    tolerance_warnings[field] = {
                        "left": left_signature[field],
                        "right": right_signature[field],
                        "absolute_delta": delta,
                        "allowed": tolerance,
                    }
            continuous_excesses = {}
            continuous_warnings = {}
            for field in continuous_fields:
                delta = abs(
                    float(left_signature[field])
                    - float(right_signature[field])
                )
                if delta > float(continuous_hold[field]):
                    continuous_excesses[field] = {
                        "left": left_signature[field],
                        "right": right_signature[field],
                        "absolute_delta": delta,
                        "hold_above": continuous_hold[field],
                    }
                elif delta > float(continuous_pass[field]):
                    continuous_warnings[field] = {
                        "left": left_signature[field],
                        "right": right_signature[field],
                        "absolute_delta": delta,
                        "pass_at_or_below": continuous_pass[field],
                        "hold_above": continuous_hold[field],
                    }
            evidence = {
                "case_ordinal": index + 1,
                "case_id": left_record.get("case_id", ""),
            }
            if hard_differences or tolerance_excesses or continuous_excesses:
                mismatches.append(
                    {
                        **evidence,
                        "hard_differences": hard_differences,
                        "tolerance_excesses": tolerance_excesses,
                        "continuous_excesses": continuous_excesses,
                    }
                )
            elif tolerance_warnings or continuous_warnings:
                warnings.append(
                    {
                        **evidence,
                        "tolerated_tick_differences": tolerance_warnings,
                        "continuous_differences": continuous_warnings,
                    }
                )
    if outcome_mode and left:
        warning_case_fraction = len(warnings) / len(left)
        if (
            len(left) >= systematic_min_case_count
            and warning_case_fraction > max_warning_case_fraction
        ):
            mismatches.append(
                {
                    "scope": "run",
                    "reason": "systematic_tolerated_drift",
                    "warning_case_count": len(warnings),
                    "case_count": len(left),
                    "warning_case_fraction": warning_case_fraction,
                    "hold_above": max_warning_case_fraction,
                }
            )
    else:
        warning_case_fraction = 0.0
    status = "HOLD" if mismatches else ("WARNING" if warnings else "PASS")
    return {
        "status": status,
        "gate_pass": not mismatches,
        "case_count": len(left),
        **(
            {
                "comparison_mode": OUTCOME_REPEATABILITY_MODE,
                "hard_fields": hard_fields,
                "tolerant_fields": tolerant_fields,
                "continuous_fields": continuous_fields,
                "warning_case_fraction": warning_case_fraction,
                "max_warning_case_fraction": max_warning_case_fraction,
                "systematic_drift_min_case_count": systematic_min_case_count,
            }
            if outcome_mode
            else {"discrete_fields": configured_fields}
        ),
        "mismatches": mismatches,
        "warnings": warnings,
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
