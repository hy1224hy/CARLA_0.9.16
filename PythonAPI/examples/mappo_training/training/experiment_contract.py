"""P0-BASE formal experiment contract and evidence helpers.

The contract is intentionally stricter than the generic trainer CLI: scientific
parameters come from one versioned JSON file, while the CLI only selects the
run identity, stop milestone, runtime endpoint, and resume source.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

from training.checkpoint_protocol import atomic_write_json, sha256_file, utc_now_iso


EXPERIMENT_CONTRACT_VERSION = "p0_base_experiment_contract_v1"
CRITIC_EXPERIMENT_CONTRACT_VERSION = "p0_base_experiment_contract_v2"
RESOLVED_CONFIG_NAME = "resolved_config.json"
EXPERIMENT_MANIFEST_NAME = "experiment_manifest.json"

LEGACY_TRAINER_ARG_NAMES = (
    "town",
    "straight_road_length",
    "timestep",
    "max_steps",
    "rollout_episodes",
    "ppo_epochs",
    "minibatch_size",
    "min_transitions_per_update",
    "lr_actor",
    "lr_critic",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "ent_coef",
    "max_grad_norm",
    "hidden_dim",
    "target_kl",
    "lr_decay",
    "lr_decay_horizon",
    "init_log_std",
    "scenario_dir",
    "scenario_sampling_mode",
    "warmup_updates",
    "full_scenario_updates",
    "curriculum_transition_window",
    "curriculum_mode",
    "seed",
    "save_interval",
    "log_interval",
    "tb_histogram_interval",
    "reward_version",
)
TRAINER_ARG_NAMES = LEGACY_TRAINER_ARG_NAMES
CRITIC_TRAINER_ARG_NAMES = LEGACY_TRAINER_ARG_NAMES[:16] + (
    "value_norm_beta",
    "value_norm_epsilon",
    "value_norm_variance_floor",
    "huber_delta",
) + LEGACY_TRAINER_ARG_NAMES[16:]

_TOP_LEVEL_FIELDS = frozenset(
    {
        "contract_version",
        "contract_id",
        "status",
        "purpose",
        "training_entrypoint",
        "source_mode",
        "allowed_seeds",
        "planned_total_updates",
        "segment_stop_updates",
        "run_id_prefix",
        "data_usage_strategy",
        "trainer_args",
        "evaluation_contract",
        "health_stop_rules",
        "registry_path",
        "critical_source_paths",
        "references",
        "parameter_rationale",
    }
)

_ALLOWED_FORMAL_OPTIONS = frozenset(
    {
        "--experiment-contract",
        "--host",
        "--port",
        "--cpu",
        "--updates",
        "--run-id",
        "--save-dir",
        "--resume",
    }
)

_REGISTRY_FIELDS = (
    "run_id",
    "contract_id",
    "contract_sha256",
    "seed",
    "git_sha",
    "code_fingerprint",
    "scenario_pool_fingerprint",
    "requested_stop_update",
    "completed_update",
    "status",
    "run_dir",
    "created_at",
    "updated_at",
    "notes",
)


class ExperimentContractError(ValueError):
    """Raised when a formal experiment violates its frozen contract."""


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expect_exact_fields(
    value: Mapping[str, object], expected: Iterable[str], label: str
) -> None:
    actual = set(value)
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing or unknown:
        raise ExperimentContractError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ExperimentContractError(f"{label} must be >= {minimum}")
    return value


def _require_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ExperimentContractError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise ExperimentContractError(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ExperimentContractError(f"{label} must be <= {maximum}")
    return number


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{label} must be a non-empty string")
    return value.strip()


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ExperimentContractError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _resolve_inside_project(project_dir: Path, value: str, label: str) -> Path:
    candidate = (project_dir / value).resolve()
    try:
        candidate.relative_to(project_dir.resolve())
    except ValueError as error:
        raise ExperimentContractError(
            f"{label} must stay inside project directory: {value!r}"
        ) from error
    return candidate


def validate_experiment_contract(
    payload: Mapping[str, object], *, project_dir: Path
) -> None:
    if not isinstance(payload, dict):
        raise ExperimentContractError("experiment contract root must be an object")
    _expect_exact_fields(payload, _TOP_LEVEL_FIELDS, "contract")

    if payload["contract_version"] not in {
        EXPERIMENT_CONTRACT_VERSION,
        CRITIC_EXPERIMENT_CONTRACT_VERSION,
    }:
        raise ExperimentContractError(
            "unsupported contract_version="
            f"{payload['contract_version']!r}"
        )
    _require_nonempty_string(payload["contract_id"], "contract_id")
    if payload["status"] != "frozen":
        raise ExperimentContractError("formal experiment contract status must be 'frozen'")
    _require_nonempty_string(payload["purpose"], "purpose")
    if payload["training_entrypoint"] != "training.train_no_rllib_mappo_robust_v2":
        raise ExperimentContractError("training_entrypoint does not match robust_v2")
    if payload["source_mode"] != "fresh":
        raise ExperimentContractError("P0-BASE contract source_mode must be 'fresh'")
    run_id_prefix = _require_nonempty_string(payload["run_id_prefix"], "run_id_prefix")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id_prefix):
        raise ExperimentContractError("run_id_prefix contains unsafe characters")
    if payload["data_usage_strategy"] != "single_full_batch_per_epoch":
        raise ExperimentContractError(
            "P0-BASE data_usage_strategy must be single_full_batch_per_epoch"
        )

    allowed_seeds = payload["allowed_seeds"]
    if not isinstance(allowed_seeds, list) or not allowed_seeds:
        raise ExperimentContractError("allowed_seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in allowed_seeds):
        raise ExperimentContractError("allowed_seeds must contain only integers")
    if len(set(allowed_seeds)) != len(allowed_seeds):
        raise ExperimentContractError("allowed_seeds must not contain duplicates")

    planned_updates = _require_int(
        payload["planned_total_updates"], "planned_total_updates", minimum=1
    )
    stop_updates = payload["segment_stop_updates"]
    if not isinstance(stop_updates, list) or not stop_updates:
        raise ExperimentContractError("segment_stop_updates must be a non-empty list")
    for index, update in enumerate(stop_updates):
        _require_int(update, f"segment_stop_updates[{index}]", minimum=1)
    if stop_updates != sorted(set(stop_updates)):
        raise ExperimentContractError(
            "segment_stop_updates must be strictly increasing and unique"
        )
    if stop_updates[-1] != planned_updates:
        raise ExperimentContractError(
            "last segment_stop_update must equal planned_total_updates"
        )

    trainer_args = payload["trainer_args"]
    if not isinstance(trainer_args, dict):
        raise ExperimentContractError("trainer_args must be an object")
    contract_trainer_arg_names = (
        CRITIC_TRAINER_ARG_NAMES
        if payload["contract_version"] == CRITIC_EXPERIMENT_CONTRACT_VERSION
        else LEGACY_TRAINER_ARG_NAMES
    )
    _expect_exact_fields(trainer_args, contract_trainer_arg_names, "trainer_args")

    for name in (
        "max_steps",
        "rollout_episodes",
        "ppo_epochs",
        "minibatch_size",
        "min_transitions_per_update",
        "hidden_dim",
        "lr_decay_horizon",
        "warmup_updates",
        "full_scenario_updates",
        "curriculum_transition_window",
        "save_interval",
        "log_interval",
    ):
        _require_int(trainer_args[name], f"trainer_args.{name}", minimum=1)
    _require_int(
        trainer_args["tb_histogram_interval"],
        "trainer_args.tb_histogram_interval",
        minimum=0,
    )
    _require_int(trainer_args["seed"], "trainer_args.seed")
    for name in (
        "straight_road_length",
        "timestep",
        "lr_actor",
        "lr_critic",
        "ent_coef",
        "max_grad_norm",
        "target_kl",
    ):
        _require_number(trainer_args[name], f"trainer_args.{name}", minimum=0.0)
    for name in (
        "straight_road_length",
        "timestep",
        "lr_actor",
        "lr_critic",
        "max_grad_norm",
        "target_kl",
    ):
        if float(trainer_args[name]) <= 0.0:
            raise ExperimentContractError(f"trainer_args.{name} must be > 0")
    for name in ("gamma", "gae_lambda", "clip_eps"):
        _require_number(
            trainer_args[name], f"trainer_args.{name}", minimum=0.0, maximum=1.0
        )
    if payload["contract_version"] == CRITIC_EXPERIMENT_CONTRACT_VERSION:
        _require_number(
            trainer_args["value_norm_beta"],
            "trainer_args.value_norm_beta",
            minimum=0.0,
            maximum=1.0,
        )
        if float(trainer_args["value_norm_beta"]) >= 1.0:
            raise ExperimentContractError("trainer_args.value_norm_beta must be < 1")
        for name in (
            "value_norm_epsilon",
            "value_norm_variance_floor",
            "huber_delta",
        ):
            _require_number(trainer_args[name], f"trainer_args.{name}", minimum=0.0)
            if float(trainer_args[name]) <= 0.0:
                raise ExperimentContractError(f"trainer_args.{name} must be > 0")
    _require_number(trainer_args["init_log_std"], "trainer_args.init_log_std")
    if not isinstance(trainer_args["lr_decay"], bool):
        raise ExperimentContractError("trainer_args.lr_decay must be boolean")
    for name in ("town", "scenario_dir", "scenario_sampling_mode", "curriculum_mode"):
        _require_nonempty_string(trainer_args[name], f"trainer_args.{name}")
    if trainer_args["reward_version"] not in {"v1", "v2", "v3"}:
        raise ExperimentContractError(
            "trainer_args.reward_version must be v1, v2 or v3"
        )
    if trainer_args["scenario_sampling_mode"] not in {
        "random",
        "stratified",
        "weighted",
    }:
        raise ExperimentContractError("invalid trainer_args.scenario_sampling_mode")
    if trainer_args["curriculum_mode"] not in {"metadata", "classic", "distance"}:
        raise ExperimentContractError("invalid trainer_args.curriculum_mode")
    if trainer_args["seed"] not in allowed_seeds:
        raise ExperimentContractError("trainer_args.seed is not present in allowed_seeds")
    if trainer_args["lr_decay_horizon"] != planned_updates:
        raise ExperimentContractError(
            "lr_decay_horizon must equal planned_total_updates, not segment stop"
        )
    if not (
        trainer_args["warmup_updates"]
        < trainer_args["full_scenario_updates"]
        <= planned_updates
    ):
        raise ExperimentContractError(
            "curriculum boundaries must satisfy warmup < full <= planned updates"
        )
    if any(update % trainer_args["save_interval"] != 0 for update in stop_updates):
        raise ExperimentContractError(
            "each segment stop must align with save_interval for a safe checkpoint"
        )
    maximum_critic_batch = (
        2 * trainer_args["rollout_episodes"] * trainer_args["max_steps"]
    )
    if trainer_args["minibatch_size"] < maximum_critic_batch:
        raise ExperimentContractError(
            "single_full_batch_per_epoch requires minibatch_size >= "
            f"2 * rollout_episodes * max_steps ({maximum_critic_batch})"
        )

    scenario_dir = _resolve_inside_project(
        project_dir, trainer_args["scenario_dir"], "trainer_args.scenario_dir"
    )
    if not scenario_dir.is_dir():
        raise ExperimentContractError(f"scenario directory not found: {scenario_dir}")
    if not any(scenario_dir.glob("*.json")):
        raise ExperimentContractError(f"scenario directory contains no JSON files: {scenario_dir}")

    evaluation = payload["evaluation_contract"]
    if not isinstance(evaluation, dict):
        raise ExperimentContractError("evaluation_contract must be an object")
    _expect_exact_fields(
        evaluation,
        {
            "protocol_version",
            "milestone_updates",
            "deterministic_actions",
            "status",
            "performance_gate_status",
        },
        "evaluation_contract",
    )
    _require_nonempty_string(evaluation["protocol_version"], "evaluation protocol")
    if evaluation["milestone_updates"] != stop_updates:
        raise ExperimentContractError(
            "evaluation milestones must equal segment stop updates"
        )
    if evaluation["deterministic_actions"] is not True:
        raise ExperimentContractError("formal C0 evaluation must be deterministic")
    if evaluation["status"] != "interface_reserved_external_p0_base_02":
        raise ExperimentContractError("unexpected evaluation contract status")
    if evaluation["performance_gate_status"] != "freeze_before_pilot_results":
        raise ExperimentContractError("performance gate must be frozen before pilot results")

    health = payload["health_stop_rules"]
    if not isinstance(health, dict):
        raise ExperimentContractError("health_stop_rules must be an object")
    _expect_exact_fields(
        health,
        {
            "nonfinite_is_hold",
            "reward_conservation_abs_error_max",
            "preupdate_ratio_abs_error_max",
            "active_transition_mismatch_max",
            "missing_required_metric_count_max",
            "carla_actor_leak_count_max",
            "allow_skipped_ppo_updates",
            "allow_checkpoint_validation_failure",
            "sync_mode_restore_failure_is_hold",
            "required_artifact_validation_failure_is_hold",
            "action_boundary_rate_warning",
        },
        "health_stop_rules",
    )
    if health["nonfinite_is_hold"] is not True:
        raise ExperimentContractError("nonfinite_is_hold must be true")
    for name in (
        "reward_conservation_abs_error_max",
        "preupdate_ratio_abs_error_max",
        "action_boundary_rate_warning",
    ):
        _require_number(
            health[name],
            f"health_stop_rules.{name}",
            minimum=0.0,
            maximum=(1.0 if name == "action_boundary_rate_warning" else None),
        )
    for name in (
        "active_transition_mismatch_max",
        "missing_required_metric_count_max",
        "carla_actor_leak_count_max",
    ):
        _require_int(health[name], f"health_stop_rules.{name}", minimum=0)
    if health["allow_skipped_ppo_updates"] is not False:
        raise ExperimentContractError("allow_skipped_ppo_updates must be false")
    if health["allow_checkpoint_validation_failure"] is not False:
        raise ExperimentContractError("allow_checkpoint_validation_failure must be false")
    if health["sync_mode_restore_failure_is_hold"] is not True:
        raise ExperimentContractError("sync_mode_restore_failure_is_hold must be true")
    if health["required_artifact_validation_failure_is_hold"] is not True:
        raise ExperimentContractError(
            "required_artifact_validation_failure_is_hold must be true"
        )

    registry_path = _resolve_inside_project(
        project_dir,
        _require_nonempty_string(payload["registry_path"], "registry_path"),
        "registry_path",
    )
    if registry_path.suffix.lower() != ".csv":
        raise ExperimentContractError("registry_path must be a CSV file")

    critical_paths = payload["critical_source_paths"]
    if not isinstance(critical_paths, list) or not critical_paths:
        raise ExperimentContractError("critical_source_paths must be a non-empty list")
    for index, relative in enumerate(critical_paths):
        _resolve_inside_project(
            project_dir,
            _require_nonempty_string(relative, f"critical_source_paths[{index}]"),
            f"critical_source_paths[{index}]",
        )
    if len(set(critical_paths)) != len(critical_paths):
        raise ExperimentContractError("critical_source_paths must not contain duplicates")

    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise ExperimentContractError("references must be a non-empty list")
    if any(not isinstance(url, str) or not url.startswith("https://") for url in references):
        raise ExperimentContractError("references must contain HTTPS URLs")
    rationale = payload["parameter_rationale"]
    if not isinstance(rationale, dict) or not rationale:
        raise ExperimentContractError("parameter_rationale must be a non-empty object")
    for key, value in rationale.items():
        _require_nonempty_string(key, "parameter_rationale key")
        _require_nonempty_string(value, f"parameter_rationale.{key}")


def load_experiment_contract(path: str | Path, *, project_dir: Path) -> Dict[str, object]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise ExperimentContractError(f"experiment contract not found: {resolved_path}")
    try:
        relative_contract_path = str(resolved_path.relative_to(project_dir.resolve()))
    except ValueError as error:
        raise ExperimentContractError(
            f"experiment contract must stay inside project directory: {resolved_path}"
        ) from error
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ExperimentContractError(
                        f"non-finite JSON constant is forbidden: {token}"
                    )
                ),
            )
    except json.JSONDecodeError as error:
        raise ExperimentContractError(f"invalid experiment contract JSON: {error}") from error
    validate_experiment_contract(payload, project_dir=project_dir)
    normalized_relative_contract_path = relative_contract_path.replace("\\", "/")
    if normalized_relative_contract_path not in payload["critical_source_paths"]:
        raise ExperimentContractError(
            "the active contract file must be listed in critical_source_paths: "
            f"{normalized_relative_contract_path!r}"
        )
    return {
        "payload": payload,
        "path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "canonical_sha256": canonical_json_sha256(payload),
        "registry_path": str(
            _resolve_inside_project(project_dir, payload["registry_path"], "registry_path")
        ),
    }


def apply_experiment_contract(
    args: Namespace, *, project_dir: Path
) -> Dict[str, object] | None:
    contract_path = str(getattr(args, "experiment_contract", "") or "").strip()
    if not contract_path:
        return None

    explicit_options = set(getattr(args, "_explicit_cli_options", set()))
    unsupported = sorted(explicit_options - _ALLOWED_FORMAL_OPTIONS)
    if unsupported:
        raise ExperimentContractError(
            "formal experiment scientific/runtime policy is contract-owned; "
            f"unsupported CLI overrides={unsupported}"
        )
    required = {"--updates", "--run-id", "--save-dir"}
    missing = sorted(required - explicit_options)
    if missing:
        raise ExperimentContractError(
            f"formal experiment requires explicit run identity/stop options: missing={missing}"
        )
    if getattr(args, "pretrained_checkpoint", ""):
        raise ExperimentContractError("P0-BASE formal contract forbids pretrained weights")
    if bool(getattr(args, "allow_resume_code_drift", False)):
        raise ExperimentContractError("formal resume forbids --allow-resume-code-drift")
    if int(getattr(args, "p0_smoke_interrupt_after_episodes", 0)) != 0:
        raise ExperimentContractError("formal run forbids the P0 smoke interrupt hook")

    record = load_experiment_contract(contract_path, project_dir=project_dir)
    payload = record["payload"]
    trainer_args = payload["trainer_args"]
    contract_trainer_arg_names = (
        CRITIC_TRAINER_ARG_NAMES
        if payload["contract_version"] == CRITIC_EXPERIMENT_CONTRACT_VERSION
        else LEGACY_TRAINER_ARG_NAMES
    )
    for name in contract_trainer_arg_names:
        value = trainer_args[name]
        if name == "scenario_dir":
            value = str(_resolve_inside_project(project_dir, value, "scenario_dir"))
        setattr(args, name, value)

    requested_stop = int(args.updates)
    if requested_stop not in payload["segment_stop_updates"]:
        raise ExperimentContractError(
            f"--updates={requested_stop} is not a registered segment stop; "
            f"allowed={payload['segment_stop_updates']}"
        )
    run_id = str(args.run_id).strip()
    prefix = str(payload["run_id_prefix"])
    if not re.fullmatch(re.escape(prefix) + r"\d{8}_\d{6}", run_id):
        raise ExperimentContractError(
            f"run_id must match {prefix}YYYYMMDD_HHMMSS; got {run_id!r}"
        )
    save_dir = Path(args.save_dir).expanduser().resolve()
    if save_dir.name != run_id:
        raise ExperimentContractError(
            "formal --save-dir basename must equal --run-id: "
            f"{save_dir.name!r} != {run_id!r}"
        )

    args.experiment_contract = record["path"]
    args.config_path = record["path"]
    metadata = {
        **record,
        "contract_id": payload["contract_id"],
        "requested_stop_update": requested_stop,
        "planned_total_updates": int(payload["planned_total_updates"]),
    }
    args._experiment_contract_metadata = metadata
    return metadata


def verify_critical_sources_clean(
    *, repository_root: Path, project_dir: Path, relative_paths: Sequence[str]
) -> None:
    repository_root = repository_root.resolve()
    repo_relative_paths = []
    for relative in relative_paths:
        absolute = _resolve_inside_project(project_dir, relative, "critical source path")
        if not absolute.exists():
            raise ExperimentContractError(f"critical source path not found: {absolute}")
        try:
            repo_relative_paths.append(str(absolute.relative_to(repository_root)))
        except ValueError as error:
            raise ExperimentContractError(
                f"critical source path is outside repository: {absolute}"
            ) from error
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *repo_relative_paths,
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise ExperimentContractError(
            f"cannot inspect critical Git paths: {result.stderr.strip()}"
        )
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    if dirty:
        raise ExperimentContractError(
            "formal experiment requires committed critical sources; dirty entries="
            + repr(dirty)
        )
    tracked_result = subprocess.run(
        ["git", "ls-files", "--", *repo_relative_paths],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if tracked_result.returncode != 0:
        raise ExperimentContractError(
            f"cannot inspect tracked critical paths: {tracked_result.stderr.strip()}"
        )
    tracked = {
        line.strip().replace("\\", "/").lower()
        for line in tracked_result.stdout.splitlines()
        if line.strip()
    }
    required_tracked = set()
    for relative in relative_paths:
        absolute = _resolve_inside_project(project_dir, relative, "critical source path")
        if absolute.is_file():
            required_tracked.add(
                absolute.relative_to(repository_root).as_posix().lower()
            )
        else:
            required_tracked.update(
                path.relative_to(repository_root).as_posix().lower()
                for path in absolute.rglob("*.json")
                if path.is_file()
            )
    missing_tracked = sorted(required_tracked - tracked)
    if missing_tracked:
        raise ExperimentContractError(
            "formal experiment critical inputs must be Git-tracked; missing="
            + repr(missing_tracked)
        )


def enforce_formal_transition_minimum(
    *, args: Namespace, total_transitions: int
) -> None:
    """Turn an otherwise diagnostic skipped PPO update into a formal HOLD."""
    if not getattr(args, "_experiment_contract_metadata", None):
        return
    required = int(args.min_transitions_per_update)
    if int(total_transitions) < required:
        raise ExperimentContractError(
            "formal experiment HOLD: total transitions "
            f"{int(total_transitions)} < required {required}; current incomplete "
            "update must not be committed"
        )


def _write_registry_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _REGISTRY_FIELDS:
                raise ExperimentContractError(
                    f"registry schema mismatch at {path}: {reader.fieldnames}"
                )
            rows = [dict(existing) for existing in reader]
    normalized = {field: str(row.get(field, "")) for field in _REGISTRY_FIELDS}
    matches = [index for index, existing in enumerate(rows) if existing["run_id"] == normalized["run_id"]]
    if len(matches) > 1:
        raise ExperimentContractError(f"registry contains duplicate run_id={normalized['run_id']!r}")
    if matches:
        index = matches[0]
        existing = rows[index]
        for immutable in ("contract_id", "contract_sha256", "seed", "run_dir"):
            if existing[immutable] and existing[immutable] != normalized[immutable]:
                raise ExperimentContractError(
                    f"registry immutable field drift for {immutable}: "
                    f"{existing[immutable]!r} != {normalized[immutable]!r}"
                )
        if existing["created_at"]:
            normalized["created_at"] = existing["created_at"]
        rows[index] = normalized
    else:
        rows.append(normalized)

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def initialize_experiment_evidence(
    *,
    args: Namespace,
    run_dir: Path,
    git_metadata: Mapping[str, object],
    code_fingerprint: str,
    scenario_pool_fingerprint: str,
    training_config: Mapping[str, object],
) -> None:
    metadata = getattr(args, "_experiment_contract_metadata", None)
    if not metadata:
        return
    payload = metadata["payload"]
    contract_trainer_arg_names = (
        CRITIC_TRAINER_ARG_NAMES
        if payload["contract_version"] == CRITIC_EXPERIMENT_CONTRACT_VERSION
        else LEGACY_TRAINER_ARG_NAMES
    )
    now = utc_now_iso()
    checkpoint_training_config = dict(training_config)
    unknown_checkpoint_fields = sorted(
        set(checkpoint_training_config) - set(contract_trainer_arg_names)
    )
    if unknown_checkpoint_fields:
        raise ExperimentContractError(
            "checkpoint training config contains fields outside the formal contract: "
            f"{unknown_checkpoint_fields}"
        )
    resolved_training_config = {
        name: getattr(args, name) for name in contract_trainer_arg_names
    }
    checkpoint_mismatches = {
        name: {
            "checkpoint": value,
            "resolved": resolved_training_config[name],
        }
        for name, value in checkpoint_training_config.items()
        if resolved_training_config[name] != value
    }
    if checkpoint_mismatches:
        raise ExperimentContractError(
            "checkpoint and formal resolved training config differ: "
            f"{checkpoint_mismatches}"
        )
    resolved = {
        "evidence_schema_version": "p0_base_resolved_config_v1",
        "created_at": now,
        "run_id": args.run_id,
        "run_dir": str(run_dir.resolve()),
        "contract_id": metadata["contract_id"],
        "contract_path": metadata["path"],
        "contract_sha256": metadata["sha256"],
        "contract_canonical_sha256": metadata["canonical_sha256"],
        "requested_stop_update": int(args.updates),
        "planned_total_updates": metadata["planned_total_updates"],
        # Checkpoint compatibility intentionally stores only resume-immutable
        # fields.  Formal experiment evidence must instead retain every
        # contract-owned trainer argument, including the seed, so downstream
        # GO/HOLD provenance checks never have to infer it from a filename or
        # a different artifact.
        "resolved_training_config": resolved_training_config,
        "resolved_scenario_dir": str(Path(args.scenario_dir).resolve()),
        "scenario_pool_fingerprint": scenario_pool_fingerprint,
        "git": dict(git_metadata),
        "code_fingerprint": code_fingerprint,
        "health_stop_rules": payload["health_stop_rules"],
        "evaluation_contract": payload["evaluation_contract"],
    }
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    manifest_path = run_dir / EXPERIMENT_MANIFEST_NAME
    if resolved_path.exists() or manifest_path.exists():
        raise ExperimentContractError(
            "fresh formal run evidence already exists; refuse implicit overwrite"
        )
    atomic_write_json(resolved, resolved_path)
    manifest = {
        "evidence_schema_version": "p0_base_experiment_manifest_v1",
        "run_id": args.run_id,
        "contract_id": metadata["contract_id"],
        "contract_sha256": metadata["sha256"],
        "resolved_config_sha256": sha256_file(resolved_path),
        "requested_stop_update": int(args.updates),
        "completed_update": 0,
        "status": "PREPARED",
        "created_at": now,
        "updated_at": now,
        "notes": "",
    }
    atomic_write_json(manifest, manifest_path)
    _write_registry_row(
        Path(metadata["registry_path"]),
        {
            "run_id": args.run_id,
            "contract_id": metadata["contract_id"],
            "contract_sha256": metadata["sha256"],
            "seed": args.seed,
            "git_sha": git_metadata.get("commit", ""),
            "code_fingerprint": code_fingerprint,
            "scenario_pool_fingerprint": scenario_pool_fingerprint,
            "requested_stop_update": args.updates,
            "completed_update": 0,
            "status": "PREPARED",
            "run_dir": str(run_dir.resolve()),
            "created_at": now,
            "updated_at": now,
            "notes": "",
        },
    )


def update_experiment_evidence(
    *, args: Namespace, run_dir: Path, completed_update: int, status: str, notes: str = ""
) -> None:
    metadata = getattr(args, "_experiment_contract_metadata", None)
    if not metadata:
        return
    manifest_path = run_dir / EXPERIMENT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ExperimentContractError(f"experiment manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest: MutableMapping[str, object] = json.load(handle)
    if manifest.get("run_id") != args.run_id or manifest.get("contract_sha256") != metadata["sha256"]:
        raise ExperimentContractError("experiment manifest identity mismatch")
    now = utc_now_iso()
    manifest.update(
        {
            "requested_stop_update": int(args.updates),
            "completed_update": int(completed_update),
            "status": str(status),
            "updated_at": now,
            "notes": str(notes)[:500],
        }
    )
    atomic_write_json(dict(manifest), manifest_path)
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    _write_registry_row(
        Path(metadata["registry_path"]),
        {
            "run_id": args.run_id,
            "contract_id": metadata["contract_id"],
            "contract_sha256": metadata["sha256"],
            "seed": args.seed,
            "git_sha": resolved["git"].get("commit", ""),
            "code_fingerprint": resolved["code_fingerprint"],
            "scenario_pool_fingerprint": resolved["scenario_pool_fingerprint"],
            "requested_stop_update": args.updates,
            "completed_update": completed_update,
            "status": status,
            "run_dir": str(run_dir.resolve()),
            "created_at": manifest["created_at"],
            "updated_at": now,
            "notes": str(notes)[:500],
        },
    )


def validate_resume_experiment_evidence(*, args: Namespace, run_dir: Path) -> None:
    metadata = getattr(args, "_experiment_contract_metadata", None)
    if not metadata:
        return
    payload = metadata["payload"]
    contract_trainer_arg_names = (
        CRITIC_TRAINER_ARG_NAMES
        if payload["contract_version"] == CRITIC_EXPERIMENT_CONTRACT_VERSION
        else LEGACY_TRAINER_ARG_NAMES
    )
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    manifest_path = run_dir / EXPERIMENT_MANIFEST_NAME
    if not resolved_path.is_file() or not manifest_path.is_file():
        raise ExperimentContractError("formal resume requires resolved config and experiment manifest")
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for document, label in ((resolved, "resolved config"), (manifest, "experiment manifest")):
        if document.get("run_id") != args.run_id:
            raise ExperimentContractError(f"{label} run_id mismatch")
        if document.get("contract_sha256") != metadata["sha256"]:
            raise ExperimentContractError(f"{label} contract SHA256 mismatch")
    expected_training_config = {
        name: getattr(args, name) for name in contract_trainer_arg_names
    }
    if resolved.get("resolved_training_config") != expected_training_config:
        raise ExperimentContractError(
            "formal resume resolved training config mismatch"
        )
    if resolved.get("resolved_scenario_dir") != str(Path(args.scenario_dir).resolve()):
        raise ExperimentContractError(
            "formal resume resolved scenario directory mismatch"
        )


def preflight_experiment_contract(
    path: str | Path, *, project_dir: Path, repository_root: Path
) -> Dict[str, object]:
    """Validate a frozen contract and its Git-critical inputs without CARLA."""
    record = load_experiment_contract(path, project_dir=project_dir)
    payload = record["payload"]
    verify_critical_sources_clean(
        repository_root=repository_root,
        project_dir=project_dir,
        relative_paths=payload["critical_source_paths"],
    )
    trainer_args = payload["trainer_args"]
    return {
        "status": "PASS",
        "contract_id": payload["contract_id"],
        "contract_sha256": record["sha256"],
        "canonical_sha256": record["canonical_sha256"],
        "planned_total_updates": payload["planned_total_updates"],
        "segment_stop_updates": payload["segment_stop_updates"],
        "maximum_critic_batch": (
            2 * trainer_args["rollout_episodes"] * trainer_args["max_steps"]
        ),
        "minibatch_size": trainer_args["minibatch_size"],
        "critical_sources_clean": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a P0-BASE experiment contract without starting CARLA."
    )
    parser.add_argument("--contract", required=True, help="contract JSON path")
    cli_args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[4]
    result = preflight_experiment_contract(
        cli_args.contract,
        project_dir=project_dir,
        repository_root=repository_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
