"""Frozen P0-BASE-04 GO/HOLD contract and pure decision logic.

The module has no CARLA dependency.  It deliberately separates immutable
thresholds from artifact collection so boundary behavior can be regression
tested before any formal pilot result exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluation.base_c0_protocol import (
    LEGACY_C0_EPISODE_RECORD_VERSION,
    load_c0_contract,
)
from training.experiment_contract import (
    load_experiment_contract,
    verify_critical_sources_clean,
)
from training.metrics_protocol import LEGACY_METRICS_PROTOCOL_VERSION
from training.checkpoint_protocol import sha256_file


GO_HOLD_PROTOCOL_VERSION = "p0_base_go_hold_v1"
GO_HOLD_CONTRACT_ID = "P0_BASE_GO_HOLD_TOWN02_V1"
GO_HOLD_REPORT_VERSION = "p0_base_go_hold_decision_v1"
GO_HOLD_CONTRACT_RELATIVE_PATH = "configs/evaluation/p0_base_go_hold_v1.json"
MILESTONE_DECISIONS = {
    "CONTINUE",
    "CONTINUE_WITH_WARNING",
    "CANDIDATE",
    "PILOT_GO",
    "PILOT_NO_GO",
    "HOLD",
}
VALIDATION_DECISIONS = {"VALIDATION_ACCEPT", "VALIDATION_REJECT", "HOLD"}


class GoHoldProtocolError(RuntimeError):
    """Raised when a GO/HOLD contract or evidence input is invalid."""


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GoHoldProtocolError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _exact(value: Any, fields: set[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoHoldProtocolError(f"{context} must be an object")
    actual = set(value)
    if actual != fields:
        raise GoHoldProtocolError(
            f"{context} fields differ; missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoHoldProtocolError(f"{context} must be finite numeric data")
    number = float(value)
    if not math.isfinite(number):
        raise GoHoldProtocolError(f"{context} must be finite numeric data")
    return number


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GoHoldProtocolError(f"{context} must be a positive integer")
    return value


def _inside(project_dir: Path, relative: Any, *, context: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise GoHoldProtocolError(f"{context} must be a non-empty relative path")
    candidate = (project_dir.resolve() / relative).resolve()
    try:
        candidate.relative_to(project_dir.resolve())
    except ValueError as error:
        raise GoHoldProtocolError(f"{context} escapes project directory") from error
    return candidate


def validate_go_hold_contract(payload: Any, *, project_dir: Path) -> None:
    top = {
        "protocol_version",
        "contract_id",
        "status",
        "purpose",
        "bindings",
        "evidence_rules",
        "training_health",
        "milestone_rules",
        "validation_acceptance",
        "decision_policy",
        "outputs",
        "critical_source_paths",
        "limitations",
    }
    _exact(payload, top, context="GO/HOLD contract")
    if payload["protocol_version"] != GO_HOLD_PROTOCOL_VERSION:
        raise GoHoldProtocolError("GO/HOLD protocol_version mismatch")
    if payload["contract_id"] != GO_HOLD_CONTRACT_ID or payload["status"] != "frozen":
        raise GoHoldProtocolError("GO/HOLD contract identity/status mismatch")
    if not isinstance(payload["purpose"], str) or not payload["purpose"].strip():
        raise GoHoldProtocolError("purpose must be non-empty")

    bindings = _exact(
        payload["bindings"],
        {
            "training_contract_path",
            "training_contract_id",
            "training_contract_sha256",
            "c0_contract_path",
            "c0_contract_id",
            "c0_contract_sha256",
            "metrics_protocol_version",
            "c0_episode_record_version",
            "pilot_seed",
            "milestone_profile",
            "validation_profile",
            "required_repeats",
            "agent_ids",
        },
        context="bindings",
    )
    if bindings["metrics_protocol_version"] != LEGACY_METRICS_PROTOCOL_VERSION:
        raise GoHoldProtocolError("bound historical metrics protocol differs")
    if bindings["c0_episode_record_version"] != LEGACY_C0_EPISODE_RECORD_VERSION:
        raise GoHoldProtocolError("bound historical C0 episode protocol differs")
    if bindings["pilot_seed"] != 42 or bindings["agent_ids"] != [0, 1]:
        raise GoHoldProtocolError("pilot seed/agent IDs differ from P0-BASE-01")
    if bindings["milestone_profile"] != "gate" or bindings["validation_profile"] != "validation":
        raise GoHoldProtocolError("C0 profile bindings are invalid")
    if bindings["required_repeats"] != 2:
        raise GoHoldProtocolError("two independent C0 executions are required")

    training_path = _inside(
        project_dir, bindings["training_contract_path"], context="training contract path"
    )
    c0_path = _inside(project_dir, bindings["c0_contract_path"], context="C0 contract path")
    training = load_experiment_contract(training_path, project_dir=project_dir)
    c0 = load_c0_contract(c0_path, project_dir=project_dir)
    if training["payload"]["contract_id"] != bindings["training_contract_id"]:
        raise GoHoldProtocolError("bound training contract ID mismatch")
    if training["sha256"] != bindings["training_contract_sha256"]:
        raise GoHoldProtocolError("bound training contract SHA256 mismatch")
    if c0["payload"]["contract_id"] != bindings["c0_contract_id"]:
        raise GoHoldProtocolError("bound C0 contract ID mismatch")
    if c0["sha256"] != bindings["c0_contract_sha256"]:
        raise GoHoldProtocolError("bound C0 contract SHA256 mismatch")

    evidence = _exact(
        payload["evidence_rules"],
        {
            "expected_episodes_per_update",
            "reward_conservation_abs_error_max",
            "preupdate_ratio_abs_error_max_exclusive",
            "missing_required_metric_count_max",
            "nonfinite_metric_count_max",
            "longitudinal_conflict_rate_max",
            "allow_skipped_ppo_updates",
            "require_segment_completed",
            "require_formal_periodic_checkpoint",
            "require_c0_validator_pass",
            "require_repeatability_pass",
            "max_other_outcome_count",
        },
        context="evidence_rules",
    )
    if evidence["expected_episodes_per_update"] != 16:
        raise GoHoldProtocolError("formal pilot requires 16 episodes per update")
    for name in (
        "reward_conservation_abs_error_max",
        "preupdate_ratio_abs_error_max_exclusive",
    ):
        if _finite(evidence[name], context=f"evidence_rules.{name}") != 1e-6:
            raise GoHoldProtocolError(f"{name} must remain frozen at 1e-6")
    for name in (
        "missing_required_metric_count_max",
        "nonfinite_metric_count_max",
        "max_other_outcome_count",
    ):
        if evidence[name] != 0:
            raise GoHoldProtocolError(f"evidence_rules.{name} must be zero")
    if _finite(evidence["longitudinal_conflict_rate_max"], context="conflict max") != 0.0:
        raise GoHoldProtocolError("longitudinal conflict must remain zero")
    expected_flags = {
        "allow_skipped_ppo_updates": False,
        "require_segment_completed": True,
        "require_formal_periodic_checkpoint": True,
        "require_c0_validator_pass": True,
        "require_repeatability_pass": True,
    }
    for name, expected in expected_flags.items():
        if evidence[name] is not expected:
            raise GoHoldProtocolError(f"evidence_rules.{name} must be {expected}")

    health = _exact(
        payload["training_health"],
        {
            "approx_kl",
            "clip_fraction",
            "gradient_clip_activation",
            "critic_explained_variance",
            "action_saturation",
            "actor_log_std",
        },
        context="training_health",
    )
    health_fields = {
        "approx_kl": {
            "warning_above",
            "hold_once_above",
            "hold_consecutive_above",
            "hold_consecutive_updates",
        },
        "clip_fraction": {
            "warning_above",
            "hold_consecutive_above",
            "hold_consecutive_updates",
        },
        "gradient_clip_activation": {"warning_above", "warning_consecutive_updates"},
        "critic_explained_variance": {"warning_below", "warning_consecutive_updates"},
        "action_saturation": {"warning_above", "warning_consecutive_updates", "channels"},
        "actor_log_std": {
            "warning_at_or_below",
            "warning_above",
            "warning_consecutive_updates",
        },
    }
    for name, fields in health_fields.items():
        section = _exact(health[name], fields, context=f"training_health.{name}")
        for key, value in section.items():
            if key == "channels":
                if value != ["throttle", "steer", "brake"]:
                    raise GoHoldProtocolError("action saturation channels changed")
            elif key.endswith("updates"):
                _positive_int(value, context=f"training_health.{name}.{key}")
            else:
                _finite(value, context=f"training_health.{name}.{key}")
    if not (
        health["approx_kl"]["warning_above"]
        < health["approx_kl"]["hold_consecutive_above"]
        < health["approx_kl"]["hold_once_above"]
    ):
        raise GoHoldProtocolError("approx_kl thresholds must be strictly ordered")
    if health["clip_fraction"]["warning_above"] >= health["clip_fraction"]["hold_consecutive_above"]:
        raise GoHoldProtocolError("clip_fraction thresholds must be strictly ordered")

    milestones = payload["milestone_rules"]
    if not isinstance(milestones, list) or len(milestones) != 8:
        raise GoHoldProtocolError("exactly eight milestone rules are required")
    expected_updates = training["payload"]["segment_stop_updates"]
    if [rule.get("update") for rule in milestones if isinstance(rule, Mapping)] != expected_updates:
        raise GoHoldProtocolError("milestone updates differ from the training contract")
    common = {"update", "rule_type", "pass_decision", "fail_decision"}
    for rule in milestones:
        rule_type = rule.get("rule_type") if isinstance(rule, Mapping) else None
        extra = {
            "health_only": {"warn_if_zero_success"},
            "minimum_capability": {"min_success_count", "max_collision_count"},
            "provisional_candidate": {
                "min_success_count",
                "max_collision_count",
                "max_timeout_count",
            },
            "pilot_go": {
                "min_success_count",
                "max_collision_count",
                "max_timeout_count",
                "min_success_per_family",
                "max_lane_invasion_episode_count",
            },
        }.get(str(rule_type))
        if extra is None:
            raise GoHoldProtocolError(f"unknown milestone rule_type={rule_type!r}")
        _exact(rule, common | extra, context=f"milestone update {rule.get('update')}")
        if rule["pass_decision"] not in MILESTONE_DECISIONS or rule["fail_decision"] not in MILESTONE_DECISIONS:
            raise GoHoldProtocolError("invalid milestone decision name")
        for key in extra - {"warn_if_zero_success"}:
            if isinstance(rule[key], bool) or not isinstance(rule[key], int) or rule[key] < 0:
                raise GoHoldProtocolError(f"milestone {rule['update']} {key} must be non-negative int")
        if "warn_if_zero_success" in extra and not isinstance(rule["warn_if_zero_success"], bool):
            raise GoHoldProtocolError("warn_if_zero_success must be boolean")

    validation = _exact(
        payload["validation_acceptance"],
        {
            "release_stage",
            "case_count",
            "min_success_count",
            "max_collision_count",
            "max_timeout_count",
            "max_other_count",
            "min_success_per_family",
            "min_success_per_scenario",
            "max_lane_invasion_episode_count",
            "pass_decision",
            "fail_decision",
        },
        context="validation_acceptance",
    )
    if validation["release_stage"] != "P0-BASE-06":
        raise GoHoldProtocolError("held-out acceptance is reserved for P0-BASE-06")
    for name in (
        "case_count",
        "min_success_count",
        "max_collision_count",
        "max_timeout_count",
        "max_other_count",
        "min_success_per_family",
        "min_success_per_scenario",
        "max_lane_invasion_episode_count",
    ):
        if isinstance(validation[name], bool) or not isinstance(validation[name], int) or validation[name] < 0:
            raise GoHoldProtocolError(f"validation_acceptance.{name} must be non-negative int")
    if validation["pass_decision"] != "VALIDATION_ACCEPT" or validation["fail_decision"] != "VALIDATION_REJECT":
        raise GoHoldProtocolError("validation decision names are frozen")

    policy = _exact(
        payload["decision_policy"],
        {
            "returns_are_selection_criteria",
            "gate_repeat_is_additional_sample",
            "warnings_block_pilot_go",
            "update_80_can_release_multiseed",
            "only_update_100_pilot_go_releases_multiseed",
            "threshold_override_allowed",
            "threshold_commit_must_be_ancestor_of_run_origin",
        },
        context="decision_policy",
    )
    expected_policy = {
        "returns_are_selection_criteria": False,
        "gate_repeat_is_additional_sample": False,
        "warnings_block_pilot_go": False,
        "update_80_can_release_multiseed": False,
        "only_update_100_pilot_go_releases_multiseed": True,
        "threshold_override_allowed": False,
        "threshold_commit_must_be_ancestor_of_run_origin": True,
    }
    if dict(policy) != expected_policy:
        raise GoHoldProtocolError("decision policy has drifted")

    outputs = _exact(payload["outputs"], {"decision_root", "decision_report"}, context="outputs")
    if dict(outputs) != {
        "decision_root": "decisions/base_mappo",
        "decision_report": "go_hold_decision.json",
    }:
        raise GoHoldProtocolError("decision output paths have drifted")
    _inside(project_dir, outputs["decision_root"], context="decision root")
    paths = payload["critical_source_paths"]
    expected_paths = [
        "evaluation/base_go_hold_protocol.py",
        "evaluation/judge_base_go_hold.py",
        GO_HOLD_CONTRACT_RELATIVE_PATH,
    ]
    if paths != expected_paths:
        raise GoHoldProtocolError("critical_source_paths have drifted")
    for index, relative in enumerate(paths):
        path = _inside(project_dir, relative, context=f"critical_source_paths[{index}]")
        if not path.is_file():
            raise GoHoldProtocolError(f"critical source file not found: {path}")
    if not isinstance(payload["limitations"], list) or len(payload["limitations"]) < 4:
        raise GoHoldProtocolError("limitations must state the registered inference boundaries")


def load_go_hold_contract(path: str | Path, *, project_dir: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GoHoldProtocolError(f"GO/HOLD contract not found: {resolved}")
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GoHoldProtocolError(f"non-finite JSON constant is forbidden: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise GoHoldProtocolError(f"invalid GO/HOLD contract JSON: {error}") from error
    validate_go_hold_contract(payload, project_dir=project_dir)
    relative = resolved.relative_to(project_dir.resolve()).as_posix()
    if relative != GO_HOLD_CONTRACT_RELATIVE_PATH:
        raise GoHoldProtocolError(
            f"GO/HOLD contract must use the registered path {GO_HOLD_CONTRACT_RELATIVE_PATH}"
        )
    if relative not in payload["critical_source_paths"]:
        raise GoHoldProtocolError("active GO/HOLD contract must be a critical source")
    return {
        "payload": payload,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "canonical_sha256": canonical_json_sha256(payload),
    }


def _max_streak(values: Sequence[float], predicate) -> int:
    current = 0
    maximum = 0
    for value in values:
        if predicate(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def evaluate_training_health(
    metric_records: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate only pre-registered numerical-health thresholds."""
    if not metric_records:
        raise GoHoldProtocolError("training health requires at least one metrics record")
    agent_ids = [int(value) for value in contract["bindings"]["agent_ids"]]
    updates: list[int] = []
    scalar_rows: list[Mapping[str, Any]] = []
    for index, record in enumerate(metric_records):
        if not isinstance(record, Mapping) or not isinstance(record.get("scalars"), Mapping):
            raise GoHoldProtocolError(f"metrics record {index} has no scalar object")
        update = record.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            raise GoHoldProtocolError(f"metrics record {index} update is invalid")
        updates.append(update)
        scalar_rows.append(record["scalars"])
    if updates != list(range(1, updates[-1] + 1)):
        raise GoHoldProtocolError("training health metrics must be contiguous from update 1")

    hard_failures: list[str] = []
    warnings: list[str] = []
    observed: Dict[str, Any] = {}

    def series(tag: str) -> list[float]:
        values: list[float] = []
        for update, scalars in zip(updates, scalar_rows):
            if tag not in scalars:
                raise GoHoldProtocolError(f"update {update} missing health tag {tag!r}")
            values.append(_finite(scalars[tag], context=f"update {update} {tag}"))
        return values

    evidence = contract["evidence_rules"]
    episode_counts = series("episode/count")
    missing_counts = series("reward_audit/missing_field_count")
    nonfinite_counts = series("reward_audit/nonfinite_count")
    reward_residuals = series("reward_audit/max_abs_step_residual")
    skipped_updates = series("train/update_skipped")
    observed["evidence"] = {
        "episode_count_min": min(episode_counts),
        "episode_count_max": max(episode_counts),
        "missing_metric_count_max": max(missing_counts),
        "nonfinite_metric_count_max": max(nonfinite_counts),
        "reward_residual_max": max(reward_residuals),
        "skipped_update_max": max(skipped_updates),
    }
    expected_episodes = evidence["expected_episodes_per_update"]
    for update, count in zip(updates, episode_counts):
        if count != expected_episodes:
            hard_failures.append(
                f"update {update} episode/count {count:g} != {expected_episodes}"
            )
    if max(missing_counts) > evidence["missing_required_metric_count_max"]:
        hard_failures.append(
            f"reward/metric missing count max {max(missing_counts):g} > "
            f"{evidence['missing_required_metric_count_max']}"
        )
    if max(nonfinite_counts) > evidence["nonfinite_metric_count_max"]:
        hard_failures.append(
            f"reward/metric nonfinite count max {max(nonfinite_counts):g} > "
            f"{evidence['nonfinite_metric_count_max']}"
        )
    if max(reward_residuals) > evidence["reward_conservation_abs_error_max"]:
        hard_failures.append(
            f"reward residual max {max(reward_residuals):.6g} > "
            f"{evidence['reward_conservation_abs_error_max']}"
        )
    if not evidence["allow_skipped_ppo_updates"] and max(skipped_updates) != 0.0:
        hard_failures.append("one or more PPO updates were skipped")

    health = contract["training_health"]
    for agent_id in agent_ids:
        prefix = f"policy/agent_{agent_id}"
        ratio_error = series(f"{prefix}/preupdate_max_abs_ratio_error")
        conflict = series(f"action/agent_{agent_id}/longitudinal_conflict_rate")
        observed[f"agent_{agent_id}/evidence"] = {
            "preupdate_ratio_error_max": max(ratio_error),
            "longitudinal_conflict_rate_max": max(conflict),
        }
        if max(ratio_error) >= evidence["preupdate_ratio_abs_error_max_exclusive"]:
            hard_failures.append(
                f"agent {agent_id} pre-update ratio error max {max(ratio_error):.6g} "
                f">= {evidence['preupdate_ratio_abs_error_max_exclusive']}"
            )
        if max(conflict) > evidence["longitudinal_conflict_rate_max"]:
            hard_failures.append(
                f"agent {agent_id} longitudinal conflict max {max(conflict):.6g} > "
                f"{evidence['longitudinal_conflict_rate_max']}"
            )
        kl = series(f"{prefix}/approx_kl")
        kl_rule = health["approx_kl"]
        kl_hold_streak = _max_streak(kl, lambda value: value > kl_rule["hold_consecutive_above"])
        observed[f"agent_{agent_id}/approx_kl"] = {
            "max": max(kl),
            "hold_streak": kl_hold_streak,
        }
        if max(kl) > kl_rule["hold_once_above"]:
            hard_failures.append(
                f"agent {agent_id} approx_kl max {max(kl):.6g} > {kl_rule['hold_once_above']}"
            )
        if kl_hold_streak >= kl_rule["hold_consecutive_updates"]:
            hard_failures.append(
                f"agent {agent_id} approx_kl > {kl_rule['hold_consecutive_above']} "
                f"for {kl_hold_streak} consecutive updates"
            )
        if max(kl) > kl_rule["warning_above"]:
            warnings.append(
                f"agent {agent_id} approx_kl max {max(kl):.6g} > warning {kl_rule['warning_above']}"
            )

        clip = series(f"{prefix}/clip_frac")
        clip_rule = health["clip_fraction"]
        clip_streak = _max_streak(clip, lambda value: value > clip_rule["hold_consecutive_above"])
        observed[f"agent_{agent_id}/clip_fraction"] = {
            "max": max(clip),
            "hold_streak": clip_streak,
        }
        if clip_streak >= clip_rule["hold_consecutive_updates"]:
            hard_failures.append(
                f"agent {agent_id} clip_fraction > {clip_rule['hold_consecutive_above']} "
                f"for {clip_streak} consecutive updates"
            )
        if max(clip) > clip_rule["warning_above"]:
            warnings.append(
                f"agent {agent_id} clip_fraction max {max(clip):.6g} > warning {clip_rule['warning_above']}"
            )

        grad = series(f"{prefix}/grad_clip_activation_rate")
        grad_rule = health["gradient_clip_activation"]
        grad_streak = _max_streak(grad, lambda value: value > grad_rule["warning_above"])
        observed[f"agent_{agent_id}/grad_clip_activation"] = {
            "max": max(grad),
            "warning_streak": grad_streak,
        }
        if grad_streak >= grad_rule["warning_consecutive_updates"]:
            warnings.append(
                f"agent {agent_id} gradient clipping > {grad_rule['warning_above']} "
                f"for {grad_streak} consecutive updates"
            )

        log_min = series(f"{prefix}/log_std_min")
        log_max = series(f"{prefix}/log_std_max")
        log_rule = health["actor_log_std"]
        low_streak = _max_streak(log_min, lambda value: value <= log_rule["warning_at_or_below"])
        high_streak = _max_streak(log_max, lambda value: value > log_rule["warning_above"])
        observed[f"agent_{agent_id}/log_std"] = {
            "min": min(log_min),
            "max": max(log_max),
            "low_warning_streak": low_streak,
            "high_warning_streak": high_streak,
        }
        if low_streak >= log_rule["warning_consecutive_updates"]:
            warnings.append(
                f"agent {agent_id} log_std_min <= {log_rule['warning_at_or_below']} "
                f"for {low_streak} consecutive updates"
            )
        if high_streak >= log_rule["warning_consecutive_updates"]:
            warnings.append(
                f"agent {agent_id} log_std_max > {log_rule['warning_above']} "
                f"for {high_streak} consecutive updates"
            )

        saturation_rule = health["action_saturation"]
        for channel in saturation_rule["channels"]:
            values = series(f"action/agent_{agent_id}/{channel}_saturation_rate")
            streak = _max_streak(values, lambda value: value > saturation_rule["warning_above"])
            observed[f"agent_{agent_id}/{channel}_saturation"] = {
                "max": max(values),
                "warning_streak": streak,
            }
            if streak >= saturation_rule["warning_consecutive_updates"]:
                warnings.append(
                    f"agent {agent_id} {channel} true saturation > "
                    f"{saturation_rule['warning_above']} for {streak} consecutive updates"
                )

    critic_grad = series("critic/grad_clip_activation_rate")
    grad_rule = health["gradient_clip_activation"]
    critic_grad_streak = _max_streak(
        critic_grad, lambda value: value > grad_rule["warning_above"]
    )
    observed["critic/grad_clip_activation"] = {
        "max": max(critic_grad),
        "warning_streak": critic_grad_streak,
    }
    if critic_grad_streak >= grad_rule["warning_consecutive_updates"]:
        warnings.append(
            f"critic gradient clipping > {grad_rule['warning_above']} "
            f"for {critic_grad_streak} consecutive updates"
        )

    explained = series("critic/explained_variance")
    explained_rule = health["critic_explained_variance"]
    explained_streak = _max_streak(
        explained, lambda value: value < explained_rule["warning_below"]
    )
    observed["critic/explained_variance"] = {
        "min": min(explained),
        "warning_streak": explained_streak,
    }
    if explained_streak >= explained_rule["warning_consecutive_updates"]:
        warnings.append(
            f"critic explained_variance < {explained_rule['warning_below']} "
            f"for {explained_streak} consecutive updates"
        )

    return {
        "status": "HOLD" if hard_failures else "PASS",
        "updates": updates,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "observed": observed,
    }


def summarize_c0_gate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise GoHoldProtocolError("C0 decision requires episode records")
    outcomes = Counter(str(record.get("outcome")) for record in records)
    if set(outcomes) - {"success", "collision", "timeout", "other"}:
        raise GoHoldProtocolError(f"unknown C0 outcomes: {sorted(set(outcomes) - {'success', 'collision', 'timeout', 'other'})}")
    family_counts: Dict[str, Counter] = {}
    scenario_counts: Dict[str, Counter] = {}
    lane_invasion_count = 0
    for index, record in enumerate(records):
        family = str(record.get("scenario_family", "")).strip()
        scenario = str(record.get("scenario_name", "")).strip()
        if not family or not scenario:
            raise GoHoldProtocolError(f"C0 record {index} lacks scenario identity")
        family_counts.setdefault(family, Counter())[str(record["outcome"])] += 1
        scenario_counts.setdefault(scenario, Counter())[str(record["outcome"])] += 1
        agents = record.get("agents")
        if not isinstance(agents, Mapping) or set(agents) != {"0", "1"}:
            raise GoHoldProtocolError(f"C0 record {index} has invalid agents")
        event = False
        for agent_id in ("0", "1"):
            try:
                maximum = agents[agent_id]["diagnostics"]["lane_invasion_sensor_event"]["max"]
            except (KeyError, TypeError) as error:
                raise GoHoldProtocolError(
                    f"C0 record {index} lacks lane-invasion diagnostics"
                ) from error
            event = event or maximum not in (None, 0, 0.0)
        lane_invasion_count += int(event)
    return {
        "case_count": len(records),
        "outcome_counts": {
            name: int(outcomes.get(name, 0))
            for name in ("success", "collision", "timeout", "other")
        },
        "family_case_counts": {
            name: int(sum(counts.values())) for name, counts in sorted(family_counts.items())
        },
        "family_success_counts": {
            name: int(counts.get("success", 0)) for name, counts in sorted(family_counts.items())
        },
        "scenario_case_counts": {
            name: int(sum(counts.values())) for name, counts in sorted(scenario_counts.items())
        },
        "scenario_success_counts": {
            name: int(counts.get("success", 0)) for name, counts in sorted(scenario_counts.items())
        },
        "lane_invasion_episode_count": lane_invasion_count,
    }


def _capability_failures(summary: Mapping[str, Any], rule: Mapping[str, Any]) -> list[str]:
    counts = summary["outcome_counts"]
    failures: list[str] = []
    checks = (
        ("min_success_count", counts["success"], lambda actual, limit: actual >= limit, ">="),
        ("max_collision_count", counts["collision"], lambda actual, limit: actual <= limit, "<="),
        ("max_timeout_count", counts["timeout"], lambda actual, limit: actual <= limit, "<="),
        ("max_other_count", counts["other"], lambda actual, limit: actual <= limit, "<="),
        (
            "max_lane_invasion_episode_count",
            summary["lane_invasion_episode_count"],
            lambda actual, limit: actual <= limit,
            "<=",
        ),
    )
    for name, actual, predicate, operator in checks:
        if name in rule and not predicate(actual, rule[name]):
            failures.append(f"{name}: actual {actual} must be {operator} {rule[name]}")
    if "min_success_per_family" in rule:
        for family, actual in summary["family_success_counts"].items():
            if actual < rule["min_success_per_family"]:
                failures.append(
                    f"family {family} success {actual} < {rule['min_success_per_family']}"
                )
    if "min_success_per_scenario" in rule:
        for scenario, actual in summary["scenario_success_counts"].items():
            if actual < rule["min_success_per_scenario"]:
                failures.append(
                    f"scenario {scenario} success {actual} < {rule['min_success_per_scenario']}"
                )
    return failures


def evaluate_milestone_decision(
    *,
    update: int,
    training_health: Mapping[str, Any],
    c0_summary: Mapping[str, Any],
    repeatability_pass: bool,
    contract: Mapping[str, Any],
    evidence_errors: Sequence[str] = (),
) -> Dict[str, Any]:
    rules = {int(rule["update"]): rule for rule in contract["milestone_rules"]}
    if update not in rules:
        raise GoHoldProtocolError(f"update {update} is not a registered milestone")
    expected_cases = int(contract["evidence_rules"]["expected_episodes_per_update"])
    structural_failures = list(evidence_errors)
    health_updates = training_health.get("updates", [])
    if not isinstance(health_updates, list) or not health_updates or health_updates[-1] != update:
        structural_failures.append(
            f"training health evidence must end at update {update}, got {health_updates!r}"
        )
    if training_health.get("status") != "PASS":
        structural_failures.extend(str(value) for value in training_health.get("hard_failures", []))
    if c0_summary.get("case_count") != expected_cases:
        structural_failures.append(
            f"gate case_count={c0_summary.get('case_count')} expected={expected_cases}"
        )
    other_count = int(c0_summary["outcome_counts"]["other"])
    if other_count > contract["evidence_rules"]["max_other_outcome_count"]:
        structural_failures.append(f"unexpected C0 other outcomes={other_count}")
    if contract["evidence_rules"]["require_repeatability_pass"] and not repeatability_pass:
        structural_failures.append("C0 discrete repeatability did not PASS")
    warnings = list(training_health.get("warnings", []))
    if c0_summary["lane_invasion_episode_count"] > 0 and update < 100:
        warnings.append(
            f"lane invasion occurred in {c0_summary['lane_invasion_episode_count']} gate episodes"
        )
    if structural_failures:
        return {
            "decision": "HOLD",
            "multiseed_released": False,
            "hard_failures": structural_failures,
            "capability_failures": [],
            "warnings": warnings,
        }

    rule = rules[update]
    capability_failures = _capability_failures(c0_summary, rule)
    if rule["rule_type"] == "health_only" and rule["warn_if_zero_success"] and c0_summary["outcome_counts"]["success"] == 0:
        warnings.append(f"update {update} gate has zero autonomous success")
    if capability_failures:
        warnings.extend(capability_failures)
        decision = str(rule["fail_decision"])
    else:
        decision = str(rule["pass_decision"])
        if decision == "CONTINUE" and warnings:
            decision = "CONTINUE_WITH_WARNING"
    return {
        "decision": decision,
        "multiseed_released": decision == "PILOT_GO" and update == 100,
        "hard_failures": [],
        "capability_failures": capability_failures,
        "warnings": warnings,
    }


def evaluate_validation_decision(
    *,
    training_health: Mapping[str, Any],
    c0_summary: Mapping[str, Any],
    repeatability_pass: bool,
    contract: Mapping[str, Any],
    evidence_errors: Sequence[str] = (),
) -> Dict[str, Any]:
    rule = contract["validation_acceptance"]
    structural_failures = list(evidence_errors)
    health_updates = training_health.get("updates", [])
    if not isinstance(health_updates, list) or not health_updates or health_updates[-1] != 100:
        structural_failures.append(
            f"validation training health evidence must end at update 100, got {health_updates!r}"
        )
    if training_health.get("status") != "PASS":
        structural_failures.extend(str(value) for value in training_health.get("hard_failures", []))
    if c0_summary.get("case_count") != rule["case_count"]:
        structural_failures.append(
            f"validation case_count={c0_summary.get('case_count')} expected={rule['case_count']}"
        )
    if contract["evidence_rules"]["require_repeatability_pass"] and not repeatability_pass:
        structural_failures.append("C0 discrete repeatability did not PASS")
    if structural_failures:
        return {
            "decision": "HOLD",
            "hard_failures": structural_failures,
            "capability_failures": [],
            "warnings": list(training_health.get("warnings", [])),
        }
    capability_failures = _capability_failures(c0_summary, rule)
    return {
        "decision": rule["pass_decision"] if not capability_failures else rule["fail_decision"],
        "hard_failures": [],
        "capability_failures": capability_failures,
        "warnings": list(training_health.get("warnings", [])),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight the frozen P0-BASE-04 GO/HOLD contract")
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[4]
    record = load_go_hold_contract(args.contract, project_dir=project_dir)
    verify_critical_sources_clean(
        repository_root=repository_root,
        project_dir=project_dir,
        relative_paths=record["payload"]["critical_source_paths"],
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol_version": GO_HOLD_PROTOCOL_VERSION,
                "contract_id": record["payload"]["contract_id"],
                "contract_sha256": record["sha256"],
                "contract_canonical_sha256": record["canonical_sha256"],
                "critical_sources_clean": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
