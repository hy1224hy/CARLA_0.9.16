"""Versioned metric schemas and strict reward-audit helpers.

The protocol is deliberately independent of CARLA, TensorBoard and the MAPPO
models.  It defines what the trainer is allowed to record and prevents a
missing reward component from being silently interpreted as a real zero.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from environment.base_reward_v3 import (
    REWARD_COMPONENTS as REWARD_V3_COMPONENTS,
    REWARD_DIAGNOSTICS as REWARD_V3_DIAGNOSTICS,
    reward_v3_spec,
)
from training.action_protocol import action_protocol_spec
from training.value_normalization import value_normalization_spec

LEGACY_METRICS_PROTOCOL_VERSION = "mappo_metrics_p0_00g_v3"
ACTION_METRICS_PROTOCOL_VERSION = "mappo_metrics_action_v2_p0_00g_v4"
PRE_OFFROAD_METRICS_PROTOCOL_VERSION = "mappo_metrics_valuenorm_p0_00g_v5"
METRICS_PROTOCOL_VERSION = "mappo_metrics_offroad_p0_00g_v6"
EPISODE_RECORD_VERSION = "mappo_episode_record_offroad_v2"
REWARD_AUDIT_ATOL = 1e-6
ACTION_ZERO_THRESHOLD = 0.01
ACTION_SATURATION_THRESHOLD = 0.99
ACTION_RATE_NAMES = (
    "throttle_boundary",
    "steer_boundary",
    "brake_boundary",
    "throttle_zero",
    "throttle_saturation",
    "steer_saturation",
    "brake_zero",
    "brake_saturation",
    "throttle_mode",
    "brake_mode",
    "coast_mode",
    "longitudinal_conflict",
)
POLICY_ACTION_RATE_NAMES = (
    "signed_longitudinal_deadzone",
    "signed_longitudinal_saturation",
)

REWARD_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "v1": (
        "progress",
        "speed",
        "centering",
        "yaw",
        "collision",
        "lane_invasion",
        "step",
        "success",
    ),
    "v2": (
        "progress",
        "approach",
        "speed",
        "alive",
        "centering",
        "yaw",
        "jerk",
        "steer_rate",
        "safety_lidar",
        "ttc",
        "collision",
        "lane_invasion",
        "step",
        "success",
        "time_bonus",
    ),
    "v3": REWARD_V3_COMPONENTS,
}

REWARD_DIAGNOSTICS: Dict[str, Tuple[str, ...]] = {
    "v1": (
        "progress_m",
        "speed_mps",
        "lateral_distance_m",
        "yaw_error_rad",
        "collision_event",
        "lane_invasion_sensor_event",
        "success_event",
    ),
    "v2": (
        "progress_m",
        "distance_to_goal_m",
        "speed_mps",
        "closing_speed_mps",
        "lateral_distance_m",
        "yaw_error_rad",
        "motion_gate",
        "lidar_min_normalized",
        "lidar_trigger_ray_count",
        "ttc_candidate_count",
        "ttc_trigger_count",
        "ttc_min_seconds_capped",
        "collision_event",
        "lane_invasion_sensor_event",
        "lane_deviation_penalty",
        "success_event",
    ),
    "v3": REWARD_V3_DIAGNOSTICS,
}

DISABLED_REWARD_COMPONENTS: Dict[str, frozenset[str]] = {
    "v1": frozenset(),
    # These fields remain in the payload for backward-compatible dashboards,
    # but the current v2 reward deliberately gives them no contribution.
    "v2": frozenset({"progress", "alive"}),
    "v3": frozenset(),
}

SPARSE_REWARD_COMPONENTS: Dict[str, frozenset[str]] = {
    "v1": frozenset({"collision", "lane_invasion", "success"}),
    "v2": frozenset(
        {
            "safety_lidar",
            "ttc",
            "collision",
            "success",
            "time_bonus",
        }
    ),
    "v3": frozenset(
        {
            "safety_lidar",
            "ttc",
            "lane_invasion",
            "offroad",
            "collision",
            "success",
        }
    ),
}


class MetricsProtocolError(RuntimeError):
    """Raised when metric input would make a dashboard untrustworthy."""


def env_action_diagnostics(
    action: Sequence[object],
    *,
    context: str,
) -> Dict[str, float]:
    """Classify one CARLA ``[throttle, steer, brake]`` command.

    Lower-bound occupancy and upper-bound saturation are deliberately
    separate. The production action adapter makes throttle and brake mutually
    exclusive, so a zero pedal is normally a control mode rather than evidence
    that the policy has saturated.
    """
    try:
        if len(action) != 3:
            raise MetricsProtocolError(
                f"{context}: environment action must have length 3"
            )
        throttle, steer, brake = (float(value) for value in action)
    except MetricsProtocolError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MetricsProtocolError(
            f"{context}: environment action must be a numeric length-3 sequence"
        ) from error

    values = (throttle, steer, brake)
    if not all(math.isfinite(value) for value in values):
        raise MetricsProtocolError(f"{context}: environment action contains NaN/Inf")
    if not 0.0 <= throttle <= 1.0:
        raise MetricsProtocolError(f"{context}: throttle is outside [0,1]")
    if not -1.0 <= steer <= 1.0:
        raise MetricsProtocolError(f"{context}: steer is outside [-1,1]")
    if not 0.0 <= brake <= 1.0:
        raise MetricsProtocolError(f"{context}: brake is outside [0,1]")

    throttle_zero = throttle <= ACTION_ZERO_THRESHOLD
    brake_zero = brake <= ACTION_ZERO_THRESHOLD
    throttle_active = not throttle_zero
    brake_active = not brake_zero
    conflict = throttle_active and brake_active
    throttle_mode = throttle_active and not brake_active
    brake_mode = brake_active and not throttle_active
    coast_mode = not throttle_active and not brake_active

    return {
        # Backward-compatible combined boundary indicators. These remain
        # observable but must not be used as throttle/brake saturation gates.
        "throttle_boundary": float(
            throttle_zero or throttle >= ACTION_SATURATION_THRESHOLD
        ),
        "steer_boundary": float(abs(steer) >= ACTION_SATURATION_THRESHOLD),
        "brake_boundary": float(
            brake_zero or brake >= ACTION_SATURATION_THRESHOLD
        ),
        # Decision-safe diagnostics.
        "throttle_zero": float(throttle_zero),
        "throttle_saturation": float(
            throttle >= ACTION_SATURATION_THRESHOLD
        ),
        "steer_saturation": float(
            abs(steer) >= ACTION_SATURATION_THRESHOLD
        ),
        "brake_zero": float(brake_zero),
        "brake_saturation": float(brake >= ACTION_SATURATION_THRESHOLD),
        "throttle_mode": float(throttle_mode),
        "brake_mode": float(brake_mode),
        "coast_mode": float(coast_mode),
        "longitudinal_conflict": float(conflict),
    }


def _strict_bool(value: object, *, field: str, context: str) -> bool:
    if not isinstance(value, bool):
        raise MetricsProtocolError(
            f"{context}: {field} must be a JSON boolean, got {value!r}"
        )
    return value


def _nonnegative_int(value: object, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise MetricsProtocolError(
            f"{context}: {field} must be an integer, got {value!r}"
        )
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise MetricsProtocolError(
            f"{context}: {field} must be an integer, got {value!r}"
        ) from error
    if integer < 0 or integer != value:
        raise MetricsProtocolError(
            f"{context}: {field} must be a non-negative integer, got {value!r}"
        )
    return integer


def validate_episode_record(
    payload: Mapping[str, object],
    *,
    run_id: str,
    agent_ids: Sequence[int],
    context: str,
) -> Dict[str, object]:
    """Validate and normalize one committed-or-pending episode audit record."""
    if not isinstance(payload, Mapping):
        raise MetricsProtocolError(f"{context}: episode record must be a mapping")

    expected_top_level = {
        "episode_record_version",
        "run_id",
        "update",
        "episode_index",
        "created_at",
        "scenario_name",
        "episode_length",
        "episode_end_reason",
        "success",
        "collision",
        "timeout",
        "offroad",
        "agents",
    }
    actual_top_level = set(payload)
    if actual_top_level != expected_top_level:
        raise MetricsProtocolError(
            f"{context}: episode record fields differ; "
            f"missing={sorted(expected_top_level - actual_top_level)}, "
            f"unknown={sorted(actual_top_level - expected_top_level)}"
        )
    if payload["episode_record_version"] != EPISODE_RECORD_VERSION:
        raise MetricsProtocolError(f"{context}: episode record protocol mismatch")
    if str(payload["run_id"]) != str(run_id):
        raise MetricsProtocolError(f"{context}: episode record run_id mismatch")

    update = _nonnegative_int(payload["update"], field="update", context=context)
    if update <= 0:
        raise MetricsProtocolError(f"{context}: episode update must be positive")
    episode_index = _nonnegative_int(
        payload["episode_index"], field="episode_index", context=context
    )
    episode_length = _nonnegative_int(
        payload["episode_length"], field="episode_length", context=context
    )
    if episode_length <= 0:
        raise MetricsProtocolError(f"{context}: episode_length must be positive")

    created_at = str(payload["created_at"]).strip()
    scenario_name = str(payload["scenario_name"]).strip()
    end_reason = str(payload["episode_end_reason"]).strip()
    if not created_at or not scenario_name or not end_reason:
        raise MetricsProtocolError(
            f"{context}: created_at, scenario_name and episode_end_reason must be non-empty"
        )

    success = _strict_bool(payload["success"], field="success", context=context)
    collision = _strict_bool(
        payload["collision"], field="collision", context=context
    )
    timeout = _strict_bool(payload["timeout"], field="timeout", context=context)
    offroad = _strict_bool(payload["offroad"], field="offroad", context=context)
    if sum((success, collision, timeout, offroad)) != 1:
        raise MetricsProtocolError(
            f"{context}: success/collision/timeout/offroad must contain exactly "
            "one terminal outcome"
        )

    agents_payload = payload["agents"]
    if not isinstance(agents_payload, Mapping):
        raise MetricsProtocolError(f"{context}: agents must be a mapping")
    normalized_agent_ids = tuple(int(agent_id) for agent_id in agent_ids)
    expected_agent_keys = {str(agent_id) for agent_id in normalized_agent_ids}
    if set(agents_payload) != expected_agent_keys:
        raise MetricsProtocolError(
            f"{context}: episode agent ids differ; "
            f"expected={sorted(expected_agent_keys)}, actual={sorted(agents_payload)}"
        )

    expected_agent_fields = {
        "episode_return",
        "active_decision_steps",
        "transition_count",
        "done",
        "terminated",
        "truncated",
        "bootstrap_enabled",
        "offroad_terminated",
        "first_offroad_evidence",
        "max_offroad_consecutive_ticks",
        "max_footprint_outside_fraction",
    }
    normalized_agents: Dict[str, Dict[str, object]] = {}
    for agent_id in normalized_agent_ids:
        key = str(agent_id)
        agent_payload = agents_payload[key]
        agent_context = f"{context} agent={agent_id}"
        if not isinstance(agent_payload, Mapping) or set(agent_payload) != expected_agent_fields:
            actual_fields = set(agent_payload) if isinstance(agent_payload, Mapping) else set()
            raise MetricsProtocolError(
                f"{agent_context}: episode fields differ; "
                f"missing={sorted(expected_agent_fields - actual_fields)}, "
                f"unknown={sorted(actual_fields - expected_agent_fields)}"
            )
        active_steps = _nonnegative_int(
            agent_payload["active_decision_steps"],
            field="active_decision_steps",
            context=agent_context,
        )
        transitions = _nonnegative_int(
            agent_payload["transition_count"],
            field="transition_count",
            context=agent_context,
        )
        if active_steps != transitions:
            raise MetricsProtocolError(
                f"{agent_context}: active decisions do not match transitions"
            )
        if active_steps > episode_length:
            raise MetricsProtocolError(
                f"{agent_context}: active steps exceed episode length"
            )
        done = _strict_bool(agent_payload["done"], field="done", context=agent_context)
        terminated = _strict_bool(
            agent_payload["terminated"], field="terminated", context=agent_context
        )
        truncated = _strict_bool(
            agent_payload["truncated"], field="truncated", context=agent_context
        )
        bootstrap_enabled = _strict_bool(
            agent_payload["bootstrap_enabled"],
            field="bootstrap_enabled",
            context=agent_context,
        )
        offroad_terminated = _strict_bool(
            agent_payload["offroad_terminated"],
            field="offroad_terminated",
            context=agent_context,
        )
        if terminated and truncated:
            raise MetricsProtocolError(
                f"{agent_context}: terminated and truncated cannot both be true"
            )
        if done != (terminated or truncated):
            raise MetricsProtocolError(
                f"{agent_context}: done must equal terminated or truncated"
            )
        if bootstrap_enabled != truncated:
            raise MetricsProtocolError(
                f"{agent_context}: bootstrap_enabled must equal truncated"
            )
        if offroad_terminated and not terminated:
            raise MetricsProtocolError(
                f"{agent_context}: offroad_terminated requires terminated=True"
            )
        max_offroad_ticks = _nonnegative_int(
            agent_payload["max_offroad_consecutive_ticks"],
            field="max_offroad_consecutive_ticks",
            context=agent_context,
        )
        max_outside_fraction = _finite_float(
            agent_payload["max_footprint_outside_fraction"],
            field="max_footprint_outside_fraction",
            context=agent_context,
        )
        if not 0.0 <= max_outside_fraction <= 1.0:
            raise MetricsProtocolError(
                f"{agent_context}: max_footprint_outside_fraction must be in [0,1]"
            )
        first_evidence = agent_payload["first_offroad_evidence"]
        normalized_first_evidence = None
        if first_evidence is not None:
            expected_evidence_fields = {
                "episode_step",
                "world_frame",
                "location",
                "yaw",
                "footprint_outside_count",
                "footprint_outside_fraction",
            }
            if (
                not isinstance(first_evidence, Mapping)
                or set(first_evidence) != expected_evidence_fields
            ):
                raise MetricsProtocolError(
                    f"{agent_context}: first_offroad_evidence fields differ"
                )
            evidence_step = _nonnegative_int(
                first_evidence["episode_step"],
                field="first_offroad_evidence.episode_step",
                context=agent_context,
            )
            if evidence_step <= 0:
                raise MetricsProtocolError(
                    f"{agent_context}: first offroad step must be positive"
                )
            world_frame = first_evidence["world_frame"]
            if isinstance(world_frame, bool) or not isinstance(world_frame, int):
                raise MetricsProtocolError(
                    f"{agent_context}: first offroad world_frame must be an integer"
                )
            location = first_evidence["location"]
            if not isinstance(location, Mapping) or set(location) != {"x", "y", "z"}:
                raise MetricsProtocolError(
                    f"{agent_context}: first offroad location fields differ"
                )
            outside_count = _nonnegative_int(
                first_evidence["footprint_outside_count"],
                field="first_offroad_evidence.footprint_outside_count",
                context=agent_context,
            )
            if outside_count > 8:
                raise MetricsProtocolError(
                    f"{agent_context}: first offroad outside count exceeds eight"
                )
            first_fraction = _finite_float(
                first_evidence["footprint_outside_fraction"],
                field="first_offroad_evidence.footprint_outside_fraction",
                context=agent_context,
            )
            if not 0.0 <= first_fraction <= 1.0:
                raise MetricsProtocolError(
                    f"{agent_context}: first offroad fraction must be in [0,1]"
                )
            normalized_first_evidence = {
                "episode_step": evidence_step,
                "world_frame": int(world_frame),
                "location": {
                    axis: _finite_float(
                        location[axis],
                        field=f"first_offroad_evidence.location.{axis}",
                        context=agent_context,
                    )
                    for axis in ("x", "y", "z")
                },
                "yaw": _finite_float(
                    first_evidence["yaw"],
                    field="first_offroad_evidence.yaw",
                    context=agent_context,
                ),
                "footprint_outside_count": outside_count,
                "footprint_outside_fraction": first_fraction,
            }
        normalized_agents[key] = {
            "episode_return": _finite_float(
                agent_payload["episode_return"],
                field="episode_return",
                context=agent_context,
            ),
            "active_decision_steps": active_steps,
            "transition_count": transitions,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "bootstrap_enabled": bootstrap_enabled,
            "offroad_terminated": offroad_terminated,
            "first_offroad_evidence": normalized_first_evidence,
            "max_offroad_consecutive_ticks": max_offroad_ticks,
            "max_footprint_outside_fraction": max_outside_fraction,
        }

    if offroad != any(
        agent["offroad_terminated"] for agent in normalized_agents.values()
    ):
        raise MetricsProtocolError(
            f"{context}: top-level offroad must match per-agent offroad termination"
        )

    return {
        "episode_record_version": EPISODE_RECORD_VERSION,
        "run_id": str(run_id),
        "update": update,
        "episode_index": episode_index,
        "created_at": created_at,
        "scenario_name": scenario_name,
        "episode_length": episode_length,
        "episode_end_reason": end_reason,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "offroad": offroad,
        "agents": normalized_agents,
    }


def reward_components(reward_version: str) -> Tuple[str, ...]:
    try:
        return REWARD_COMPONENTS[str(reward_version)]
    except KeyError as error:
        raise MetricsProtocolError(
            f"unsupported reward metrics version: {reward_version!r}"
        ) from error


def reward_diagnostics(reward_version: str) -> Tuple[str, ...]:
    try:
        return REWARD_DIAGNOSTICS[str(reward_version)]
    except KeyError as error:
        raise MetricsProtocolError(
            f"unsupported reward diagnostics version: {reward_version!r}"
        ) from error


def reward_schema_document(reward_version: str) -> Dict[str, object]:
    components = reward_components(reward_version)
    diagnostics = reward_diagnostics(reward_version)
    disabled = DISABLED_REWARD_COMPONENTS[reward_version]
    sparse = SPARSE_REWARD_COMPONENTS[reward_version]
    document = {
        "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
        "action_protocol": action_protocol_spec(),
        "critic_protocol": value_normalization_spec(),
        "reward_version": reward_version,
        "reward_components": {
            name: {
                "enabled": name not in disabled,
                "sparse": name in sparse,
                "value_semantics": "weighted contribution included in env reward",
            }
            for name in components
        },
        "reward_diagnostics": {
            name: {"value_semantics": "raw finite environment diagnostic"}
            for name in diagnostics
        },
        "aggregation": {
            "mean_per_active_step": "mean over transitions owned by this agent",
            "sum_per_episode_mean": "mean episode sum with episodes equally weighted",
            "nonzero_rate": "non-zero active transitions / active transitions",
            "nonzero_mean": "mean over non-zero active transitions only",
            "team_mean_agent_episode_sum": "mean of per-agent episode-sum means",
            "pooled_mean_per_active_step": "transition-weighted mean across agents",
        },
        "reward_audit_atol": REWARD_AUDIT_ATOL,
    }
    if reward_version == "v3":
        document["reward_protocol"] = reward_v3_spec()
    return document


def _finite_float(value: object, *, field: str, context: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise MetricsProtocolError(
            f"{context}: {field} must be a numeric scalar, got {value!r}"
        ) from error
    if not math.isfinite(scalar):
        raise MetricsProtocolError(
            f"{context}: {field} contains NaN/Inf: {scalar!r}"
        )
    return scalar


def _validate_exact_fields(
    payload: Mapping[str, object],
    expected_fields: Iterable[str],
    *,
    field_group: str,
    context: str,
) -> Dict[str, float]:
    if not isinstance(payload, Mapping):
        raise MetricsProtocolError(
            f"{context}: {field_group} must be a mapping"
        )
    expected = tuple(expected_fields)
    expected_set = set(expected)
    actual_set = set(payload)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing or unknown:
        raise MetricsProtocolError(
            f"{context}: {field_group} schema mismatch; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        name: _finite_float(
            payload[name], field=f"{field_group}.{name}", context=context
        )
        for name in expected
    }


def validate_active_reward_payload(
    *,
    active_agent_ids: Iterable[int],
    rewards: Mapping[int, object],
    reward_details_by_agent: Mapping[int, Mapping[str, object]],
    reward_diagnostics_by_agent: Mapping[int, Mapping[str, object]],
    reward_version: str,
    context: str,
    atol: float = REWARD_AUDIT_ATOL,
) -> Tuple[
    Dict[int, float],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    Dict[int, float],
]:
    """Validate and normalize one environment step for active agents only."""
    active_ids = tuple(int(agent_id) for agent_id in active_agent_ids)
    if len(set(active_ids)) != len(active_ids):
        raise MetricsProtocolError(f"{context}: duplicate active agent ids")
    if not active_ids:
        raise MetricsProtocolError(f"{context}: no active agent to record")
    if not isinstance(rewards, Mapping):
        raise MetricsProtocolError(f"{context}: rewards must be a mapping")
    if not isinstance(reward_details_by_agent, Mapping):
        raise MetricsProtocolError(
            f"{context}: reward_details must be a mapping"
        )
    if not isinstance(reward_diagnostics_by_agent, Mapping):
        raise MetricsProtocolError(
            f"{context}: reward_diagnostics must be a mapping"
        )

    component_fields = reward_components(reward_version)
    diagnostic_fields = reward_diagnostics(reward_version)
    normalized_rewards: Dict[int, float] = {}
    normalized_details: Dict[int, Dict[str, float]] = {}
    normalized_diagnostics: Dict[int, Dict[str, float]] = {}
    residuals: Dict[int, float] = {}

    for agent_id in active_ids:
        agent_context = f"{context} agent={agent_id}"
        if agent_id not in rewards:
            raise MetricsProtocolError(
                f"{agent_context}: active agent reward is missing"
            )
        if agent_id not in reward_details_by_agent:
            raise MetricsProtocolError(
                f"{agent_context}: active agent reward_details are missing"
            )
        if agent_id not in reward_diagnostics_by_agent:
            raise MetricsProtocolError(
                f"{agent_context}: active agent reward_diagnostics are missing"
            )
        reward = _finite_float(
            rewards[agent_id], field="reward", context=agent_context
        )
        details = _validate_exact_fields(
            reward_details_by_agent[agent_id],
            component_fields,
            field_group="reward_details",
            context=agent_context,
        )
        diagnostics = _validate_exact_fields(
            reward_diagnostics_by_agent[agent_id],
            diagnostic_fields,
            field_group="reward_diagnostics",
            context=agent_context,
        )
        residual = reward - math.fsum(details.values())
        if abs(residual) > atol:
            raise MetricsProtocolError(
                f"{agent_context}: reward conservation failed: reward={reward}, "
                f"sum(components)={math.fsum(details.values())}, "
                f"residual={residual}, atol={atol}"
            )
        normalized_rewards[agent_id] = reward
        normalized_details[agent_id] = details
        normalized_diagnostics[agent_id] = diagnostics
        residuals[agent_id] = residual

    return (
        normalized_rewards,
        normalized_details,
        normalized_diagnostics,
        residuals,
    )
