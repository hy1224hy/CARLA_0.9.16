"""Versioned metric schemas and strict reward-audit helpers.

The protocol is deliberately independent of CARLA, TensorBoard and the MAPPO
models.  It defines what the trainer is allowed to record and prevents a
missing reward component from being silently interpreted as a real zero.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Tuple


METRICS_PROTOCOL_VERSION = "mappo_metrics_p0_00f_tb_v1"
REWARD_AUDIT_ATOL = 1e-6

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
}

DISABLED_REWARD_COMPONENTS: Dict[str, frozenset[str]] = {
    "v1": frozenset(),
    # These fields remain in the payload for backward-compatible dashboards,
    # but the current v2 reward deliberately gives them no contribution.
    "v2": frozenset({"progress", "alive"}),
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
}


class MetricsProtocolError(RuntimeError):
    """Raised when metric input would make a dashboard untrustworthy."""


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
    return {
        "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
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
