"""Frozen reward-v5 mathematics for P0-T05-10/R7.

Reward v5 preserves R6's discounted mission potential and bounded costs.  It
changes lane shaping from "nearest Driving lane plus Broken-line exemption" to
"current legal reference lane plus topology-approved transition exemption" and
adds an explicit persistent road-rule terminal.  No lane-centre dwell bonus is
introduced: both staying in a legal lane and completing a legal lane change are
reward-neutral except for route progress and bounded costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict

from environment.base_reward_v4 import (
    RewardV4Config,
    approach_gated_lidar_penalty,
    control_rate_penalties,
    discounted_return,
    lateral_distance_to_lane_center,
    mission_timeout_reward as _mission_timeout_reward_v4,
    normalized_rate_cost,
    offroad_terminal_reward,
    overspeed_penalty,
    potential_discounted_return,
    projected_ttc_seconds,
    route_completion,
    route_potential_reward,
    terminal_rewards,
    ttc_risk_penalty,
    validate_gamma,
    validate_reward_v4_config,
)


REWARD_VERSION = "v5"
REWARD_PROTOCOL_VERSION = "base_route_road_rule_reward_v5_r7"

REWARD_COMPONENTS = (
    "progress",
    "overspeed",
    "longitudinal_rate",
    "steer_rate",
    "safety_lidar",
    "ttc",
    "lane_deviation",
    "lane_invasion",
    "offroad",
    "road_rule",
    "collision",
    "success",
    "mission_timeout",
)

REWARD_DIAGNOSTICS = (
    "progress_m",
    "route_completion_prev",
    "route_completion",
    "remaining_route_fraction",
    "remaining_time_fraction",
    "distance_to_goal_m",
    "speed_mps",
    "closing_speed_mps",
    "overspeed_risk",
    "longitudinal_command",
    "longitudinal_rate_normalized",
    "steer_rate_normalized",
    "lidar_min_normalized",
    "lidar_previous_min_normalized",
    "lidar_trigger_ray_count",
    "lidar_closing_speed_mps",
    "lidar_approach_gate",
    "lidar_risk",
    "ttc_candidate_count",
    "ttc_trigger_count",
    "ttc_min_seconds_capped",
    "ttc_risk",
    "lateral_distance_m",
    "lane_deviation_risk",
    "legal_lane_change_event",
    "legal_lane_change_grace_remaining",
    "lane_deviation_exempted",
    "yaw_error_rad",
    "driving_center_valid",
    "footprint_outside_count",
    "footprint_outside_fraction",
    "offroad_raw",
    "offroad_consecutive_ticks",
    "offroad_terminated",
    "road_same_reference_lane",
    "road_adjacent_to_reference",
    "road_lane_change_permitted",
    "road_current_to_reference_cosine",
    "road_vehicle_to_current_cosine",
    "road_in_junction",
    "road_violation_raw",
    "road_violation_consecutive_ticks",
    "road_violation_terminated",
    "road_success_eligible",
    "collision_event",
    "lane_invasion_sensor_event",
    "success_event",
    "mission_timeout_event",
    "stall_low_speed",
    "stall_no_progress_ticks",
    "stall_raw",
    "stall_front_blocker_exempt",
    "stall_complex_interaction_exempt",
    "stall_reportable",
)


@dataclass(frozen=True)
class RewardV5Config(RewardV4Config):
    road_rule_penalty: float = -50.0
    lane_deviation_deadzone_m: float = 0.35
    lane_deviation_full_penalty_m: float = 1.25
    lane_deviation_weight: float = 0.01
    # A legal topology-approved lane change has no event cost.  Broken by
    # itself is not legal evidence; the environment applies solid-line cost to
    # every rejected crossing irrespective of marking label.
    legal_lane_change_penalty: float = 0.0
    illegal_lane_change_penalty: float = -2.0


DEFAULT_REWARD_V5_CONFIG = RewardV5Config()


def validate_reward_v5_config(config: RewardV5Config) -> None:
    if not isinstance(config, RewardV5Config):
        raise ValueError("reward v5 config must be RewardV5Config")
    validate_reward_v4_config(config)
    if config.road_rule_penalty >= -config.route_potential_scale:
        raise ValueError("road-rule penalty must dominate complete-route potential")
    if not (
        0.0 <= config.lane_deviation_deadzone_m
        < config.lane_deviation_full_penalty_m
    ):
        raise ValueError("lane-deviation thresholds are invalid")
    if config.lane_deviation_weight <= 0.0:
        raise ValueError("lane-deviation weight must be positive")
    if config.legal_lane_change_penalty != 0.0:
        raise ValueError("legal lane changes must not carry an event penalty")
    if config.illegal_lane_change_penalty >= 0.0:
        raise ValueError("illegal lane-change penalty must be negative")


validate_reward_v5_config(DEFAULT_REWARD_V5_CONFIG)


def reward_v5_spec(
    config: RewardV5Config = DEFAULT_REWARD_V5_CONFIG,
) -> Dict[str, object]:
    validate_reward_v5_config(config)
    return {
        "reward_version": REWARD_VERSION,
        "reward_protocol_version": REWARD_PROTOCOL_VERSION,
        "components": list(REWARD_COMPONENTS),
        "diagnostics": list(REWARD_DIAGNOSTICS),
        "parameters": asdict(config),
        "positive_dense_components": ["progress"],
        "lane_reference": "current_topology_approved_legal_lane",
        "legal_lane_change_reward": "no_bonus_no_event_penalty",
        "illegal_lane_change_reward": "bounded_event_cost_then_persistent_terminal",
        "terminal_priority": [
            "collision",
            "offroad",
            "road_rule",
            "success",
            "mission_timeout",
        ],
        "anti_hacking_invariants": [
            "discounted_route_potential_telescopes",
            "speed_alone_has_no_positive_reward",
            "lane_center_has_no_positive_dwell_reward",
            "broken_marking_alone_never_legalizes_a_lane_change",
            "opposing_driving_lane_is_not_treated_as_offroad",
            "road_rule_terminal_dominates_complete_route_potential",
            "success_requires_current_road_legality",
            "stall_is_diagnostic_only",
        ],
    }


def lane_deviation_penalty(
    lateral_distance_m: object,
    *,
    exempt: bool = False,
    config: RewardV5Config = DEFAULT_REWARD_V5_CONFIG,
):
    lateral = float(lateral_distance_m)
    if not math.isfinite(lateral) or lateral < 0.0:
        raise ValueError("lateral distance cannot be negative")
    if not isinstance(exempt, bool):
        raise ValueError("lane-deviation exemption must be boolean")
    if exempt:
        return 0.0, 0.0
    risk = min(
        1.0,
        max(
            0.0,
            (lateral - config.lane_deviation_deadzone_m)
            / (config.lane_deviation_full_penalty_m - config.lane_deviation_deadzone_m),
        ),
    )
    return -config.lane_deviation_weight * risk, risk


def lane_change_event_penalty(
    *,
    sensor_event: bool,
    legal_lane_change: bool,
    config: RewardV5Config = DEFAULT_REWARD_V5_CONFIG,
) -> float:
    if not isinstance(sensor_event, bool) or not isinstance(legal_lane_change, bool):
        raise ValueError("lane-change flags must be booleans")
    if not sensor_event:
        return 0.0
    return (
        config.legal_lane_change_penalty
        if legal_lane_change
        else config.illegal_lane_change_penalty
    )


def road_rule_terminal_reward(
    *,
    road_violation: bool,
    collision: bool = False,
    offroad: bool = False,
    config: RewardV5Config = DEFAULT_REWARD_V5_CONFIG,
) -> float:
    if not all(isinstance(value, bool) for value in (road_violation, collision, offroad)):
        raise ValueError("road-rule terminal flags must be booleans")
    if not road_violation or collision or offroad:
        return 0.0
    return config.road_rule_penalty


def mission_timeout_reward(
    remaining_route_fraction: object,
    *,
    mission_timeout: bool,
    collision: bool = False,
    offroad: bool = False,
    road_violation: bool = False,
    success: bool = False,
    config: RewardV5Config = DEFAULT_REWARD_V5_CONFIG,
) -> float:
    if not isinstance(road_violation, bool):
        raise ValueError("road_violation must be boolean")
    if road_violation:
        return 0.0
    return _mission_timeout_reward_v4(
        remaining_route_fraction,
        mission_timeout=mission_timeout,
        collision=collision,
        offroad=offroad,
        success=success,
        config=config,
    )
