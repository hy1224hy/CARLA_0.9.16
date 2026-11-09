"""CARLA-independent reward-v4 mathematics for P0-T05-09/R6.

Reward v4 is a new contract; reward-v3 remains immutable historical evidence.
The only positive dense term is a discounted route-completion potential.  A
natural mission deadline is represented by an explicit terminal failure whose
magnitude increases with the unfinished route.  All other dense terms are
bounded non-positive comfort or risk costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple


REWARD_VERSION = "v4"
REWARD_PROTOCOL_VERSION = "base_route_mission_reward_v4_r6"

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


class RewardV4Error(ValueError):
    """Raised when a reward-v4 value violates the frozen contract."""


@dataclass(frozen=True)
class RewardV4Config:
    # Task budget calibrated for gamma=0.999 and a 500-step mission horizon.
    route_potential_scale: float = 20.0
    success_bonus: float = 20.0
    collision_penalty: float = -50.0
    offroad_penalty: float = -50.0
    mission_timeout_base_penalty: float = 10.0
    mission_timeout_remaining_penalty: float = 10.0

    overspeed_start_mps: float = 15.0 / 3.6
    overspeed_full_mps: float = 30.0 / 3.6
    overspeed_weight: float = 0.005

    control_rate_reference_per_s: float = 2.0
    longitudinal_rate_weight: float = 0.001
    steer_rate_weight: float = 0.001

    lidar_threshold_normalized: float = 0.5
    lidar_closing_reference_mps: float = 2.0
    lidar_risk_weight: float = 0.002

    ttc_threshold_s: float = 3.0
    ttc_tau_s: float = 1.0
    ttc_risk_weight: float = 0.005

    lane_deviation_deadzone_m: float = 0.5
    lane_deviation_full_penalty_m: float = 1.5
    lane_deviation_weight: float = 0.002
    legal_lane_change_grace_ticks: int = 10

    broken_line_penalty: float = -0.25
    solid_line_penalty: float = -2.0
    curb_penalty: float = -4.0


DEFAULT_REWARD_V4_CONFIG = RewardV4Config()


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RewardV4Error(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise RewardV4Error(f"{name} contains NaN/Inf")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise RewardV4Error(f"{name} must be in [0,1]")
    return result


def validate_reward_v4_config(config: RewardV4Config) -> None:
    if not isinstance(config, RewardV4Config):
        raise RewardV4Error("reward v4 config must be RewardV4Config")
    for name, value in asdict(config).items():
        _finite(value, name=name)
    positive = (
        "route_potential_scale",
        "success_bonus",
        "mission_timeout_base_penalty",
        "mission_timeout_remaining_penalty",
        "overspeed_weight",
        "control_rate_reference_per_s",
        "longitudinal_rate_weight",
        "steer_rate_weight",
        "lidar_closing_reference_mps",
        "lidar_risk_weight",
        "ttc_threshold_s",
        "ttc_tau_s",
        "ttc_risk_weight",
        "lane_deviation_weight",
    )
    for name in positive:
        if float(getattr(config, name)) <= 0.0:
            raise RewardV4Error(f"{name} must be positive")
    if config.collision_penalty >= -config.route_potential_scale:
        raise RewardV4Error("collision penalty must dominate complete-route potential")
    if config.offroad_penalty >= -config.route_potential_scale:
        raise RewardV4Error("offroad penalty must dominate complete-route potential")
    if not 0.0 < config.overspeed_start_mps < config.overspeed_full_mps:
        raise RewardV4Error("overspeed thresholds are invalid")
    if not 0.0 < config.lidar_threshold_normalized <= 1.0:
        raise RewardV4Error("lidar threshold must be in (0,1]")
    if not (
        0.0 <= config.lane_deviation_deadzone_m
        < config.lane_deviation_full_penalty_m
    ):
        raise RewardV4Error("lane-deviation thresholds are invalid")
    if config.legal_lane_change_grace_ticks < 0:
        raise RewardV4Error("legal lane-change grace must be non-negative")
    if not (
        0.0 >= config.broken_line_penalty
        >= config.solid_line_penalty
        >= config.curb_penalty
    ):
        raise RewardV4Error("lane penalties must be ordered Broken >= Solid >= Curb")


validate_reward_v4_config(DEFAULT_REWARD_V4_CONFIG)


def reward_v4_spec(
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Dict[str, object]:
    validate_reward_v4_config(config)
    return {
        "reward_version": REWARD_VERSION,
        "reward_protocol_version": REWARD_PROTOCOL_VERSION,
        "components": list(REWARD_COMPONENTS),
        "diagnostics": list(REWARD_DIAGNOSTICS),
        "parameters": asdict(config),
        "positive_dense_components": ["progress"],
        "disabled_positive_dense_terms": [
            "speed",
            "alive",
            "centering",
            "yaw",
            "step",
            "time_bonus",
        ],
        "terminal_priority": [
            "collision",
            "offroad",
            "success",
            "mission_timeout",
        ],
        "anti_hacking_invariants": [
            "discounted_route_potential_telescopes",
            "stationary_timeout_is_explicitly_negative",
            "unfinished_timeout_cost_increases_with_remaining_route",
            "speed_alone_has_no_positive_reward",
            "lane_center_has_no_positive_dwell_reward",
            "lidar_cost_requires_obstacle_approach",
            "ttc_uses_projected_relative_velocity",
            "stall_is_diagnostic_only",
        ],
    }


def validate_gamma(gamma: object) -> float:
    value = _finite(gamma, name="reward gamma")
    if not 0.0 < value <= 1.0:
        raise RewardV4Error("reward gamma must be in (0,1]")
    return value


def route_completion(initial_distance_m: object, distance_to_goal_m: object) -> float:
    initial = _finite(initial_distance_m, name="initial distance")
    current = _finite(distance_to_goal_m, name="distance to goal")
    if initial <= 0.0 or current < 0.0:
        raise RewardV4Error("route distances are invalid")
    return min(1.0, max(0.0, (initial - current) / initial))


def route_potential_reward(
    previous_completion: object,
    current_completion: object,
    *,
    gamma: object,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> float:
    previous = _unit_interval(previous_completion, name="previous completion")
    current = _unit_interval(current_completion, name="current completion")
    discount = validate_gamma(gamma)
    return config.route_potential_scale * (discount * current - previous)


def potential_discounted_return(
    initial_completion: object,
    final_completion: object,
    *,
    steps: int,
    gamma: object,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> float:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise RewardV4Error("steps must be a non-negative integer")
    initial = _unit_interval(initial_completion, name="initial completion")
    final = _unit_interval(final_completion, name="final completion")
    discount = validate_gamma(gamma)
    return config.route_potential_scale * (
        discount**steps * final - initial
    )


def overspeed_penalty(
    speed_mps: object,
    *,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float]:
    speed = _finite(speed_mps, name="speed")
    if speed < 0.0:
        raise RewardV4Error("speed cannot be negative")
    risk = min(
        1.0,
        max(
            0.0,
            (speed - config.overspeed_start_mps)
            / (config.overspeed_full_mps - config.overspeed_start_mps),
        ),
    )
    return -config.overspeed_weight * risk, risk


def _signed_unit(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not -1.0 <= result <= 1.0:
        raise RewardV4Error(f"{name} must be in [-1,1]")
    return result


def normalized_rate_cost(
    current: object,
    previous: object,
    *,
    timestep_s: object,
    reference_rate_per_s: object,
) -> float:
    current_value = _signed_unit(current, name="current command")
    previous_value = _signed_unit(previous, name="previous command")
    timestep = _finite(timestep_s, name="timestep")
    reference = _finite(reference_rate_per_s, name="reference rate")
    if timestep <= 0.0 or reference <= 0.0:
        raise RewardV4Error("timestep and reference rate must be positive")
    normalized = abs(current_value - previous_value) / (timestep * reference)
    return min(1.0, normalized * normalized)


def control_rate_penalties(
    *,
    current_longitudinal: object,
    previous_longitudinal: object,
    current_steer: object,
    previous_steer: object,
    timestep_s: object,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float, float, float]:
    long_cost = normalized_rate_cost(
        current_longitudinal,
        previous_longitudinal,
        timestep_s=timestep_s,
        reference_rate_per_s=config.control_rate_reference_per_s,
    )
    steer_cost = normalized_rate_cost(
        current_steer,
        previous_steer,
        timestep_s=timestep_s,
        reference_rate_per_s=config.control_rate_reference_per_s,
    )
    return (
        -config.longitudinal_rate_weight * long_cost,
        -config.steer_rate_weight * steer_cost,
        long_cost,
        steer_cost,
    )


def approach_gated_lidar_penalty(
    normalized_distances: Iterable[object],
    previous_normalized_distances: Optional[Iterable[object]],
    *,
    timestep_s: object,
    lidar_range_m: object,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float, float, float, float, float]:
    """Return penalty, risk, current/previous minima, closing speed and gate."""
    current_values = [
        _unit_interval(value, name="normalized lidar distance")
        for value in normalized_distances
    ]
    if not current_values:
        raise RewardV4Error("lidar distances cannot be empty")
    current_minimum = min(current_values)
    previous_values = (
        current_values
        if previous_normalized_distances is None
        else [
            _unit_interval(value, name="previous normalized lidar distance")
            for value in previous_normalized_distances
        ]
    )
    if len(previous_values) != len(current_values):
        raise RewardV4Error("current/previous lidar ray counts must match")
    previous_minimum = min(previous_values)
    timestep = _finite(timestep_s, name="timestep")
    lidar_range = _finite(lidar_range_m, name="lidar range")
    if timestep <= 0.0 or lidar_range <= 0.0:
        raise RewardV4Error("timestep and lidar range must be positive")
    closing_speed = max(
        0.0, (previous_minimum - current_minimum) * lidar_range / timestep
    )
    approach_gate = min(
        1.0, closing_speed / config.lidar_closing_reference_mps
    )
    proximity = min(
        1.0,
        max(
            0.0,
            (config.lidar_threshold_normalized - current_minimum)
            / config.lidar_threshold_normalized,
        ),
    )
    risk = proximity * approach_gate
    return (
        -config.lidar_risk_weight * risk,
        risk,
        current_minimum,
        previous_minimum,
        closing_speed,
        approach_gate,
    )


def projected_ttc_seconds(
    relative_position_xyz: Sequence[object],
    ego_velocity_xyz: Sequence[object],
    other_velocity_xyz: Sequence[object],
) -> Optional[float]:
    """TTC from relative position and closing velocity projected on its ray."""
    if not (
        len(relative_position_xyz)
        == len(ego_velocity_xyz)
        == len(other_velocity_xyz)
        == 3
    ):
        raise RewardV4Error("TTC vectors must have exactly three dimensions")
    relative = tuple(
        _finite(value, name="relative position")
        for value in relative_position_xyz
    )
    ego = tuple(_finite(value, name="ego velocity") for value in ego_velocity_xyz)
    other = tuple(
        _finite(value, name="other velocity") for value in other_velocity_xyz
    )
    distance = math.sqrt(sum(value * value for value in relative))
    if distance <= 1e-9:
        return 0.0
    unit = tuple(value / distance for value in relative)
    closing_speed = sum(
        (ego[index] - other[index]) * unit[index] for index in range(3)
    )
    if closing_speed <= 0.0:
        return None
    return distance / closing_speed


def ttc_risk_penalty(
    minimum_ttc_s: Optional[object],
    *,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float]:
    if minimum_ttc_s is None:
        return 0.0, 0.0
    ttc = _finite(minimum_ttc_s, name="minimum TTC")
    if ttc < 0.0:
        raise RewardV4Error("minimum TTC cannot be negative")
    if ttc >= config.ttc_threshold_s:
        return 0.0, 0.0
    floor = math.exp(-config.ttc_threshold_s / config.ttc_tau_s)
    risk = (math.exp(-ttc / config.ttc_tau_s) - floor) / (1.0 - floor)
    return -config.ttc_risk_weight * risk, risk


def lane_deviation_penalty(
    lateral_distance_m: object,
    *,
    exempt: bool = False,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float]:
    lateral = _finite(lateral_distance_m, name="lateral distance")
    if lateral < 0.0:
        raise RewardV4Error("lateral distance cannot be negative")
    if not isinstance(exempt, bool):
        raise RewardV4Error("lane-deviation exemption must be boolean")
    if exempt:
        return 0.0, 0.0
    risk = min(
        1.0,
        max(
            0.0,
            (lateral - config.lane_deviation_deadzone_m)
            / (
                config.lane_deviation_full_penalty_m
                - config.lane_deviation_deadzone_m
            ),
        ),
    )
    return -config.lane_deviation_weight * risk, risk


def lateral_distance_to_lane_center(
    *,
    vehicle_x: object,
    vehicle_y: object,
    lane_center_x: object,
    lane_center_y: object,
    lane_right_x: object,
    lane_right_y: object,
) -> float:
    dx = _finite(vehicle_x, name="vehicle x") - _finite(
        lane_center_x, name="lane centre x"
    )
    dy = _finite(vehicle_y, name="vehicle y") - _finite(
        lane_center_y, name="lane centre y"
    )
    right_x = _finite(lane_right_x, name="lane right x")
    right_y = _finite(lane_right_y, name="lane right y")
    norm = math.hypot(right_x, right_y)
    if norm <= 1e-12:
        raise RewardV4Error("lane right vector cannot be zero")
    return abs(dx * right_x / norm + dy * right_y / norm)


def is_legal_lane_change_marking(marking_types: Iterable[object]) -> bool:
    values = [str(value) for value in marking_types]
    return bool(values) and all(value == "Broken" for value in values)


def lane_invasion_penalty(
    marking_types: Iterable[object],
    *,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> float:
    severity = {
        "Broken": config.broken_line_penalty,
        "Solid": config.solid_line_penalty,
        "Curb": config.curb_penalty,
        "NONE": 0.0,
    }
    values = [str(value) for value in marking_types]
    if not values:
        return config.solid_line_penalty
    return min(severity.get(value, config.solid_line_penalty) for value in values)


def terminal_rewards(
    *,
    success: bool,
    collision: bool,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> Tuple[float, float]:
    if not isinstance(success, bool) or not isinstance(collision, bool):
        raise RewardV4Error("terminal flags must be booleans")
    if success and collision:
        raise RewardV4Error("success and collision cannot both be true")
    return (
        config.collision_penalty if collision else 0.0,
        config.success_bonus if success else 0.0,
    )


def offroad_terminal_reward(
    *,
    offroad: bool,
    collision: bool,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> float:
    if not isinstance(offroad, bool) or not isinstance(collision, bool):
        raise RewardV4Error("offroad/collision flags must be booleans")
    if collision:
        return 0.0
    return config.offroad_penalty if offroad else 0.0


def mission_timeout_reward(
    remaining_route_fraction: object,
    *,
    mission_timeout: bool,
    collision: bool = False,
    offroad: bool = False,
    success: bool = False,
    config: RewardV4Config = DEFAULT_REWARD_V4_CONFIG,
) -> float:
    if not all(
        isinstance(value, bool)
        for value in (mission_timeout, collision, offroad, success)
    ):
        raise RewardV4Error("timeout/terminal flags must be booleans")
    remaining = _unit_interval(
        remaining_route_fraction, name="remaining route fraction"
    )
    if not mission_timeout or collision or offroad or success:
        return 0.0
    return -(
        config.mission_timeout_base_penalty
        + config.mission_timeout_remaining_penalty * remaining
    )


def discounted_return(rewards: Iterable[object], *, gamma: object) -> float:
    discount = validate_gamma(gamma)
    result = 0.0
    factor = 1.0
    for index, reward in enumerate(rewards):
        result += factor * _finite(reward, name=f"reward[{index}]")
        factor *= discount
    return result
