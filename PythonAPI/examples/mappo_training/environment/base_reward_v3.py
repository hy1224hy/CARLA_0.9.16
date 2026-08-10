"""CARLA-independent reward-v3 mathematics for the base MAPPO policy.

The design deliberately keeps the task objective small and auditable:

* bounded route-completion potential is the only positive dense task signal;
* success/collision are explicit terminal outcomes;
* comfort and risk terms are bounded costs;
* stationary, cyclic and near-goal lingering trajectories cannot farm reward.

All constants are dimensionless reward-budget choices.  Physical quantities
are normalized before a weight is applied, so the values can be tested without
starting CARLA and can be recorded exactly in experiment evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Optional, Tuple


REWARD_VERSION = "v3"
REWARD_PROTOCOL_VERSION = "base_route_completion_reward_v3"

REWARD_COMPONENTS = (
    "progress",
    "overspeed",
    "longitudinal_rate",
    "steer_rate",
    "safety_lidar",
    "ttc",
    "lane_deviation",
    "lane_invasion",
    "collision",
    "success",
)

REWARD_DIAGNOSTICS = (
    "progress_m",
    "route_completion_prev",
    "route_completion",
    "distance_to_goal_m",
    "speed_mps",
    "closing_speed_mps",
    "overspeed_risk",
    "longitudinal_command",
    "longitudinal_rate_normalized",
    "steer_rate_normalized",
    "lidar_min_normalized",
    "lidar_trigger_ray_count",
    "lidar_risk",
    "ttc_candidate_count",
    "ttc_trigger_count",
    "ttc_min_seconds_capped",
    "ttc_risk",
    "lateral_distance_m",
    "lane_deviation_risk",
    "yaw_error_rad",
    "collision_event",
    "lane_invasion_sensor_event",
    "success_event",
)


class RewardV3Error(ValueError):
    """Raised when reward-v3 input violates a frozen mathematical boundary."""


@dataclass(frozen=True)
class RewardV3Config:
    """Frozen reward-v3 budget.

    The terminal and route scales make the discounted task-level ordering
    algebraically visible before any rollout:

    ``success <= +20``, ``safe partial progress in (0,+10)``, ``stopped = 0``
    and ``collision <= -10`` before bounded comfort/risk costs.
    """

    route_potential_scale: float = 10.0
    success_bonus: float = 10.0
    collision_penalty: float = -20.0

    overspeed_start_mps: float = 15.0 / 3.6
    overspeed_full_mps: float = 30.0 / 3.6
    overspeed_weight: float = 0.02

    control_rate_reference_per_s: float = 2.0
    longitudinal_rate_weight: float = 0.005
    steer_rate_weight: float = 0.005

    lidar_threshold_normalized: float = 0.5
    lidar_risk_weight: float = 0.01

    ttc_threshold_s: float = 3.0
    ttc_tau_s: float = 1.0
    ttc_risk_weight: float = 0.02

    lane_deviation_deadzone_m: float = 0.5
    lane_deviation_full_penalty_m: float = 1.5
    lane_deviation_weight: float = 0.01

    broken_line_penalty: float = -0.5
    solid_line_penalty: float = -2.0
    curb_penalty: float = -4.0


DEFAULT_REWARD_V3_CONFIG = RewardV3Config()


def validate_reward_v3_config(config: RewardV3Config) -> None:
    """Fail fast when a configured budget breaks the registered hierarchy."""
    if not isinstance(config, RewardV3Config):
        raise RewardV3Error("reward v3 config must be RewardV3Config")
    for name, raw_value in asdict(config).items():
        _finite(raw_value, name=name)
    positive = {
        "route_potential_scale": config.route_potential_scale,
        "success_bonus": config.success_bonus,
        "overspeed_weight": config.overspeed_weight,
        "control_rate_reference_per_s": config.control_rate_reference_per_s,
        "longitudinal_rate_weight": config.longitudinal_rate_weight,
        "steer_rate_weight": config.steer_rate_weight,
        "lidar_risk_weight": config.lidar_risk_weight,
        "ttc_threshold_s": config.ttc_threshold_s,
        "ttc_tau_s": config.ttc_tau_s,
        "ttc_risk_weight": config.ttc_risk_weight,
        "lane_deviation_weight": config.lane_deviation_weight,
    }
    for name, raw_value in positive.items():
        value = _finite(raw_value, name=name)
        if value <= 0.0:
            raise RewardV3Error(f"{name} must be positive")
    if config.collision_penalty >= -config.route_potential_scale:
        raise RewardV3Error(
            "collision penalty magnitude must exceed the complete-route scale"
        )
    if not (
        0.0 < config.overspeed_start_mps < config.overspeed_full_mps
    ):
        raise RewardV3Error("overspeed thresholds are invalid")
    if not 0.0 < config.lidar_threshold_normalized <= 1.0:
        raise RewardV3Error("lidar threshold must be in (0,1]")
    if not (
        0.0 <= config.lane_deviation_deadzone_m
        < config.lane_deviation_full_penalty_m
    ):
        raise RewardV3Error("lane-deviation thresholds are invalid")
    if not (
        0.0 >= config.broken_line_penalty
        >= config.solid_line_penalty
        >= config.curb_penalty
    ):
        raise RewardV3Error(
            "lane penalties must be non-positive and ordered Broken >= Solid >= Curb"
        )


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RewardV3Error(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise RewardV3Error(f"{name} contains NaN/Inf")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise RewardV3Error(f"{name} must be in [0,1]")
    return result


def _signed_unit_interval(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not -1.0 <= result <= 1.0:
        raise RewardV3Error(f"{name} must be in [-1,1]")
    return result


def validate_gamma(gamma: object) -> float:
    result = _finite(gamma, name="reward gamma")
    if not 0.0 < result <= 1.0:
        raise RewardV3Error("reward gamma must be in (0,1]")
    return result


def reward_v3_spec(
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Dict[str, object]:
    """Return the immutable machine-readable reward design."""
    validate_reward_v3_config(config)
    return {
        "reward_version": REWARD_VERSION,
        "reward_protocol_version": REWARD_PROTOCOL_VERSION,
        "components": list(REWARD_COMPONENTS),
        "diagnostics": list(REWARD_DIAGNOSTICS),
        "parameters": asdict(config),
        "disabled_legacy_dense_terms": [
            "alive",
            "approach",
            "centering",
            "yaw",
            "step",
            "time_bonus",
        ],
        "anti_hacking_invariants": [
            "discounted_route_potential_telescopes",
            "stationary_start_return_is_zero",
            "forward_reverse_cycle_cannot_farm_progress",
            "near_goal_lingering_is_discounted",
            "positive_speed_alone_has_no_reward",
            "throttle_and_brake_rate_costs_are_symmetric",
            "all_dense_costs_are_bounded_per_step",
            "lidar_risk_is_ray_count_invariant",
            "lane_event_penalty_is_marking_count_invariant",
            "ttc_risk_is_continuous_at_threshold",
        ],
    }


def route_completion(initial_distance_m: object, distance_to_goal_m: object) -> float:
    """Return clipped route completion in ``[0,1]``."""
    initial = _finite(initial_distance_m, name="initial distance")
    current = _finite(distance_to_goal_m, name="distance to goal")
    if initial <= 0.0:
        raise RewardV3Error("initial distance must be positive")
    if current < 0.0:
        raise RewardV3Error("distance to goal cannot be negative")
    return min(1.0, max(0.0, (initial - current) / initial))


def route_potential_reward(
    previous_completion: object,
    current_completion: object,
    *,
    gamma: object,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> float:
    """Compute ``scale * (gamma*q_next - q_prev)``.

    When PPO applies the same discount ``gamma``, the discounted sum telescopes
    to ``scale * (gamma**T*q_T - q_0)``.  Back-and-forth motion therefore
    cannot manufacture progress return by collecting an early positive delta
    and postponing its compensating negative delta.
    """
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
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> float:
    """Closed-form discounted return of the route-potential component."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise RewardV3Error("steps must be a non-negative integer")
    initial = _unit_interval(initial_completion, name="initial completion")
    final = _unit_interval(final_completion, name="final completion")
    discount = validate_gamma(gamma)
    return config.route_potential_scale * (
        (discount ** steps) * final - initial
    )


def overspeed_penalty(
    closing_speed_mps: object,
    *,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float]:
    """Return bounded overspeed penalty and normalized overspeed risk.

    Speed below the registered upper comfort boundary receives no reward.  It
    is progress, not wheel speed, that advances the task, which prevents
    lateral motion or stationary wheel activity from farming a positive term.
    """
    speed = _finite(closing_speed_mps, name="closing speed")
    start = config.overspeed_start_mps
    full = config.overspeed_full_mps
    if start <= 0.0 or full <= start:
        raise RewardV3Error("overspeed thresholds are invalid")
    risk = min(1.0, max(0.0, (speed - start) / (full - start)))
    return -config.overspeed_weight * risk, risk


def normalized_rate_cost(
    current: object,
    previous: object,
    *,
    timestep_s: object,
    reference_rate_per_s: object,
) -> float:
    """Return a bounded quadratic normalized command-rate cost in ``[0,1]``."""
    current_value = _signed_unit_interval(current, name="current command")
    previous_value = _signed_unit_interval(previous, name="previous command")
    timestep = _finite(timestep_s, name="timestep")
    reference = _finite(reference_rate_per_s, name="reference command rate")
    if timestep <= 0.0 or reference <= 0.0:
        raise RewardV3Error("timestep and reference command rate must be positive")
    normalized = abs(current_value - previous_value) / (timestep * reference)
    return min(1.0, normalized * normalized)


def control_rate_penalties(
    *,
    current_longitudinal: object,
    previous_longitudinal: object,
    current_steer: object,
    previous_steer: object,
    timestep_s: object,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float, float, float]:
    """Return longitudinal/steer penalties and their normalized costs."""
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


def lidar_risk_penalty(
    normalized_distances: Iterable[object],
    *,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float, float]:
    """Return nearest-ray bounded risk, penalty and minimum distance.

    A maximum is used rather than a sum, so changing the number of lidar rays
    cannot silently multiply the safety reward budget.
    """
    distances = [
        _unit_interval(value, name="normalized lidar distance")
        for value in normalized_distances
    ]
    if not distances:
        raise RewardV3Error("lidar distances cannot be empty")
    minimum = min(distances)
    threshold = config.lidar_threshold_normalized
    if not 0.0 < threshold <= 1.0:
        raise RewardV3Error("lidar threshold must be in (0,1]")
    risk = min(1.0, max(0.0, (threshold - minimum) / threshold))
    return -config.lidar_risk_weight * risk, risk, minimum


def ttc_risk_penalty(
    minimum_ttc_s: Optional[object],
    *,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float]:
    """Return a bounded TTC risk cost based on the most urgent candidate."""
    if minimum_ttc_s is None:
        return 0.0, 0.0
    ttc = _finite(minimum_ttc_s, name="minimum TTC")
    if ttc < 0.0:
        raise RewardV3Error("minimum TTC cannot be negative")
    threshold = config.ttc_threshold_s
    if ttc >= threshold:
        return 0.0, 0.0
    if config.ttc_tau_s <= 0.0:
        raise RewardV3Error("TTC tau must be positive")
    # Shift and rescale the exponential so the cost is continuous: risk is
    # exactly one at impact and exactly zero at the registered TTC threshold.
    threshold_floor = math.exp(-threshold / config.ttc_tau_s)
    risk = (
        math.exp(-ttc / config.ttc_tau_s) - threshold_floor
    ) / (1.0 - threshold_floor)
    return -config.ttc_risk_weight * risk, risk


def lane_deviation_penalty(
    lateral_distance_m: object,
    *,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float]:
    """Return bounded continuous lane-deviation cost and normalized risk."""
    lateral = _finite(lateral_distance_m, name="lateral distance")
    if lateral < 0.0:
        raise RewardV3Error("lateral distance cannot be negative")
    deadzone = config.lane_deviation_deadzone_m
    full = config.lane_deviation_full_penalty_m
    if deadzone < 0.0 or full <= deadzone:
        raise RewardV3Error("lane-deviation thresholds are invalid")
    risk = min(1.0, max(0.0, (lateral - deadzone) / (full - deadzone)))
    return -config.lane_deviation_weight * risk, risk


def lane_invasion_penalty(
    marking_types: Iterable[object],
    *,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> float:
    """Return the finite sparse lane-marking penalty for one sensor event."""
    severity = {
        "Broken": config.broken_line_penalty,
        "Solid": config.solid_line_penalty,
        "Curb": config.curb_penalty,
        "NONE": 0.0,
    }
    values = [str(value) for value in marking_types]
    if not values:
        return config.solid_line_penalty
    # A single sensor callback may report duplicate or composite markings.
    # Use the most severe violation rather than summing list cardinality, so
    # the sparse event has the frozen finite budget [curb, 0].
    return min(severity.get(value, config.solid_line_penalty) for value in values)


def terminal_rewards(
    *,
    success: bool,
    collision: bool,
    config: RewardV3Config = DEFAULT_REWARD_V3_CONFIG,
) -> Tuple[float, float]:
    """Return collision and success components with strict exclusivity."""
    if not isinstance(success, bool) or not isinstance(collision, bool):
        raise RewardV3Error("terminal flags must be booleans")
    if success and collision:
        raise RewardV3Error("success and collision cannot both be true")
    return (
        config.collision_penalty if collision else 0.0,
        config.success_bonus if success else 0.0,
    )


def discounted_return(rewards: Iterable[object], *, gamma: object) -> float:
    """Compute a finite discounted trajectory return for reward-budget tests."""
    discount = validate_gamma(gamma)
    total = 0.0
    factor = 1.0
    for index, reward in enumerate(rewards):
        total += factor * _finite(reward, name=f"reward[{index}]")
        factor *= discount
    return total


def discounted_step_budget(
    per_step_magnitude: object,
    *,
    steps: int,
    gamma: object,
) -> float:
    """Return the maximum discounted magnitude of a bounded step component."""
    magnitude = _finite(per_step_magnitude, name="per-step magnitude")
    if magnitude < 0.0:
        raise RewardV3Error("per-step magnitude cannot be negative")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise RewardV3Error("steps must be a non-negative integer")
    discount = validate_gamma(gamma)
    if steps == 0:
        return 0.0
    if discount == 1.0:
        return magnitude * steps
    return magnitude * (1.0 - discount**steps) / (1.0 - discount)


# Importing the protocol must fail immediately if a future edit makes the
# repository default internally inconsistent. Candidate sensitivity configs
# are validated explicitly by the trajectory-budget tests.
validate_reward_v3_config(DEFAULT_REWARD_V3_CONFIG)
