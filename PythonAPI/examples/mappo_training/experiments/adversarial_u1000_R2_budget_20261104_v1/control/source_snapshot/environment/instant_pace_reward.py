"""CARLA-independent candidate efficiency and overspeed costs.

``instant_pace_v2`` replaces the v6 pace component and the inherited v4
overspeed component.  Existing profiles are not changed by this module.
The efficiency input is the velocity projected toward the route goal, not
the velocity magnitude.  No accumulated lateness state is retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Tuple

from environment.base_reward_v6 import PaceDeficitDecision


PROFILE = "instant_pace_v2"


@dataclass(frozen=True)
class InstantPaceConfig:
    reference_speed_mps: float = 2.5
    cost_per_second: float = 0.5
    grace_seconds: float = 2.0
    near_goal_distance_m: float = 5.0
    overspeed_start_mps: float = 15.0 / 3.6
    overspeed_scale_mps: float = 15.0 / 3.6
    overspeed_cost_per_second: float = 0.2


CONFIG = InstantPaceConfig()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be numeric, not boolean or text")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_instant_pace_config(config: InstantPaceConfig) -> None:
    if not isinstance(config, InstantPaceConfig):
        raise ValueError("instant pace config must be InstantPaceConfig")
    values = {
        name: _finite(value, name=name)
        for name, value in asdict(config).items()
    }
    for name in (
        "reference_speed_mps", "cost_per_second", "overspeed_start_mps",
        "overspeed_scale_mps", "overspeed_cost_per_second",
    ):
        if values[name] <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("grace_seconds", "near_goal_distance_m"):
        if values[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if values["reference_speed_mps"] > values["overspeed_start_mps"]:
        raise ValueError("pace reference cannot exceed overspeed start")


validate_instant_pace_config(CONFIG)


def instant_pace_deficit(
    forward_speed_mps: object,
    *,
    distance_to_goal_m: object,
    elapsed_seconds: object,
    timestep: object,
    eligible: bool = True,
    front_blocker: bool = False,
    complex_interaction: bool = False,
    terminal_event: bool = False,
    config: InstantPaceConfig = CONFIG,
) -> PaceDeficitDecision:
    """Return an immediate own-route speed cost and its exemption evidence.

    ``forward_speed_mps`` may be negative for backward movement.  Such motion
    cannot reduce the deficit below the cost of standing still.  ``raw_deficit``
    is a speed gap in m/s; unlike route_pace_v1 it is not a distance debt.
    Risk is computed before exemptions so a neutral exempt row stays auditable.
    ``eligible`` must come from the caller's explicit free-flow scene allowlist.
    """
    validate_instant_pace_config(config)
    forward = _finite(forward_speed_mps, name="forward speed")
    distance = _finite(distance_to_goal_m, name="distance to goal")
    elapsed = _finite(elapsed_seconds, name="elapsed seconds")
    dt = _finite(timestep, name="timestep")
    if distance < 0.0 or elapsed < 0.0:
        raise ValueError("distance and elapsed seconds must be non-negative")
    if dt <= 0.0:
        raise ValueError("timestep must be positive")
    flags = (eligible, front_blocker, complex_interaction, terminal_event)
    if not all(isinstance(flag, bool) for flag in flags):
        raise ValueError("instant pace eligibility flags must be booleans")

    raw_deficit = max(0.0, config.reference_speed_mps - max(0.0, forward))
    risk = min(1.0, raw_deficit / config.reference_speed_mps)
    grace_exempt = elapsed <= config.grace_seconds
    near_goal_exempt = distance <= config.near_goal_distance_m
    active = bool(
        eligible and not front_blocker and not complex_interaction
        and not grace_exempt and not near_goal_exempt and not terminal_event
    )
    reward = -config.cost_per_second * dt * risk if active else 0.0
    if not math.isfinite(reward):
        raise ValueError("instant pace reward overflow")
    return PaceDeficitDecision(
        reward=float(reward), raw_deficit=float(raw_deficit), risk=float(risk),
        eligible=active, grace_exempt=grace_exempt,
        near_goal_exempt=near_goal_exempt, terminal_exempt=terminal_event,
    )


def continuous_overspeed_penalty(
    speed_mps: object,
    *,
    timestep: object,
    config: InstantPaceConfig = CONFIG,
) -> Tuple[float, float]:
    """Return a non-saturating quadratic speed cost and dimensionless risk.

    The risk may exceed one.  There is no hard ceiling at 30 km/h.  This
    function does not decide lifecycle eligibility; the environment must call
    it only for the transitions to which the speed component applies.
    """
    validate_instant_pace_config(config)
    speed = _finite(speed_mps, name="speed")
    dt = _finite(timestep, name="timestep")
    if speed < 0.0:
        raise ValueError("speed must be non-negative")
    if dt <= 0.0:
        raise ValueError("timestep must be positive")
    excess = max(0.0, speed - config.overspeed_start_mps)
    normalized = excess / config.overspeed_scale_mps
    risk = normalized * normalized
    reward = -config.overspeed_cost_per_second * dt * risk
    if not math.isfinite(risk) or not math.isfinite(reward):
        raise ValueError("continuous overspeed reward overflow")
    return float(reward), float(risk)


def instant_pace_spec(config: InstantPaceConfig = CONFIG) -> Dict[str, object]:
    validate_instant_pace_config(config)
    return {
        "profile": PROFILE,
        "status": "candidate_parameters_pending_live_validation",
        "parameters": asdict(config),
        "parent": "v6; replaces free_flow_pace_deficit and inherited overspeed",
        "replaced_components": ["free_flow_pace_deficit", "overspeed"],
        "retained_components": "all other v6 components and terminal definitions",
        "pace": {
            "formula": "-cost_per_second * dt * clip((v_ref - max(0, v_forward))/v_ref, 0, 1)",
            "raw_deficit_units": "m/s",
            "direction": "actual velocity projected toward route goal; lateral speed alone does not count",
            "clock": "elapsed episode seconds only for startup grace; no accumulated pace debt",
            "eligibility": "explicit free-flow scene allowlist supplied by caller",
            "exemptions": ["startup_grace", "front_blocker", "complex_interaction", "near_goal", "terminal"],
            "grace_boundary": "elapsed_seconds <= grace_seconds",
            "near_goal_boundary": "distance_to_goal_m <= near_goal_distance_m",
        },
        "overspeed": {
            "formula": "-overspeed_cost_per_second * dt * (max(0, speed - upper)/scale)^2",
            "risk_range": "[0, infinity); no finite saturation threshold",
            "threshold_meaning": "task speed preference; not a claim about legal road speed limit",
        },
        "positive_dense_components": ["progress"],
        "policy_invariance_guarantee": False,
        "observation_action_change": False,
        "anti_hacking_invariants": [
            "current_speed_recovery_clears_pace_cost_immediately",
            "negative_forward_speed_does_not_count_as_progress",
            "speed_alone_has_no_positive_reward",
            "no_teammate_speed_or_arrival_gap_component",
            "costs_scale_with_elapsed_simulation_time",
        ],
    }
