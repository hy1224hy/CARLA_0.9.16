"""Candidate V6 costs for speed, reference-lane tracking and steering changes.

Pure reward mathematics only.  No simulator connection, action filtering or
change to the observation, goal, safety termination or low-speed contracts.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from environment.instant_pace_reward import CONFIG, InstantPaceConfig, _finite
from environment.instant_pace_reward_v5 import (
    LANE_DEVIATION_SCALE_M,
    continuous_lane_deviation_penalty as _v5_lane,
    continuous_overspeed_penalty as _v5_speed,
    instant_pace_deficit,
    instant_pace_spec as _v5_spec,
)

PROFILE = "instant_pace_v6"
SPEED_LINEAR_COST_PER_SECOND = 1.0
SPEED_QUADRATIC_COST_PER_SECOND = 1.0
SPEED_SHOULDER_COST_PER_SECOND = 2.0
LANE_CENTER_COST_PER_SECOND = 0.1
STEER_RATE_COST_PER_SECOND = 0.01
STEER_REFERENCE_RATE_PER_SECOND = 2.0


def _checked(cost: float, risk: float) -> Tuple[float, float]:
    if not math.isfinite(cost) or not math.isfinite(risk):
        raise ValueError("V6 reward cost/risk overflow")
    return float(cost), float(risk)


def continuous_overspeed_penalty(
    speed_mps: object, *, timestep: object, config: InstantPaceConfig = CONFIG,
) -> Tuple[float, float]:
    """Double V5 base cost and add a quadratic shoulder above 30 km/h."""
    _, risk = _v5_speed(speed_mps, timestep=timestep, config=config)
    excess = math.sqrt(risk)
    shoulder = max(0.0, excess - 1.0)
    cost = -float(timestep) * (
        SPEED_LINEAR_COST_PER_SECOND * excess
        + SPEED_QUADRATIC_COST_PER_SECOND * risk
        + SPEED_SHOULDER_COST_PER_SECOND * shoulder * shoulder
    )
    return _checked(cost, risk)


def continuous_lane_deviation_penalty(
    lateral_distance_m: object, *, timestep: object,
    exempt: bool = False, road_violation: bool = False,
    opposing_lane: bool = False, in_junction: bool = False,
) -> Tuple[float, float, bool]:
    """Preserve V5 grace and large-error cost; add zero-centered feedback."""
    old_cost, old_risk, effective_exempt = _v5_lane(
        lateral_distance_m, timestep=timestep, exempt=exempt,
        road_violation=road_violation, opposing_lane=opposing_lane,
        in_junction=in_junction,
    )
    centered = float(lateral_distance_m) / LANE_DEVIATION_SCALE_M
    center_risk = centered * centered
    extra_cost = (0.0 if effective_exempt else
                  -LANE_CENTER_COST_PER_SECOND * float(timestep) * center_risk)
    cost, risk = _checked(old_cost + extra_cost, old_risk + center_risk)
    return cost, risk, effective_exempt


def steer_rate_penalty(
    current_steer: object, previous_steer: object, *, timestep: object,
) -> Tuple[float, float]:
    """Penalize command rate squared without the previous early saturation.

    The environment owns history and resets it at each episode.  None means
    no prior valid command, so the first command has no invented rate cost.
    """
    dt = _finite(timestep, name="timestep")
    current = _finite(current_steer, name="current steer")
    if dt <= 0.0 or not -1.0 <= current <= 1.0:
        raise ValueError("positive timestep and bounded steer are required")
    if previous_steer is None:
        return 0.0, 0.0
    previous = _finite(previous_steer, name="previous steer")
    if not -1.0 <= previous <= 1.0:
        raise ValueError("previous steer must be within [-1,1]")
    normalized = abs(current - previous) / (dt * STEER_REFERENCE_RATE_PER_SECOND)
    risk = normalized * normalized
    return _checked(-STEER_RATE_COST_PER_SECOND * dt * risk, risk)


def instant_pace_spec(config: InstantPaceConfig = CONFIG) -> Dict[str, object]:
    spec = _v5_spec(config)
    spec.update({
        "profile": PROFILE,
        "status": "implemented_candidate_pending_new_policy_live_validation",
        "parent": "instant_pace_v5; replaces overspeed, lane_deviation and steer_rate",
        "retained_components": "V5 instantaneous pace, progress, longitudinal control, safety and terminal definitions",
    })
    for key in ("difference_from_v2", "difference_from_v3", "difference_from_v4"):
        spec.pop(key, None)
    spec["replaced_components"].append("steer_rate")
    spec["parameters"].update({
        "overspeed_linear_cost_per_second": SPEED_LINEAR_COST_PER_SECOND,
        "overspeed_quadratic_cost_per_second": SPEED_QUADRATIC_COST_PER_SECOND,
        "overspeed_shoulder_cost_per_second": SPEED_SHOULDER_COST_PER_SECOND,
        "overspeed_shoulder_normalized_excess": 1.0,
        "lane_center_cost_per_second": LANE_CENTER_COST_PER_SECOND,
        "steer_rate_cost_per_second": STEER_RATE_COST_PER_SECOND,
        "steer_reference_rate_per_second": STEER_REFERENCE_RATE_PER_SECOND,
    })
    spec["difference_from_v5"] = {
        "changed_components": ["overspeed", "lane_deviation", "steer_rate"],
        "pace_math_and_exemptions_unchanged": True,
        "terminal_math_unchanged": True,
        "causal_scope": "combined engineering candidate; component attribution requires matched ablations",
    }
    spec["overspeed"].update({
        "formula": "-dt*(e + e^2 + 2*max(0,e-1)^2); e=max(0,speed-upper)/scale",
        "risk_diagnostic": "e^2; unchanged from V5",
        "shoulder_meaning": "extra cost above upper+scale (30 km/h under default config)",
    })
    spec["lane_deviation"].update({
        "formula": "-dt*0.1*(u+u^2+(d/scale)^2) unless effectively exempt; u=max(0,d-deadzone)/scale",
        "risk_diagnostic": "unclipped u+u^2+(d/scale)^2 before effective exemption",
        "center_feedback": "zero at center; negative for every nonzero distance outside valid exemption",
    })
    spec["steer_rate"] = {
        "formula": "-dt*0.01*(abs(current_steer-previous_steer)/(2*dt))^2",
        "risk_diagnostic": "unclipped squared normalized command rate; may exceed 1",
        "first_control": "zero cost without a previous valid control",
        "history": "existing environment prev_controls; episode reset; no inactive-agent updates",
        "action_filter": False,
    }
    return spec
