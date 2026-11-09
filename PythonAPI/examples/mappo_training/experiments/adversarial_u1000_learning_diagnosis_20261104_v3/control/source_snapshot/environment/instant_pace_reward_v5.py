"""Candidate reference-lane dense feedback; v4 speed and pace stay unchanged."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from environment.instant_pace_reward import InstantPaceConfig, _finite
from environment.instant_pace_reward_v4 import (
    CONFIG, instant_pace_deficit, continuous_overspeed_penalty,
    instant_pace_spec as _v4_spec,
)


PROFILE = "instant_pace_v5"
LANE_DEVIATION_DEADZONE_M = 0.35
LANE_DEVIATION_REFERENCE_M = 1.25
LANE_DEVIATION_SCALE_M = LANE_DEVIATION_REFERENCE_M - LANE_DEVIATION_DEADZONE_M
LANE_DEVIATION_COST_PER_SECOND = 0.1


def continuous_lane_deviation_penalty(
    lateral_distance_m: object,
    *,
    timestep: object,
    exempt: bool = False,
    road_violation: bool = False,
    opposing_lane: bool = False,
    in_junction: bool = False,
) -> Tuple[float, float, bool]:
    """Return dense lane cost, uncapped raw risk and effective exemption.

    The caller supplies reference-lane lateral distance and road evidence.
    A valid legal-transition exemption remains intact.  Only confirmed road
    violation or opposing-lane occupancy outside a junction invalidates it.
    The risk is still computed for exempt rows to preserve audit evidence.
    This function introduces no terminal penalty or termination condition.
    """
    distance = _finite(lateral_distance_m, name="lateral distance")
    dt = _finite(timestep, name="timestep")
    if distance < 0.0:
        raise ValueError("lateral distance must be non-negative")
    if dt <= 0.0:
        raise ValueError("timestep must be positive")
    if not all(isinstance(flag, bool) for flag in (
        exempt, road_violation, opposing_lane, in_junction,
    )):
        raise ValueError("lane exemption and road evidence flags must be booleans")
    effective_exempt = bool(
        exempt and not ((road_violation or opposing_lane) and not in_junction)
    )
    normalized = max(0.0, distance - LANE_DEVIATION_DEADZONE_M) / LANE_DEVIATION_SCALE_M
    risk = normalized + normalized * normalized
    cost = -LANE_DEVIATION_COST_PER_SECOND * dt * risk if not effective_exempt else 0.0
    if not math.isfinite(risk) or not math.isfinite(cost):
        raise ValueError("continuous lane deviation reward overflow")
    return float(cost), float(risk), effective_exempt


def instant_pace_spec(config: InstantPaceConfig = CONFIG) -> Dict[str, object]:
    spec = _v4_spec(config)
    spec["profile"] = PROFILE
    spec["status"] = "candidate_reference_lane_feedback_pending_live_validation"
    spec["parent"] = "instant_pace_v4; replaces lane_deviation and validates its legal-transition exemption"
    spec["retained_components"] = "v4 pace, v3 overspeed and all other base components and terminal definitions"
    spec["replaced_components"].append("lane_deviation")
    spec["parameters"].update({
        "lane_deviation_deadzone_m": LANE_DEVIATION_DEADZONE_M,
        "lane_deviation_reference_m": LANE_DEVIATION_REFERENCE_M,
        "lane_deviation_scale_m": LANE_DEVIATION_SCALE_M,
        "lane_deviation_cost_per_second": LANE_DEVIATION_COST_PER_SECOND,
    })
    spec.pop("difference_from_v3", None)
    spec["difference_from_v4"] = {
        "changed_components": ["lane_deviation"],
        "pace_math_unchanged": True,
        "overspeed_math_unchanged": True,
        "terminal_math_unchanged": True,
        "purpose": "candidate dense feedback for drifting into an opposing adjacent lane; not a validated behavior fix",
    }
    spec["lane_deviation"] = {
        "formula": "u = max(0, lateral_distance_m - deadzone_m) / scale_m; risk = u + u^2; cost = -cost_per_second * dt * risk unless effectively exempt",
        "geometry": "lateral distance from the task reference lane supplied by the environment",
        "distance_units": "m",
        "risk_range": "[0, infinity); no finite saturation threshold",
        "risk_diagnostic": "raw u + u^2 even for exempt rows",
        "effective_exemption": "exempt and not ((road_violation or opposing_lane) and not in_junction)",
        "road_grace_validity": "confirmed non-junction violation or opposing-lane evidence cancels only the dense lane-cost exemption",
        "junction_handling": "retain the caller's original exemption in ambiguous junction geometry",
        "termination_change": False,
        "new_terminal_penalty": False,
        "parameter_status": "candidate deadzone, scale and strength pending live validation",
    }
    spec["anti_hacking_invariants"].append("confirmed_nonjunction_opposing_lane_cannot_keep_dense_lane_cost_exemption")
    return spec
