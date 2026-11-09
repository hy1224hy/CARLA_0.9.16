"""Candidate goal-approach pace: slow movement replaces the v3 exemption.

The v3 speed cost is retained.  Outside the goal, being within five metres
does not establish arrival and no longer makes indefinite stopping free.
"""

from __future__ import annotations

import math
from typing import Dict

from environment.base_reward_v6 import PaceDeficitDecision
from environment.instant_pace_reward import (
    InstantPaceConfig, instant_pace_deficit as _v2_pace_deficit,
)
from environment.instant_pace_reward_v3 import (
    CONFIG, continuous_overspeed_penalty,
    instant_pace_spec as _v3_spec,
)


PROFILE = "instant_pace_v4"
APPROACH_REFERENCE_SPEED_MPS = 0.5


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
    """Charge current speed deficit against a far or goal-approach reference.

    The original pure function validates every input and supplies unchanged
    startup and terminal evidence.  Its near-goal exemption is deliberately
    replaced with a lower reference speed.  ``raw_deficit`` is in m/s against
    that effective reference, and ``near_goal_exempt`` is always false.
    The caller must pass successful arrival as ``terminal_event``.
    """
    original = _v2_pace_deficit(
        forward_speed_mps, distance_to_goal_m=distance_to_goal_m,
        elapsed_seconds=elapsed_seconds, timestep=timestep,
        eligible=eligible, front_blocker=front_blocker,
        complex_interaction=complex_interaction, terminal_event=terminal_event,
        config=config,
    )
    if config.reference_speed_mps < APPROACH_REFERENCE_SPEED_MPS:
        raise ValueError("far pace reference cannot be lower than approach reference")
    effective_reference = (
        APPROACH_REFERENCE_SPEED_MPS
        if float(distance_to_goal_m) <= config.near_goal_distance_m
        else config.reference_speed_mps
    )
    raw_deficit = max(0.0, effective_reference - max(0.0, float(forward_speed_mps)))
    risk = min(1.0, raw_deficit / effective_reference)
    active = bool(
        eligible and not front_blocker and not complex_interaction
        and not original.grace_exempt and not terminal_event
    )
    reward = -config.cost_per_second * float(timestep) * risk if active else 0.0
    if not math.isfinite(reward):
        raise ValueError("v4 instant pace reward overflow")
    return PaceDeficitDecision(
        reward=float(reward), raw_deficit=float(raw_deficit), risk=float(risk),
        eligible=active, grace_exempt=original.grace_exempt,
        near_goal_exempt=False, terminal_exempt=terminal_event,
    )


def instant_pace_spec(config: InstantPaceConfig = CONFIG) -> Dict[str, object]:
    spec = _v3_spec(config)
    if config.reference_speed_mps < APPROACH_REFERENCE_SPEED_MPS:
        raise ValueError("far pace reference cannot be lower than approach reference")
    spec["profile"] = PROFILE
    spec["status"] = "candidate_goal_approach_reference_pending_live_validation"
    spec["parent"] = "instant_pace_v3; replaces near-goal pace exemption with a slow approach reference"
    spec["retained_components"] = "v3 overspeed, far-goal pace and all other base components and terminal definitions"
    spec["parameters"]["approach_reference_speed_mps"] = APPROACH_REFERENCE_SPEED_MPS
    spec.pop("difference_from_v2", None)
    spec["difference_from_v3"] = {
        "changed_components": ["free_flow_pace_deficit"],
        "overspeed_math_unchanged": True,
        "far_goal_pace_math_unchanged": True,
        "purpose": "charge stationary unfinished goal approaches after observed v3 exemption failure; candidate, not a validated fix",
    }
    spec["pace"].update({
        "formula": "-cost_per_second * dt * clip((effective_ref - max(0, v_forward))/effective_ref, 0, 1)",
        "effective_reference": "approach_reference_speed_mps when distance_to_goal_m <= near_goal_distance_m; reference_speed_mps otherwise",
        "approach_reference_status": "0.5 m/s is a candidate slow approach preference",
        "near_goal_exemption": False,
        "near_goal_exempt_diagnostic": "always false; slow reference is not an exemption",
        "exemptions": ["startup_grace", "front_blocker", "complex_interaction", "terminal"],
        "near_goal_boundary": "distance_to_goal_m <= near_goal_distance_m selects lower reference; does not establish successful arrival",
        "success_handling": "successful goal-box arrival must set terminal_event and has zero pace cost",
    })
    spec["anti_hacking_invariants"].append("unfinished_near_goal_stopping_is_not_blanket_exempt")
    return spec
