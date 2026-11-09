"""Free-flow pace-deficit shaping for the R8-C1 reward ablation.

Reward v6 preserves every reward-v5 contribution and adds one bounded dense
cost.  The cost is active only for explicitly tagged, non-interactive
free-flow tasks.  It compares normalized route remaining with normalized time
remaining, so an on-schedule trajectory is neutral while an increasingly late
trajectory receives immediate negative feedback.

This is deliberately *not* claimed to be policy-invariant potential-based
reward shaping.  It changes the finite-horizon objective by assigning a small
cost to states from which the route is falling behind its time budget.  The
separate reward version and diagnostics make that experimental intervention
auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, NamedTuple

from environment.base_reward_v5 import (
    REWARD_COMPONENTS as REWARD_V5_COMPONENTS,
    REWARD_DIAGNOSTICS as REWARD_V5_DIAGNOSTICS,
    RewardV5Config,
    validate_reward_v5_config,
)


REWARD_VERSION = "v6"
REWARD_PROTOCOL_VERSION = "base_route_free_flow_pace_reward_v6_r8c1"

PACE_COMPONENT = "free_flow_pace_deficit"
REWARD_COMPONENTS = REWARD_V5_COMPONENTS + (PACE_COMPONENT,)
REWARD_DIAGNOSTICS = REWARD_V5_DIAGNOSTICS + (
    "pace_free_flow_tag",
    "pace_non_interactive_tag",
    "pace_grace_exempt",
    "pace_front_blocker_exempt",
    "pace_complex_interaction_exempt",
    "pace_near_goal_exempt",
    "pace_terminal_exempt",
    "pace_deficit_raw",
    "pace_deficit_risk",
    "pace_deficit_eligible",
    "pace_deficit_reward",
)


@dataclass(frozen=True)
class RewardV6Config(RewardV5Config):
    pace_deficit_deadzone: float = 0.05
    pace_deficit_weight: float = 0.01
    pace_grace_ticks: int = 20
    pace_near_goal_remaining_fraction: float = 0.10


DEFAULT_REWARD_V6_CONFIG = RewardV6Config()


class PaceDeficitDecision(NamedTuple):
    reward: float
    raw_deficit: float
    risk: float
    eligible: bool
    grace_exempt: bool
    near_goal_exempt: bool
    terminal_exempt: bool


def validate_reward_v6_config(config: RewardV6Config) -> None:
    if not isinstance(config, RewardV6Config):
        raise ValueError("reward v6 config must be RewardV6Config")
    validate_reward_v5_config(config)
    if not 0.0 <= config.pace_deficit_deadzone < 1.0:
        raise ValueError("pace-deficit deadzone must be in [0, 1)")
    if not math.isfinite(config.pace_deficit_weight) or config.pace_deficit_weight <= 0.0:
        raise ValueError("pace-deficit weight must be finite and positive")
    if (
        isinstance(config.pace_grace_ticks, bool)
        or not isinstance(config.pace_grace_ticks, int)
        or config.pace_grace_ticks < 0
    ):
        raise ValueError("pace grace ticks must be a non-negative integer")
    if not 0.0 <= config.pace_near_goal_remaining_fraction <= 1.0:
        raise ValueError("near-goal remaining fraction must be in [0, 1]")


validate_reward_v6_config(DEFAULT_REWARD_V6_CONFIG)


def _unit_fraction(value: object, *, field: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return fraction


def free_flow_pace_deficit(
    remaining_route_fraction: object,
    remaining_time_fraction: object,
    *,
    current_step: int,
    free_flow: bool,
    non_interactive: bool,
    front_blocker: bool = False,
    complex_interaction: bool = False,
    terminal_event: bool = False,
    config: RewardV6Config = DEFAULT_REWARD_V6_CONFIG,
) -> PaceDeficitDecision:
    """Return the bounded R8-C1 pace cost and its audit evidence.

    ``raw_deficit`` and ``risk`` describe lateness independently of eligibility.
    This lets the logs distinguish an on-time state from a late state that was
    correctly exempted because of traffic or a terminal event.
    """

    validate_reward_v6_config(config)
    route_remaining = _unit_fraction(
        remaining_route_fraction, field="remaining_route_fraction"
    )
    time_remaining = _unit_fraction(
        remaining_time_fraction, field="remaining_time_fraction"
    )
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise ValueError("current_step must be an integer")
    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    flags = {
        "free_flow": free_flow,
        "non_interactive": non_interactive,
        "front_blocker": front_blocker,
        "complex_interaction": complex_interaction,
        "terminal_event": terminal_event,
    }
    if not all(isinstance(value, bool) for value in flags.values()):
        raise ValueError("pace-deficit eligibility flags must be booleans")

    raw_deficit = max(
        0.0,
        route_remaining - time_remaining - config.pace_deficit_deadzone,
    )
    risk = min(
        1.0,
        raw_deficit / (1.0 - config.pace_deficit_deadzone),
    )
    grace_exempt = current_step < config.pace_grace_ticks
    near_goal_exempt = (
        route_remaining <= config.pace_near_goal_remaining_fraction
    )
    terminal_exempt = terminal_event
    eligible = bool(
        free_flow
        and non_interactive
        and not grace_exempt
        and not front_blocker
        and not complex_interaction
        and not near_goal_exempt
        and not terminal_exempt
    )
    reward = -config.pace_deficit_weight * risk if eligible else 0.0
    return PaceDeficitDecision(
        reward=float(reward),
        raw_deficit=float(raw_deficit),
        risk=float(risk),
        eligible=eligible,
        grace_exempt=grace_exempt,
        near_goal_exempt=near_goal_exempt,
        terminal_exempt=terminal_exempt,
    )


def reward_v6_spec(
    config: RewardV6Config = DEFAULT_REWARD_V6_CONFIG,
) -> Dict[str, object]:
    validate_reward_v6_config(config)
    return {
        "reward_version": REWARD_VERSION,
        "reward_protocol_version": REWARD_PROTOCOL_VERSION,
        "parent_reward_protocol": "base_route_road_rule_reward_v5_r7",
        "components": list(REWARD_COMPONENTS),
        "diagnostics": list(REWARD_DIAGNOSTICS),
        "parameters": asdict(config),
        "positive_dense_components": ["progress"],
        "pace_deficit": {
            "formula": "max(0, remaining_route_fraction - remaining_time_fraction - deadzone)",
            "normalization": "linear_by_one_minus_deadzone_clipped_0_1",
            "eligibility": "free_flow_and_non_interactive_only",
            "low_speed_hard_gate": False,
            "front_blocker_exemption": True,
            "complex_interaction_exemption": True,
            "policy_invariance_guarantee": False,
        },
        "terminal_priority": [
            "collision",
            "offroad",
            "road_rule",
            "success",
            "mission_timeout",
        ],
        "anti_hacking_invariants": [
            "all_reward_v5_terms_are_unchanged",
            "speed_alone_has_no_positive_reward",
            "pace_cost_is_bounded",
            "on_schedule_or_ahead_has_zero_pace_cost",
            "front_blocker_and_complex_interactions_are_exempt",
            "no_hard_low_speed_threshold_to_exploit_by_creeping",
            "terminal_steps_have_zero_pace_cost",
        ],
    }
