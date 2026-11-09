"""Frozen R8-A curriculum identity and deterministic balanced sampler.

R8-A is a curriculum-only ablation.  This module deliberately contains no
CARLA, reward, model, or optimizer code so its support sets and sampling
invariants can be audited offline.
"""

from __future__ import annotations

from numbers import Integral
from typing import Dict, List, Sequence, Tuple


R8A_CURRICULUM_PROTOCOL_VERSION = "town05_r8a_curriculum_v1"
R8A_SAMPLING_PROTOCOL_VERSION = "town05_r8a_exact_balanced_v1"
R8A_MASTERY_PROTOCOL_VERSION = "town05_r8a_stratified_mastery_v1"

R8A_STAGE_NAMES: Tuple[str, ...] = ("S0-A", "S0-B", "S1", "S2")
R8A_STAGE_NEW_SCENARIOS: Tuple[Tuple[str, ...], ...] = (
    (
        "scenario_01_north_short_rolling_a0_left.json",
        "scenario_02_south_short_rolling_a1_left.json",
    ),
    (
        "scenario_03_north_short_stationary_a0_left.json",
        "scenario_04_south_short_stationary_a1_left.json",
    ),
    (
        "scenario_05_northbound_free_flow.json",
        "scenario_06_southbound_free_flow.json",
        "scenario_07_eastbound_free_flow.json",
        "scenario_08_westbound_free_flow.json",
    ),
    (
        "scenario_09_north_parallel_stationary_a0_left.json",
        "scenario_10_south_parallel_stationary_a1_left.json",
        "scenario_11_east_parallel_stationary_a1_left.json",
        "scenario_12_west_parallel_stationary_a0_left.json",
    ),
)


def r8a_cumulative_stages() -> List[List[str]]:
    """Return the exact cumulative support set registered for every stage."""
    stages: List[List[str]] = []
    cumulative: List[str] = []
    for additions in R8A_STAGE_NEW_SCENARIOS:
        cumulative.extend(additions)
        stages.append(list(cumulative))
    return stages


def validate_r8a_stages(stages: Sequence[Sequence[str]]) -> None:
    """Reject any scenario-pool or metadata drift from the frozen stage sets."""
    normalized = [list(stage) for stage in stages]
    expected = r8a_cumulative_stages()
    if normalized != expected:
        raise ValueError(
            "R8-A curriculum stages differ from the frozen support sets: "
            f"actual={normalized!r}, expected={expected!r}"
        )


def r8a_required_mastery_scenarios(stage_index: int) -> Tuple[str, ...]:
    """Return strata that must pass independently at the current stage.

    The two short mirrored stages are the causal intervention under test and
    therefore cannot be hidden by pooled evidence.  S1/S2 retain the frozen R7
    pooled mastery predicate after the short-task mechanism has been proven.
    """
    index = int(stage_index)
    if index < 0 or index >= len(R8A_STAGE_NAMES):
        raise ValueError(f"invalid R8-A stage index: {stage_index!r}")
    if index <= 1:
        return R8A_STAGE_NEW_SCENARIOS[index]
    return ()


def sample_r8a_balanced(
    available_scenarios: Sequence[str],
    n_episodes: int,
    *,
    phase: int,
) -> List[str]:
    """Build a deterministic, maximally balanced episode schedule.

    Every scenario receives ``floor(n/k)`` episodes.  The ``n % k`` remainder
    rotates by completed-update phase, which makes long-run exposure fair when
    the episode budget is not divisible by the cumulative stage size.  For the
    registered S0-A support (k=2, n=16), the result is exactly 8/8 on every
    update.  No success-rate weight can alter these counts.
    """
    if (
        isinstance(n_episodes, bool)
        or not isinstance(n_episodes, Integral)
        or n_episodes <= 0
    ):
        raise ValueError("n_episodes must be a positive integer")
    if isinstance(phase, bool) or not isinstance(phase, Integral) or phase < 0:
        raise ValueError("phase must be a non-negative integer")
    scenarios = [str(value) for value in available_scenarios]
    if not scenarios:
        raise ValueError("available_scenarios must not be empty")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("available_scenarios must be unique")

    scenarios = sorted(scenarios)
    episode_count = int(n_episodes)
    rotation = int(phase) % len(scenarios)
    rotated = scenarios[rotation:] + scenarios[:rotation]
    base_count, remainder = divmod(episode_count, len(scenarios))
    remaining: Dict[str, int] = {
        scenario: base_count + int(index < remainder)
        for index, scenario in enumerate(rotated)
    }

    selected: List[str] = []
    while len(selected) < episode_count:
        wrote = False
        for scenario in rotated:
            if remaining[scenario] <= 0:
                continue
            selected.append(scenario)
            remaining[scenario] -= 1
            wrote = True
        if not wrote:
            raise RuntimeError("R8-A sampler exhausted before filling the batch")
    return selected


def r8a_sampler_state(completed_update: int) -> Dict[str, object]:
    if (
        isinstance(completed_update, bool)
        or not isinstance(completed_update, Integral)
        or completed_update < 0
    ):
        raise ValueError("completed_update must be a non-negative integer")
    return {
        "protocol_version": R8A_SAMPLING_PROTOCOL_VERSION,
        "completed_update": int(completed_update),
    }


def validate_r8a_sampler_state(
    state: object, *, expected_completed_update: int
) -> None:
    expected = r8a_sampler_state(expected_completed_update)
    if state != expected:
        raise ValueError(
            f"R8-A sampler state mismatch: saved={state!r}, expected={expected!r}"
        )


def r8a_curriculum_protocol_spec() -> Dict[str, object]:
    return {
        "curriculum_protocol_version": R8A_CURRICULUM_PROTOCOL_VERSION,
        "sampling_protocol_version": R8A_SAMPLING_PROTOCOL_VERSION,
        "stage_order": list(R8A_STAGE_NAMES),
        "stage_new_scenarios": {
            stage: list(scenarios)
            for stage, scenarios in zip(R8A_STAGE_NAMES, R8A_STAGE_NEW_SCENARIOS)
        },
        "stage_support_sets": {
            stage: scenarios
            for stage, scenarios in zip(R8A_STAGE_NAMES, r8a_cumulative_stages())
        },
        "initial_batch_counts": {
            R8A_STAGE_NEW_SCENARIOS[0][0]: 8,
            R8A_STAGE_NEW_SCENARIOS[0][1]: 8,
        },
        "rollout_episodes": 16,
        "short_route_distance_m": 15.0,
        "short_episode_max_steps": 200,
        "short_rolling_initial_speed_mps": 2.0,
        "promotion_rule": "pooled_r7_predicate_and_each_short_mirror_stratum",
        "automatic_update_schedule": False,
    }
