"""CARLA-independent road-legality state machine for P0-T05-10/R7.

CARLA's exact ``Driving`` membership answers whether a vehicle footprint is on
some drivable lane.  It deliberately does not answer whether that lane belongs
to the intended direction of travel.  This module freezes the second question
without adding any feature to the policy or critic observation spaces.

The CARLA adapter is responsible for resolving waypoint topology and passing
the resulting booleans/cosines into :func:`update_road_legality`.  Keeping the
state machine independent of CARLA makes every transition deterministic and
unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Tuple


ROAD_LEGALITY_PROTOCOL_VERSION = "town05_road_legality_r7_v1"
LaneKey = Tuple[int, int]


@dataclass(frozen=True)
class RoadLegalityConfig:
    # A candidate lane must point within 60 degrees of the reference lane.
    same_direction_min_cosine: float = 0.5
    # A lane pointing more than 120 degrees away is explicitly opposing.
    opposing_direction_max_cosine: float = -0.5
    # Vehicle heading more than 100 degrees from its current lane is wrong-way.
    wrong_way_heading_max_cosine: float = -0.1736481776669303
    wrong_way_min_speed_mps: float = 0.5
    violation_persistence_ticks: int = 5
    legal_transition_grace_ticks: int = 10


DEFAULT_ROAD_LEGALITY_CONFIG = RoadLegalityConfig()


@dataclass(frozen=True)
class RoadLegalityState:
    reference_lane: Optional[LaneKey] = None
    violation_reason: str = ""
    consecutive_violation_ticks: int = 0
    legal_transition_grace_remaining: int = 0


@dataclass(frozen=True)
class RoadLegalityObservation:
    current_lane: Optional[LaneKey]
    exact_driving: bool
    same_reference_lane: bool
    adjacent_to_reference: bool
    lane_change_permitted: bool
    current_to_reference_cosine: float
    vehicle_to_current_cosine: float
    speed_mps: float
    in_junction: bool
    broken_marking_event: bool


@dataclass(frozen=True)
class RoadLegalityDecision:
    state: RoadLegalityState
    classification: str
    legal_lane_change: bool
    lane_deviation_exempted: bool
    violation_raw: bool
    violation_confirmed: bool
    success_eligible: bool
    update_reference_lane: bool


class RoadLegalityError(ValueError):
    """Raised when road-legality evidence violates the frozen protocol."""


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RoadLegalityError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise RoadLegalityError(f"{name} contains NaN/Inf")
    return result


def _lane_key(value: Optional[LaneKey], *, name: str) -> Optional[LaneKey]:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise RoadLegalityError(f"{name} must be a (road_id, lane_id) tuple")
    if any(isinstance(item, bool) for item in value):
        raise RoadLegalityError(f"{name} cannot contain booleans")
    return int(value[0]), int(value[1])


def validate_road_legality_config(config: RoadLegalityConfig) -> None:
    if not isinstance(config, RoadLegalityConfig):
        raise RoadLegalityError("config must be RoadLegalityConfig")
    for name in (
        "same_direction_min_cosine",
        "opposing_direction_max_cosine",
        "wrong_way_heading_max_cosine",
    ):
        value = _finite(getattr(config, name), name=name)
        if not -1.0 <= value <= 1.0:
            raise RoadLegalityError(f"{name} must be in [-1,1]")
    if config.opposing_direction_max_cosine >= config.same_direction_min_cosine:
        raise RoadLegalityError("opposing/same-direction thresholds overlap")
    if _finite(config.wrong_way_min_speed_mps, name="wrong_way_min_speed_mps") < 0.0:
        raise RoadLegalityError("wrong_way_min_speed_mps cannot be negative")
    for name in ("violation_persistence_ticks", "legal_transition_grace_ticks"):
        value = getattr(config, name)
        if isinstance(value, bool) or int(value) < 1:
            raise RoadLegalityError(f"{name} must be a positive integer")


validate_road_legality_config(DEFAULT_ROAD_LEGALITY_CONFIG)


def road_legality_protocol_spec(
    config: RoadLegalityConfig = DEFAULT_ROAD_LEGALITY_CONFIG,
) -> Dict[str, object]:
    validate_road_legality_config(config)
    return {
        "road_legality_protocol_version": ROAD_LEGALITY_PROTOCOL_VERSION,
        "parameters": asdict(config),
        "observation_space_change": False,
        "reference_lane_key": ["road_id", "lane_id"],
        "legal_lane_change_evidence": [
            "adjacent_waypoint_topology",
            "carla_lane_change_permission",
            "same_direction_alignment",
        ],
        "broken_marking_semantics": "audit_only_never_sufficient_or_required",
        "junction_semantics": "nonterminal_ambiguous_but_success_ineligible",
        "terminal_rule": "same_reason_persists_for_frozen_tick_threshold",
        "offroad_delegation": "exact_driving_footprint_protocol",
    }


def update_road_legality(
    state: RoadLegalityState,
    observation: RoadLegalityObservation,
    *,
    config: RoadLegalityConfig = DEFAULT_ROAD_LEGALITY_CONFIG,
) -> RoadLegalityDecision:
    """Advance one legality tick using topology already resolved by CARLA.

    Junction ambiguity never creates a road-rule terminal because projection
    can legitimately jump between connector waypoints.  It still blocks a goal
    success, preventing an ambiguous or illegal pose from being rewarded.
    Exact non-Driving occupancy remains exclusively owned by the offroad FSM.
    """

    validate_road_legality_config(config)
    if not isinstance(state, RoadLegalityState):
        raise RoadLegalityError("state must be RoadLegalityState")
    if not isinstance(observation, RoadLegalityObservation):
        raise RoadLegalityError("observation must be RoadLegalityObservation")
    reference = _lane_key(state.reference_lane, name="state.reference_lane")
    current = _lane_key(observation.current_lane, name="observation.current_lane")
    lane_cos = _finite(
        observation.current_to_reference_cosine,
        name="current_to_reference_cosine",
    )
    heading_cos = _finite(
        observation.vehicle_to_current_cosine,
        name="vehicle_to_current_cosine",
    )
    speed = _finite(observation.speed_mps, name="speed_mps")
    if not -1.0 <= lane_cos <= 1.0 or not -1.0 <= heading_cos <= 1.0:
        raise RoadLegalityError("direction cosines must be in [-1,1]")
    if speed < 0.0:
        raise RoadLegalityError("speed_mps cannot be negative")

    grace = max(0, int(state.legal_transition_grace_remaining) - 1)
    legal_change = False
    update_reference = False

    if not observation.exact_driving or current is None:
        classification = "offroad_delegated"
        violation_reason = ""
        success_eligible = False
    elif reference is None:
        classification = "reference_initialized"
        violation_reason = ""
        success_eligible = not observation.in_junction
        update_reference = True
    elif observation.in_junction:
        classification = "junction_ambiguous"
        violation_reason = ""
        success_eligible = False
    elif (
        speed >= config.wrong_way_min_speed_mps
        and heading_cos <= config.wrong_way_heading_max_cosine
    ):
        classification = "wrong_way_heading"
        violation_reason = classification
        success_eligible = False
    elif heading_cos <= config.wrong_way_heading_max_cosine:
        # A nearly stationary vehicle may be yielding or completing a tight
        # manoeuvre, so do not terminate it.  Its pose is nevertheless not a
        # legal goal pose and must never pass the success gate.
        classification = "wrong_way_heading_stationary"
        violation_reason = ""
        success_eligible = False
    elif (
        not observation.same_reference_lane
        and lane_cos <= config.opposing_direction_max_cosine
    ):
        classification = "opposing_lane"
        violation_reason = classification
        success_eligible = False
    elif observation.same_reference_lane:
        classification = "reference_lane"
        violation_reason = ""
        success_eligible = True
        update_reference = True
    elif (
        observation.adjacent_to_reference
        and observation.lane_change_permitted
        and lane_cos >= config.same_direction_min_cosine
    ):
        classification = "legal_lane_change"
        violation_reason = ""
        success_eligible = True
        legal_change = True
        update_reference = True
        grace = config.legal_transition_grace_ticks
    else:
        classification = "illegal_lane_change"
        violation_reason = classification
        success_eligible = False

    violation_raw = bool(violation_reason)
    if violation_raw and violation_reason == state.violation_reason:
        ticks = int(state.consecutive_violation_ticks) + 1
    elif violation_raw:
        ticks = 1
    else:
        ticks = 0
    confirmed = bool(ticks >= config.violation_persistence_ticks)
    next_reference = current if update_reference else reference
    next_state = RoadLegalityState(
        reference_lane=next_reference,
        violation_reason=violation_reason,
        consecutive_violation_ticks=ticks,
        legal_transition_grace_remaining=grace,
    )
    return RoadLegalityDecision(
        state=next_state,
        classification=classification,
        legal_lane_change=legal_change,
        lane_deviation_exempted=bool(legal_change or grace > 0),
        violation_raw=violation_raw,
        violation_confirmed=confirmed,
        success_eligible=bool(success_eligible and not violation_raw),
        update_reference_lane=update_reference,
    )


def decision_as_dict(decision: RoadLegalityDecision) -> Dict[str, object]:
    if not isinstance(decision, RoadLegalityDecision):
        raise RoadLegalityError("decision must be RoadLegalityDecision")
    return {
        "classification": decision.classification,
        "reference_lane": (
            None if decision.state.reference_lane is None
            else list(decision.state.reference_lane)
        ),
        "legal_lane_change": decision.legal_lane_change,
        "lane_deviation_exempted": decision.lane_deviation_exempted,
        "violation_reason": decision.state.violation_reason,
        "violation_raw": decision.violation_raw,
        "consecutive_violation_ticks": decision.state.consecutive_violation_ticks,
        "violation_confirmed": decision.violation_confirmed,
        "legal_transition_grace_remaining": (
            decision.state.legal_transition_grace_remaining
        ),
        "success_eligible": decision.success_eligible,
        "update_reference_lane": decision.update_reference_lane,
    }
