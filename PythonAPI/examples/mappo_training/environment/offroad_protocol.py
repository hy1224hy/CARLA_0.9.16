"""CARLA-independent offroad classification for the formal Town05 task.

The simulator adapter is responsible for asking OpenDRIVE whether each sample
point belongs to a ``Driving`` lane.  This module freezes how those exact
point-membership results become an agent terminal event, so the thresholds can
be unit tested and written into checkpoints without starting CARLA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Sequence, Tuple


OFFROAD_PROTOCOL_VERSION = "driving_footprint_offroad_v1"
FOOTPRINT_INSET_FRACTION = 0.90
FOOTPRINT_BOUNDARY_SAMPLE_COUNT = 8
FOOTPRINT_OUTSIDE_COUNT_THRESHOLD = 3
OFFROAD_PERSISTENCE_TICKS = 3
OFFROAD_TERMINAL_PENALTY = -20.0


class OffroadProtocolError(ValueError):
    """Raised when an offroad input or frozen parameter is invalid."""


@dataclass(frozen=True)
class OffroadDecision:
    """One tick of audited footprint classification."""

    driving_center_valid: bool
    footprint_outside_count: int
    footprint_outside_fraction: float
    offroad_raw: bool
    offroad_consecutive_ticks: int
    offroad_confirmed: bool


def footprint_local_offsets(
    extent_x_m: object,
    extent_y_m: object,
    *,
    center_x_m: object = 0.0,
    center_y_m: object = 0.0,
) -> Tuple[Tuple[float, float], ...]:
    """Return the centre plus eight inset perimeter samples in vehicle space."""

    try:
        extent_x = float(extent_x_m)
        extent_y = float(extent_y_m)
        center_x = float(center_x_m)
        center_y = float(center_y_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise OffroadProtocolError("footprint geometry must be numeric") from error
    if not all(math.isfinite(value) for value in (extent_x, extent_y, center_x, center_y)):
        raise OffroadProtocolError("footprint geometry contains NaN/Inf")
    if extent_x <= 0.0 or extent_y <= 0.0:
        raise OffroadProtocolError("vehicle bounding-box extents must be positive")

    x = extent_x * FOOTPRINT_INSET_FRACTION
    y = extent_y * FOOTPRINT_INSET_FRACTION
    perimeter = (
        (-x, -y),
        (-x, 0.0),
        (-x, y),
        (0.0, -y),
        (0.0, y),
        (x, -y),
        (x, 0.0),
        (x, y),
    )
    return ((center_x, center_y),) + tuple(
        (center_x + dx, center_y + dy) for dx, dy in perimeter
    )


def classify_offroad_tick(
    *,
    driving_center_valid: object,
    boundary_driving_valid: Sequence[object],
    previous_consecutive_ticks: object,
) -> OffroadDecision:
    """Classify one tick and apply the frozen three-tick persistence rule."""

    if not isinstance(driving_center_valid, bool):
        raise OffroadProtocolError("driving_center_valid must be boolean")
    if len(boundary_driving_valid) != FOOTPRINT_BOUNDARY_SAMPLE_COUNT:
        raise OffroadProtocolError(
            "boundary_driving_valid must contain exactly eight samples"
        )
    if any(not isinstance(value, bool) for value in boundary_driving_valid):
        raise OffroadProtocolError("boundary membership values must be booleans")
    if isinstance(previous_consecutive_ticks, bool) or not isinstance(
        previous_consecutive_ticks, int
    ):
        raise OffroadProtocolError("previous_consecutive_ticks must be an integer")
    if previous_consecutive_ticks < 0:
        raise OffroadProtocolError("previous_consecutive_ticks cannot be negative")

    outside_count = sum(not value for value in boundary_driving_valid)
    outside_fraction = outside_count / FOOTPRINT_BOUNDARY_SAMPLE_COUNT
    raw = (
        not driving_center_valid
        or outside_count >= FOOTPRINT_OUTSIDE_COUNT_THRESHOLD
    )
    consecutive = previous_consecutive_ticks + 1 if raw else 0
    return OffroadDecision(
        driving_center_valid=driving_center_valid,
        footprint_outside_count=outside_count,
        footprint_outside_fraction=outside_fraction,
        offroad_raw=raw,
        offroad_consecutive_ticks=consecutive,
        offroad_confirmed=consecutive >= OFFROAD_PERSISTENCE_TICKS,
    )


def offroad_protocol_spec() -> Dict[str, object]:
    """Return the immutable machine-readable protocol stored with a run."""

    return {
        "protocol_version": OFFROAD_PROTOCOL_VERSION,
        "footprint_inset_fraction": FOOTPRINT_INSET_FRACTION,
        "sample_count": 1 + FOOTPRINT_BOUNDARY_SAMPLE_COUNT,
        "boundary_sample_count": FOOTPRINT_BOUNDARY_SAMPLE_COUNT,
        "outside_count_threshold": FOOTPRINT_OUTSIDE_COUNT_THRESHOLD,
        "persistence_ticks": OFFROAD_PERSISTENCE_TICKS,
        "terminal_penalty": OFFROAD_TERMINAL_PENALTY,
        "raw_rule": "center_not_driving_or_boundary_outside_count_gte_threshold",
        "membership_rule": "exact_OpenDRIVE_Driving_union_no_projection",
    }


def decision_as_dict(decision: OffroadDecision) -> Dict[str, object]:
    """Return a JSON-safe decision mapping."""

    return asdict(decision)
