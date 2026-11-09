"""Record-only stall diagnostic for P0-T05-09/R6.

This module never changes reward or termination.  It distinguishes persistent
no-progress behaviour from reasonable yielding behind a blocker or inside a
registered interaction-heavy scenario family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict


STALL_PROTOCOL_VERSION = "stall_diagnostic_v1_record_only"


@dataclass(frozen=True)
class StallConfig:
    low_speed_threshold_mps: float = 0.2
    progress_epsilon: float = 0.002
    initial_grace_ticks: int = 50
    no_progress_window_ticks: int = 100
    front_blocker_gap_m: float = 12.0


DEFAULT_STALL_CONFIG = StallConfig()


@dataclass(frozen=True)
class StallState:
    best_completion: float = 0.0
    last_progress_step: int = 0
    low_speed_ticks: int = 0


@dataclass(frozen=True)
class StallDecision:
    state: StallState
    low_speed: bool
    no_progress_ticks: int
    raw: bool
    front_blocker_exempt: bool
    complex_interaction_exempt: bool
    reportable: bool


def stall_protocol_spec(config: StallConfig = DEFAULT_STALL_CONFIG) -> Dict[str, object]:
    _validate_config(config)
    return {
        "stall_protocol_version": STALL_PROTOCOL_VERSION,
        "mode": "record_only",
        "parameters": asdict(config),
        "complex_interaction_families": [
            "required_merge",
            "stagger_rear_closing",
            "car_following_steady",
            "car_following_closing",
            "car_following_near_gap",
        ],
        "effects": {"reward": False, "termination": False, "curriculum": False},
    }


def is_complex_interaction_family(family: object) -> bool:
    value = str(family or "").strip()
    return value in set(stall_protocol_spec()["complex_interaction_families"])


def update_stall_diagnostic(
    state: StallState,
    *,
    current_step: int,
    speed_mps: float,
    route_completion: float,
    has_front_blocker: bool,
    complex_interaction: bool,
    config: StallConfig = DEFAULT_STALL_CONFIG,
) -> StallDecision:
    _validate_config(config)
    if not isinstance(state, StallState):
        raise ValueError("state must be StallState")
    if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < 0:
        raise ValueError("current_step must be a non-negative integer")
    speed = float(speed_mps)
    completion = float(route_completion)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    if not math.isfinite(completion) or not 0.0 <= completion <= 1.0:
        raise ValueError("route_completion must be in [0,1]")
    if not isinstance(has_front_blocker, bool) or not isinstance(complex_interaction, bool):
        raise ValueError("stall exemptions must be booleans")

    best = state.best_completion
    last_progress = state.last_progress_step
    if completion > best + config.progress_epsilon:
        best = completion
        last_progress = current_step
    low_speed = speed <= config.low_speed_threshold_mps
    low_speed_ticks = state.low_speed_ticks + 1 if low_speed else 0
    no_progress_ticks = max(0, current_step - last_progress)
    raw = bool(
        current_step >= config.initial_grace_ticks
        and low_speed_ticks >= config.no_progress_window_ticks
        and no_progress_ticks >= config.no_progress_window_ticks
    )
    front_exempt = bool(raw and has_front_blocker)
    complex_exempt = bool(raw and complex_interaction)
    return StallDecision(
        state=StallState(
            best_completion=best,
            last_progress_step=last_progress,
            low_speed_ticks=low_speed_ticks,
        ),
        low_speed=low_speed,
        no_progress_ticks=no_progress_ticks,
        raw=raw,
        front_blocker_exempt=front_exempt,
        complex_interaction_exempt=complex_exempt,
        reportable=bool(raw and not front_exempt and not complex_exempt),
    )


def _validate_config(config: StallConfig) -> None:
    if not isinstance(config, StallConfig):
        raise ValueError("config must be StallConfig")
    if not math.isfinite(config.low_speed_threshold_mps) or config.low_speed_threshold_mps < 0.0:
        raise ValueError("low-speed threshold must be finite and non-negative")
    if not math.isfinite(config.progress_epsilon) or config.progress_epsilon <= 0.0:
        raise ValueError("progress epsilon must be finite and positive")
    if config.initial_grace_ticks < 0 or config.no_progress_window_ticks <= 0:
        raise ValueError("stall tick thresholds are invalid")
    if not math.isfinite(config.front_blocker_gap_m) or config.front_blocker_gap_m <= 0.0:
        raise ValueError("front blocker gap must be finite and positive")


_validate_config(DEFAULT_STALL_CONFIG)
