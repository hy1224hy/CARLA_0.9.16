"""Optional v6 route_pace_v1 intervention; frozen R8-C1 defaults stay intact."""
from dataclasses import asdict, dataclass
import math

from environment.base_reward_v6 import PaceDeficitDecision

PROFILE = "route_pace_v1"


@dataclass(frozen=True)
class RoutePaceConfig:
    reference_speed_mps: float = 2.5
    grace_seconds: float = 2.0
    lag_deadzone_m: float = 2.0
    lag_full_cost_m: float = 5.0
    cost_per_second: float = 0.5


CONFIG = RoutePaceConfig()


def route_pace_spec():
    return {
        "profile": PROFILE,
        "parameters": asdict(CONFIG),
        "formula": "-cost_per_second * dt * clip((min(L, v*t_free) - L*(1-r) - deadzone_m)/lag_full_cost_m, 0, 1)",
        "clock": "per-agent unblocked eligible time after initial grace; reset each episode",
        "eligibility": "free_flow+non_interactive tags OR free_flow/parallel_stationary family; complex interactions excluded",
        "parent": "v6; replaces only free_flow_pace_deficit",
        "policy_invariance_guarantee": False,
    }


def route_pace_deficit(remaining_route_fraction, *, initial_distance_m,
                       free_seconds, timestep, grace_exempt=False,
                       eligible=True, terminal_event=False):
    """Charge only own-route lag; there is no teammate progress/arrival term.

    raw_deficit is metres beyond the deadzone (unlike frozen v6's fraction).
    At 0.1s the maximum cost is 0.05 per transition.
    """
    values = (remaining_route_fraction, initial_distance_m, free_seconds, timestep)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("route pace inputs must be finite")
    if not 0 <= remaining_route_fraction <= 1 or initial_distance_m <= 0:
        raise ValueError("invalid route pace distance/fraction")
    if free_seconds < 0 or timestep <= 0:
        raise ValueError("invalid route pace time")
    expected_m = min(initial_distance_m, CONFIG.reference_speed_mps * free_seconds)
    travelled_m = initial_distance_m * (1 - remaining_route_fraction)
    lag_m = max(0.0, expected_m - travelled_m - CONFIG.lag_deadzone_m)
    risk = min(1.0, lag_m / CONFIG.lag_full_cost_m)
    active = bool(eligible and not grace_exempt and not terminal_event)
    return PaceDeficitDecision(
        -CONFIG.cost_per_second * timestep * risk if active else 0.0,
        lag_m, risk, active, bool(grace_exempt), False, bool(terminal_event),
    )
