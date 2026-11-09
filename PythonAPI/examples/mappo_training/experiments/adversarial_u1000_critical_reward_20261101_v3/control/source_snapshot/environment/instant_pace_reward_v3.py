"""Candidate v3 speed cost; all v2 pace mathematics remain unchanged.

The two cost rates below are frozen candidate coefficients.  The shared v2
config supplies pace parameters and speed thresholds; its old quadratic
``overspeed_cost_per_second`` coefficient is not used by v3.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from environment.instant_pace_reward import (
    CONFIG, InstantPaceConfig, instant_pace_deficit,
    continuous_overspeed_penalty as _v2_overspeed_penalty,
    instant_pace_spec as _v2_spec,
)


PROFILE = "instant_pace_v3"
LINEAR_COST_PER_SECOND = 0.5
QUADRATIC_COST_PER_SECOND = 0.5


def continuous_overspeed_penalty(
    speed_mps: object,
    *,
    timestep: object,
    config: InstantPaceConfig = CONFIG,
) -> Tuple[float, float]:
    """Return linear-plus-quadratic cost and the unchanged squared risk.

    Risk remains the square of normalized excess, without clipping.  Its
    square root is the nonnegative linear excess.  The linear cost provides
    a nonzero right-hand slope immediately above the preferred upper speed.
    Input validation and numerical-overflow checks are inherited from v2.
    Lifecycle eligibility remains the environment's responsibility.
    """
    _old_reward, risk = _v2_overspeed_penalty(
        speed_mps, timestep=timestep, config=config,
    )
    normalized_excess = math.sqrt(risk)
    reward = -float(timestep) * (
        LINEAR_COST_PER_SECOND * normalized_excess
        + QUADRATIC_COST_PER_SECOND * risk
    )
    if not math.isfinite(reward):
        raise ValueError("v3 continuous overspeed reward overflow")
    return float(reward), risk


def instant_pace_spec(config: InstantPaceConfig = CONFIG) -> Dict[str, object]:
    """Describe both deviations from frozen v6 and the sole v2 change."""
    spec = _v2_spec(config)
    spec["profile"] = PROFILE
    spec["status"] = "candidate_stronger_speed_cost_pending_live_validation"
    spec["parent"] = "instant_pace_v2; replaces only its overspeed mathematics"
    spec["retained_components"] = "v2 instantaneous pace, all other base components and terminal definitions"
    spec["parameters"].pop("overspeed_cost_per_second")
    spec["parameters"].update({
        "overspeed_linear_cost_per_second": LINEAR_COST_PER_SECOND,
        "overspeed_quadratic_cost_per_second": QUADRATIC_COST_PER_SECOND,
    })
    spec["difference_from_v2"] = {
        "changed_components": ["overspeed"],
        "pace_math_and_exemptions_unchanged": True,
        "purpose": "stronger candidate cost after unsuccessful v2 pilot; not a validated behavior fix",
    }
    spec["overspeed"] = {
        "formula": "-dt * (linear_cost_per_second * normalized_excess + quadratic_cost_per_second * normalized_excess^2)",
        "normalized_excess": "max(0, speed - upper) / scale",
        "risk_diagnostic": "normalized_excess^2; unchanged from v2",
        "risk_range": "[0, infinity); no finite saturation threshold",
        "threshold_slope": "nonzero right-hand reward slope: -dt * linear_cost_per_second / scale",
        "threshold_meaning": "task speed preference; not a claim about legal road speed limit",
    }
    return spec
