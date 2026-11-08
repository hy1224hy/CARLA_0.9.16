"""Versioned own-car efficiency intervention; pure and CARLA independent."""
from dataclasses import asdict
from environment.instant_pace_reward import InstantPaceConfig
from environment.instant_pace_reward_v4 import instant_pace_deficit

PROFILES = ("v6", "startup_v1")
STARTUP_CONFIG = InstantPaceConfig(cost_per_second=1.5, grace_seconds=0.8)


def reward_spec(profile):
    if profile not in PROFILES:
        raise ValueError("unregistered base driving reward profile")
    return {"protocol": "base_driving_reward_v1", "profile": profile,
            "parent": "instant_pace_v6", "status": "IMPLEMENTED_PENDING_LIVE_VALIDATION",
            "changed_components": [] if profile == "v6" else ["free_flow_pace_deficit"],
            "startup_parameters": None if profile == "v6" else asdict(STARTUP_CONFIG),
            "terminal_semantics": "inherited V6 own goal-zone entry then despawn; no simultaneous-arrival objective",
            "history_debt": False, "observation_action_change": False}


def startup_efficiency(forward_speed_mps, *, distance_to_goal_m, elapsed_seconds,
                       timestep, eligible, front_blocker, complex_interaction,
                       terminal_event):
    return instant_pace_deficit(
        forward_speed_mps, distance_to_goal_m=distance_to_goal_m,
        elapsed_seconds=elapsed_seconds, timestep=timestep, eligible=eligible,
        front_blocker=front_blocker, complex_interaction=complex_interaction,
        terminal_event=terminal_event, config=STARTUP_CONFIG,
    )


def projected_half_extent(half_length, half_width, relative_yaw_radians):
    import math
    values = (float(half_length), float(half_width), float(relative_yaw_radians))
    if not all(math.isfinite(x) for x in values) or min(values[:2]) < 0:
        raise ValueError("invalid projected vehicle extent")
    return abs(values[0] * math.cos(values[2])) + abs(values[1] * math.sin(values[2]))
