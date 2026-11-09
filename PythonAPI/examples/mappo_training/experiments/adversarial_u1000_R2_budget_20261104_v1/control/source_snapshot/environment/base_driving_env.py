"""Explicit base-driving experiment adapter; preserves the parent 20/2 protocol."""
import math
from collections.abc import Mapping
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.base_driving_reward import (
    reward_spec, startup_efficiency, projected_half_extent,
)


class BaseDrivingEnv(CarlaMultiAgentEnv):
    def __init__(self, *args, base_driving_reward_profile="v6", **kwargs):
        self.base_driving_reward_profile = base_driving_reward_profile
        self.base_driving_reward_spec = reward_spec(base_driving_reward_profile)
        if kwargs.get("reward_version") != "v6" or kwargs.get("pace_profile") != "instant_pace_v6":
            raise ValueError("base driving requires the registered V6 parent")
        super().__init__(*args, **kwargs)

    def _compute_rewards_v6(self):
        rewards, details_by_agent = super()._compute_rewards_v6()
        if self.base_driving_reward_profile == "v6":
            return rewards, details_by_agent
        for aid, details in details_by_agent.items():
            diagnostic = self.last_reward_diagnostics[aid]
            decision = startup_efficiency(
                diagnostic["closing_speed_mps"],
                distance_to_goal_m=diagnostic["distance_to_goal_m"],
                elapsed_seconds=self.current_step * self.timestep,
                timestep=self.timestep,
                eligible=bool((diagnostic["pace_free_flow_tag"] and diagnostic["pace_non_interactive_tag"])
                              or self.current_scenario_family in {"free_flow", "parallel_stationary"}),
                front_blocker=bool(diagnostic["pace_front_blocker_exempt"]),
                complex_interaction=bool(diagnostic["pace_complex_interaction_exempt"]),
                terminal_event=bool(diagnostic["pace_terminal_exempt"]),
            )
            details["free_flow_pace_deficit"] = decision.reward
            diagnostic.update({"pace_grace_exempt": float(decision.grace_exempt),
                               "pace_near_goal_exempt": float(decision.near_goal_exempt),
                               "pace_terminal_exempt": float(decision.terminal_exempt),
                               "pace_deficit_raw": decision.raw_deficit,
                               "pace_deficit_risk": decision.risk,
                               "pace_deficit_eligible": float(decision.eligible),
                               "pace_deficit_reward": decision.reward})
            rewards[aid] = math.fsum(details.values())
        return rewards, details_by_agent

    def get_longitudinal_safety_state(self):
        """Actual bbox projected net gap and road-tangent closing speed, diagnostic only."""
        result = {}
        vehicles = list(self.vehicles.items() if isinstance(self.vehicles, Mapping) else enumerate(self.vehicles))
        for ego_id, ego in vehicles:
            if ego is None or not ego.is_alive:
                continue
            transform = ego.get_transform()
            reference = getattr(self, "road_reference_waypoints", {}).get(ego_id)
            yaw = math.radians((reference.transform if reference else transform).rotation.yaw)
            forward = (math.cos(yaw), math.sin(yaw))
            right = (-forward[1], forward[0])
            velocity = ego.get_velocity()
            ego_center = _world_bbox_center(transform, ego.bounding_box.location)
            candidates = []
            for other_id, other in vehicles:
                if other_id == ego_id or other is None or not other.is_alive:
                    continue
                other_transform = other.get_transform()
                # CARLA bounding-box centres can differ from actor origins.
                # Transform.transform may modify Location in place. Never hand
                # it the actor-owned bbox.location: repeated logging must not
                # change the geometry that the next measurement will observe.
                other_center = _world_bbox_center(other_transform, other.bounding_box.location)
                dx, dy = other_center.x - ego_center.x, other_center.y - ego_center.y
                centre_distance = dx * forward[0] + dy * forward[1]
                lateral = dx * right[0] + dy * right[1]
                if centre_distance <= 0 or abs(lateral) >= 2.0:
                    continue
                ego_half = projected_half_extent(ego.bounding_box.extent.x, ego.bounding_box.extent.y,
                                                math.radians(transform.rotation.yaw + ego.bounding_box.rotation.yaw) - yaw)
                other_half = projected_half_extent(other.bounding_box.extent.x, other.bounding_box.extent.y,
                                                  math.radians(other_transform.rotation.yaw + other.bounding_box.rotation.yaw) - yaw)
                gap = centre_distance - ego_half - other_half
                other_velocity = other.get_velocity()
                closing = ((velocity.x - other_velocity.x) * forward[0]
                           + (velocity.y - other_velocity.y) * forward[1])
                candidates.append((centre_distance, other_id, gap, closing))
            if not candidates:
                result[ego_id] = {"has_front_vehicle": 0.0, "front_agent_id": -1.0,
                                  "front_center_distance_m": None, "front_gap_m": None,
                                  "relative_speed_mps": 0.0, "ttc_s": None}
                continue
            centre, other_id, gap, closing = min(candidates)
            result[ego_id] = {"has_front_vehicle": 1.0, "front_agent_id": float(other_id),
                              "front_center_distance_m": centre, "front_gap_m": gap,
                              "relative_speed_mps": closing,
                              "ttc_s": max(0.0, gap) / closing if closing > 0 else None}
        return result


def _world_bbox_center(transform, location):
    """Return a world point without modifying CARLA's local bbox Location."""
    point = type(location)(x=float(location.x), y=float(location.y), z=float(location.z))
    transformed = transform.transform(point)
    return point if transformed is None else transformed
