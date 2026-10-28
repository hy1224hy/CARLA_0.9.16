"""Thin, explicitly versioned four-car Mut-MAPPO adapter.

Owns one synchronous CARLA world. It never starts CARLA and callers must hold
the shared port-2000 ownership lease. Four-car geometry is candidate until a
live reset verifies the recorded map/lane/marking/spawn evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import carla
import numpy as np

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.base_reward_v4 import route_completion
from environment.mut_mappo_events import (
    BLUE_BASE_COMPONENTS, BLUE_BASE_RISK_COMPONENTS,
    MutEpisodeEvents, MutRewardConfig, SAFETY_KEYS, blue_driving_reward,
)
from utils.adversarial_metrics import extract_kinematics
from environment.mut_mappo_efficiency_reward import PROFILE as TEAM_PROFILE, TeamEfficiencyReward


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class MutMAPPOEnv(CarlaMultiAgentEnv):
    """20-dimensional actors, three-dimensional environment controls, 2 or 4 cars."""

    def __init__(self, red_checkpoint_config: dict, task_config: dict,
                 host="127.0.0.1", port=2000, seed=42, num_agents=4,
                 audit_callback=None, reference_store=None):
        if num_agents not in (2, 4):
            raise ValueError("MutMAPPO supports two-car C0 or four-car C1/C2")
        self.task_config = deepcopy(task_config)
        self.reference_store = reference_store
        self.team_reward = (TeamEfficiencyReward(task_config.get("team_reward"),
                            dt=float(task_config.get("timestep", 0.1)),
                            horizon=int(task_config.get("max_steps", 500)),
                            gamma=float(task_config.get("team_reward", {}).get("gamma", 0.999)))
                            if task_config.get("reward_profile") == TEAM_PROFILE else None)
        self._route_progress = {}
        self._route_axes = {}
        self._team_reference = None
        self.red_checkpoint_config = deepcopy(red_checkpoint_config)
        self.red_ids = (0, 1)
        self.blue_ids = (2, 3) if num_agents == 4 else ()
        self._case = None
        self._case_geometry_evidence = {}
        self._last_mut_reward_details = {}
        self._last_mut_step_events = {}
        self._mut_valid = True
        self._mut_invalid_reason = None
        self._task_violations = {bid: False for bid in self.blue_ids}
        self._stall_seconds = {bid: 0.0 for bid in self.blue_ids}
        self._task_stall_evidence = {}
        self._mut_outcomes = {aid: "running" for aid in range(num_agents)}
        self._agent_result_evidence = {}
        timestep = float(task_config.get("timestep", 0.1))
        self.mut_reward_config = MutRewardConfig.from_dict(task_config.get("reward"), timestep=timestep)
        self._cost_coefficient = self.mut_reward_config.safety_lambda
        self.mut_events = MutEpisodeEvents(self.mut_reward_config, self.red_ids, self.blue_ids)
        self._cameras_enabled = bool(task_config.get("cameras_enabled", False))
        self._physical_contacts_seen = {aid: set() for aid in range(num_agents)}
        reward_version = str(red_checkpoint_config.get("reward_version", "v6"))
        if reward_version not in {"v5", "v6"}:
            raise ValueError("first Mut task requires a frozen 20-dimensional v5/v6 road-legality contract")
        super().__init__(host=host, port=port, town=task_config.get("town", "Town05"),
                         num_agents=num_agents, timestep=timestep,
                         max_steps=int(task_config.get("max_steps", 500)),
                         spawn_vehicles=False, adversarial_agent_ids=list(self.blue_ids),
                         reward_version=reward_version, disable_tensorboard=True, seed=seed,
                         reward_gamma=float(red_checkpoint_config.get("reward_gamma", 0.999)),
                         adversary_collision_ends_episode=False,
                         follow_spectator=bool(task_config.get("follow_spectator", True)),
                         lidar_range_m=float(red_checkpoint_config.get("lidar_range_m", 4.0)),
                         ttc_max_range_m=float(red_checkpoint_config.get("ttc_max_range_m", 10.0)),
                         pace_profile=red_checkpoint_config.get("pace_profile", "frozen_v6"),
                         role_reward_mode="mut_mappo", tick_audit_callback=audit_callback)

    def _attach_camera(self, agent_id, vehicle):
        if self._cameras_enabled:
            return super()._attach_camera(agent_id, vehicle)

    def set_cost_coefficient(self, value):
        """Set the per-team, checkpointed dual coefficient before its rollout."""
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("cost coefficient must be finite and nonnegative")
        self._cost_coefficient = value

    @staticmethod
    def _waypoint_record(wp):
        return {"road_id": int(wp.road_id), "section_id": int(wp.section_id),
                "lane_id": int(wp.lane_id), "s": float(wp.s),
                "lane_change": str(wp.lane_change),
                "left_marking": str(wp.left_lane_marking.type),
                "right_marking": str(wp.right_lane_marking.type),
                "is_junction": bool(wp.is_junction)}

    def _resolve_case(self, case):
        if not case.get("case_key") or case.get("family") not in {"blocking", "cut_in", "parallel", "staggered"}:
            raise ValueError("case requires stable case_key and known family")
        base_ref = case.get("base_scenario")
        if isinstance(base_ref, dict):
            scenario = deepcopy(base_ref)
        else:
            base_path = Path(str(base_ref))
            if not base_path.is_absolute():
                base_path = PACKAGE_ROOT / base_path
            with base_path.open(encoding="utf-8") as handle:
                scenario = json.load(handle)
        scenario["episode_max_steps"] = int(case.get("episode_max_steps", self.default_max_steps))
        agents = scenario["agents"]
        if set(agents) != {"agent0", "agent1"}:
            raise ValueError("base scenario must contain exactly the two frozen red initial states")
        evidence = {"case_key": case["case_key"], "status": "map_geometry_checked_spawn_pending",
                    "base_scenario": str(base_ref), "waypoints": {}}
        for rid in self.red_ids:
            key = f"agent{rid}"
            agents[key].update(case.get("red_overrides", {}).get(key, {}))
            item = agents[key]
            loc = carla.Location(x=float(item["x"]), y=float(item["y"]), z=float(item.get("z", 0.5)))
            wp = self._get_exact_driving_waypoint(loc)
            if wp is None or wp.is_junction:
                raise ValueError(f"{case['case_key']}: red {rid} lacks a non-junction Driving waypoint")
            evidence["waypoints"][key] = self._waypoint_record(wp)
        if self.blue_ids:
            offsets = case.get("blue_spawn_offsets", {})
            if set(offsets) != {"agent2", "agent3"}:
                raise ValueError("four-car case requires two explicit blue waypoint offsets")
            for bid in self.blue_ids:
                spec = offsets[f"agent{bid}"]
                rid = int(spec["anchor_red_id"])
                if rid not in self.red_ids:
                    raise ValueError("blue anchor must be physical red ID 0 or 1")
                start = agents[f"agent{rid}"]
                wp = self._get_exact_driving_waypoint(carla.Location(
                    x=float(start["x"]), y=float(start["y"]), z=float(start.get("z", 0.5))))
                distance = float(spec["forward_m"])
                options = wp.next(distance) if distance > 0 else wp.previous(-distance)
                legal = [candidate for candidate in options
                         if candidate.road_id == wp.road_id and candidate.lane_id == wp.lane_id
                         and not candidate.is_junction]
                if len(legal) != 1:
                    raise ValueError(f"{case['case_key']}: blue {bid} offset is not unique on the same legal lane")
                blue_wp = legal[0]
                yaw = float(blue_wp.transform.rotation.yaw)
                delta_yaw = (yaw - float(start["yaw"]) + 180.0) % 360.0 - 180.0
                if abs(delta_yaw) > 5.0:
                    raise ValueError("blue/red route directions differ")
                if case["family"] == "cut_in":
                    target = agents[f"agent{1 - rid}"]
                    target_wp = self._get_exact_driving_waypoint(carla.Location(
                        x=float(target["x"]), y=float(target["y"]), z=float(target.get("z", 0.5))))
                    adjacent, permitted = self._road_adjacent_evidence(blue_wp, target_wp)
                    markings = (str(blue_wp.left_lane_marking.type), str(blue_wp.right_lane_marking.type))
                    if (not adjacent or not permitted or not any(t.rsplit(".", 1)[-1] in
                            {"Broken", "BrokenSolid", "SolidBroken"} for t in markings)):
                        raise ValueError("candidate cut-in lacks a permitted adjacent Driving lane and broken boundary")
                transform = blue_wp.transform
                agents[f"agent{bid}"] = {"x": float(transform.location.x),
                                        "y": float(transform.location.y), "z": 0.5,
                                        "yaw": yaw,
                                        "initial_speed_mps": float(spec["initial_speed_mps"])}
                if case.get("blue_goal_mode") == "anchor_red_goal":
                    goals = scenario.get("agent_goal_zones", {})
                    if scenario.get("goal_mode") != "per_agent_explicit" or f"agent{rid}" not in goals:
                        raise ValueError("paired first-point cases require explicit red goals")
                    goals[f"agent{bid}"] = deepcopy(goals[f"agent{rid}"])
                evidence["waypoints"][f"agent{bid}"] = self._waypoint_record(blue_wp)
        # Require legal, separated initial footprints without moving any paired case.
        for ai in range(self.num_agents):
            for bi in range(ai + 1, self.num_agents):
                a, b = agents[f"agent{ai}"], agents[f"agent{bi}"]
                yaw = math.radians(float(a["yaw"]))
                dx, dy = float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])
                along, across = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
                if abs(along) < 5.0 and abs(across) < 2.1:
                    raise ValueError(f"candidate spawn footprints overlap: {ai}/{bi}")
        # Preserve the red task's original family/tags for its frozen reward.
        # The Mut family is separate metadata, never silently substituted.
        return scenario, evidence

    def reset_case(self, case: dict):
        self._case = deepcopy(case)
        self._mut_valid, self._mut_invalid_reason = True, None
        self._task_violations = {b: False for b in self.blue_ids}
        self._stall_seconds = {b: 0.0 for b in self.blue_ids}
        self._task_stall_evidence = {}
        self._mut_outcomes = {aid: "running" for aid in range(self.num_agents)}
        self._physical_contacts_seen = {aid: set() for aid in range(self.num_agents)}
        self.mut_events.reset()
        if getattr(self, "team_reward", None) is not None:
            self.team_reward.reset()
            self._team_reference = self.reference_store.curve(case["case_key"]) if self.reference_store else None
        scenario, evidence = self._resolve_case(case)
        self.set_scenario(scenario)
        evidence["goal_waypoints"] = {}
        for aid in range(self.num_agents):
            goal = self._get_target_location(aid)
            wp = self._get_exact_driving_waypoint(goal)
            start = evidence["waypoints"][f"agent{aid}"]
            if (wp is None or wp.is_junction or wp.road_id != start["road_id"]
                    or wp.lane_id != start["lane_id"]):
                raise ValueError(f"{case['case_key']}: agent {aid} goal is not on its same non-junction Driving lane")
            evidence["goal_waypoints"][f"agent{aid}"] = self._waypoint_record(wp)
        obs, info = super().reset()
        actor_to_id = {int(vehicle.id): aid for aid, vehicle in enumerate(self.vehicles)}
        spawn_contacts = [e for values in self.raw_collision_events.values() for e in values
                          if e["other_actor_id"] in actor_to_id]
        if spawn_contacts:
            raise RuntimeError(f"candidate case had setup vehicle contact: {spawn_contacts}")
        evidence["realized_initial_states"] = self._snapshot_kinematics()
        evidence["status"] = "runtime_map_and_spawn_checked"
        for aid, snapshot in evidence["realized_initial_states"].items():
            configured = scenario["agents"][f"agent{aid}"]
            loc = snapshot["location"]
            error = math.hypot(loc["x"] - configured["x"], loc["y"] - configured["y"])
            if error > 1.0:
                raise RuntimeError(f"paired case spawn drift {aid}: {error:.3f} m")
        self._case_geometry_evidence = evidence
        self.traffic_manager.set_random_device_seed(int(case.get("seed", self.seed)))
        self._post_tick_kinematics = deepcopy(evidence["realized_initial_states"])
        self._route_axes, self._route_progress = {}, {}
        for aid, snapshot in self._post_tick_kinematics.items():
            start = snapshot["location"]
            yaw = math.radians(snapshot["rotation"]["yaw"])
            goal = self._get_target_location(aid)
            length = (goal.x - start["x"]) * math.cos(yaw) + (goal.y - start["y"]) * math.sin(yaw)
            self._route_axes[aid] = (start["x"], start["y"], math.cos(yaw), math.sin(yaw), max(length, 1e-6))
            self._route_progress[aid] = 0.0
        self._last_mut_reward_details = {}
        self._agent_result_evidence = {}
        self._capture_agent_result_evidence(self._post_tick_kinematics, decision_sample=False)
        # Warm-up ground contacts are setup evidence, never policy contact costs.
        info["setup_contacts"] = deepcopy(self.raw_collision_events)
        self.raw_collision_events = {aid: [] for aid in range(self.num_agents)}
        info.update(self._mut_info())
        return obs, info

    def _blue_flags(self, bid):
        return {"collision": bool(self.collision_events.get(bid, False)),
                "offroad": bool(self.offroad_terminal_events.get(bid, False)),
                "road_rule": bool(self.road_violation_terminal_events.get(bid, False)
                                  or self.crosswalk_invasion_events.get(bid, False)),
                "task_violation": bool(self._task_violations.get(bid, False))}

    def _capture_agent_result_evidence(self, snapshots, *, decision_sample=True):
        """Preserve actual pre-despawn distances, decision speeds and arrival."""
        for aid, snapshot in snapshots.items():
            goal = self._get_target_location(aid)
            location = snapshot.get("location")
            if location is None:
                continue
            distance = math.sqrt((float(location["x"]) - goal.x) ** 2
                                 + (float(location["y"]) - goal.y) ** 2
                                 + (float(location["z"]) - goal.z) ** 2)
            initial = self.initial_distances.get(aid)
            result = self._agent_result_evidence.setdefault(aid, {
                "arrival_seconds": None, "arrival_frame": None,
                "speed_sum_mps": 0.0, "speed_samples": 0, "max_speed_mps": None})
            speed = snapshot.get("speed")
            if speed is not None and math.isfinite(float(speed)):
                speed = float(speed)
                result["max_speed_mps"] = max(speed, result["max_speed_mps"] or 0.0)
                if decision_sample:
                    result["speed_sum_mps"] += speed
                    result["speed_samples"] += 1
            result.update({"route_completion": route_completion(initial, distance)
                           if initial is not None else None,
                           "distance_to_goal_m": distance,
                           "initial_distance_to_goal_m": float(initial) if initial is not None else None,
                           "last_speed_mps": speed, "evidence_step": self.current_step,
                           "evidence_frame": self._last_world_frame})

    def _agent_results(self):
        results = {}
        for aid in range(self.num_agents):
            evidence = self._agent_result_evidence.get(aid, {})
            samples = int(evidence.get("speed_samples", 0))
            results[aid] = {"success": bool(self.reached_target.get(aid, False)),
                            "end_reason": self._mut_outcomes.get(aid, "running"),
                            "censored": bool(self.last_truncated.get(aid, False)),
                            "route_completion": evidence.get("route_completion"),
                            "arrival_seconds": evidence.get("arrival_seconds"),
                            "arrival_frame": evidence.get("arrival_frame"),
                            "distance_to_goal_m": evidence.get("distance_to_goal_m"),
                            "initial_distance_to_goal_m": evidence.get("initial_distance_to_goal_m"),
                            "last_speed_mps": evidence.get("last_speed_mps"),
                            "max_speed_mps": evidence.get("max_speed_mps"),
                            "mean_decision_speed_mps": evidence.get("speed_sum_mps", 0.0) / samples if samples else None,
                            "decision_speed_samples": samples,
                            "evidence_step": evidence.get("evidence_step"),
                            "evidence_frame": evidence.get("evidence_frame"),
                            "completion_basis": "frozen_route_completion_from_actual_goal_distance"}
        return results

    def _near_goal(self, aid):
        vehicle = self.vehicles[aid]
        return bool(vehicle is None or not vehicle.is_alive
                    or vehicle.get_location().distance(self._get_target_location(aid)) <= 6.0)

    def _update_blue_task_stall(self, kinematics):
        stall = self.task_config.get("stall", {})
        enabled = bool(stall.get("enabled", self.task_config.get("stall_enabled", True)))
        limit = float(stall.get("seconds", self.task_config.get("stall_seconds", 3.0)))
        threshold = float(stall.get("speed_mps", 0.2))
        if not math.isfinite(limit) or limit <= 0 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("candidate task stall constants must be valid")
        protected = self.current_step * self.timestep <= self.mut_reward_config.birth_protect_secs + 1e-9
        for bid in self.blue_ids:
            kin = kinematics.get(bid)
            reference = self.road_reference_waypoints.get(bid)
            route_speed = 0.0
            if kin is not None and reference is not None:
                yaw = math.radians(float(reference.transform.rotation.yaw))
                vel = self.vehicles[bid].get_velocity()
                route_speed = float(vel.x * math.cos(yaw) + vel.y * math.sin(yaw))
            blocker = self._pace_has_front_blocker(bid)
            near_goal = self._near_goal(bid)
            eligible = (enabled and not protected and kin is not None and kin.alive
                        and not near_goal and not blocker and route_speed < threshold
                        and self.active_mask.get(bid, False))
            self._stall_seconds[bid] = self._stall_seconds.get(bid, 0.0) + self.timestep if eligible else 0.0
            self._task_violations[bid] |= eligible and self._stall_seconds[bid] + 1e-9 >= limit
            self._task_stall_evidence[bid] = {"enabled": enabled, "consecutive_seconds": self._stall_seconds[bid],
                                            "route_speed_mps": route_speed, "front_blocker_exempt": blocker,
                                            "near_goal_exempt": near_goal, "birth_protected": protected,
                                            "task_violation": self._task_violations[bid], "partial_observability": True}

    def _compute_role_rewards(self):
        self._computing_role_base_reward = True
        try:
            method = getattr(self, "_compute_rewards_" + self.reward_version)
            rewards, details = method()
        finally:
            self._computing_role_base_reward = False
        self._team_base_details = deepcopy(details)
        self._post_tick_kinematics = self._snapshot_kinematics()
        self._capture_agent_result_evidence(self._post_tick_kinematics)
        kinematics = extract_kinematics(self, list(range(self.num_agents)),
                                       kinematic_snapshots=self._post_tick_kinematics)
        self._update_blue_task_stall(kinematics)
        flags = {b: self._blue_flags(b) for b in self.blue_ids}
        excluded = {r: self._near_goal(r) or any((
            self.collision_events.get(r, False), self.offroad_terminal_events.get(r, False),
            self.road_violation_terminal_events.get(r, False), self.crosswalk_invasion_events.get(r, False)))
                    for r in self.red_ids}
        event = self.mut_events.step(step=self.current_step, kinematics=kinematics,
                                     safety_flags=flags, near_goal=excluded, frame=self._last_world_frame)
        self._last_mut_step_events = event
        self._last_mut_reward_details = {}
        for bid in self.blue_ids:
            if not self.active_mask.get(bid, False):
                rewards[bid], details[bid] = 0.0, {"inactive": 0.0}
                continue
            kin = kinematics.get(bid)
            route_speed = self._task_stall_evidence.get(bid, {}).get("route_speed_mps", 0.0)
            missing = set(BLUE_BASE_COMPONENTS + BLUE_BASE_RISK_COMPONENTS) - set(details.get(bid, {}))
            if missing:
                raise RuntimeError(f"frozen driving reward fields missing for blue {bid}: {sorted(missing)}")
            drive, costs = blue_driving_reward(
                base_details=details.get(bid, {}), speed_mps=kin.speed if kin else 0.0,
                route_speed_mps=route_speed, front_blocker=self._pace_has_front_blocker(bid),
                near_goal=self._near_goal(bid), terminal=any(flags[bid].values()),
                safety_flags=flags[bid], config=self.mut_reward_config)
            cost = math.fsum(costs.values())
            signed_details = {"team_impact": event["team_impact"], "team_threat": event["team_threat"],
                              "driving": drive,
                              **{"cost_" + k: -self._cost_coefficient * v for k, v in costs.items()}}
            rewards[bid] = math.fsum(signed_details.values())
            details[bid] = signed_details
            self._last_mut_reward_details[bid] = {"safety_cost": cost, "costs": costs,
                                                "lambda": self._cost_coefficient,
                                                "birth_protected": event["birth_protected"],
                                                "team_impact": event["team_impact"],
                                                "team_threat": event["team_threat"],
                                                "team_timeout": 0.0, "driving": drive}
        return rewards, details

    def _check_dones(self):
        previous_active = dict(self.active_mask)
        snapshots = deepcopy(getattr(self, "_post_tick_kinematics", {}))
        event_flags = {aid: {"collision": bool(self.collision_events.get(aid, False)),
                           "offroad": bool(self.offroad_terminal_events.get(aid, False)),
                           "road_rule": bool(self.road_violation_terminal_events.get(aid, False)
                                             or self.crosswalk_invasion_events.get(aid, False))}
                       for aid in range(self.num_agents)}
        dones = super()._check_dones()
        # Unexpected actor loss is an invalid execution, not an attack success.
        unexpected = [aid for aid, active in previous_active.items() if active and aid not in snapshots]
        if unexpected:
            self._mut_valid = False
            self._mut_invalid_reason = f"unexpected_actor_loss:{unexpected}"
            self.episode_end_reason = "environment_fault"
            for aid in range(self.num_agents):
                dones[aid] = True
                self.last_terminated[aid] = not previous_active.get(aid, False)
                self.last_truncated[aid] = previous_active.get(aid, False)
                self.active_mask[aid] = False
        for bid in self.blue_ids:
            if self._task_violations.get(bid, False) and previous_active.get(bid, False):
                dones[bid], self.last_terminated[bid], self.last_truncated[bid] = True, True, False
                self.active_mask[bid] = False
                self._despawn_reached_vehicle(bid)
        # Crosswalk evidence is a road event, even though the historical base
        # lifecycle conservatively treats it as severe for red termination.
        if self.episode_end_reason == "cooperative_collision":
            red_contacts = any(event_flags[r]["collision"] for r in self.red_ids)
            if not red_contacts and any(self.crosswalk_invasion_events.get(r, False) for r in self.red_ids):
                self.episode_end_reason = "cooperative_road_violation"
        for aid in range(self.num_agents):
            if self.reached_target.get(aid, False):
                result = self._agent_result_evidence.setdefault(aid, {})
                if result.get("arrival_seconds") is None:
                    result["arrival_seconds"] = self.current_step * self.timestep
                    result["arrival_frame"] = self._last_world_frame
            if self._mut_outcomes.get(aid, "running") != "running":
                continue
            if self.reached_target.get(aid, False):
                self._mut_outcomes[aid] = "success"
            elif event_flags[aid]["collision"]:
                self._mut_outcomes[aid] = "collision"
            elif event_flags[aid]["offroad"]:
                self._mut_outcomes[aid] = "offroad"
            elif event_flags[aid]["road_rule"]:
                self._mut_outcomes[aid] = "road_rule"
            elif self._task_violations.get(aid, False):
                self._mut_outcomes[aid] = "task_violation"
            elif self.episode_end_reason == "mission_timeout":
                self._mut_outcomes[aid] = "timeout"
            elif self.last_truncated.get(aid, False):
                self._mut_outcomes[aid] = "external_truncation"
            elif self.last_terminated.get(aid, False):
                self._mut_outcomes[aid] = "adversarial_task_end" if aid in self.blue_ids else "unknown"
        return dones

    def _mut_info(self):
        metrics = self.mut_events.metrics()
        results = self._agent_results()
        contacts = deepcopy(self.raw_collision_events)
        actor_ids = {int(vehicle.id): aid for aid, vehicle in enumerate(self.vehicles)
                     if vehicle is not None}
        for events in contacts.values():
            for event in events:
                aid, other = event["agent_id"], actor_ids.get(event["other_actor_id"])
                event["other_agent_id"] = other
                if other is None:
                    event["pair_class"] = ("external_vehicle" if event["other_type_id"].startswith("vehicle.")
                                           else "static_or_unknown")
                else:
                    colors = {"red" if i in self.red_ids else "blue" for i in (aid, other)}
                    event["pair_class"] = "red_blue" if len(colors) == 2 else ("red_red" if "red" in colors else "blue_blue")
        deduplicated = {}
        for events in contacts.values():
            for event in events:
                actors = sorted((event["actor_id"], event["other_actor_id"] or -1))
                key = (event["frame"], *actors)
                if key not in deduplicated:
                    deduplicated[key] = deepcopy(event)
                    deduplicated[key]["raw_record_count"] = 0
                deduplicated[key]["raw_record_count"] += 1
                # Two vehicle sensors can report the same physical pair/frame;
                # retain both raw records while counting one event for metrics.
                deduplicated[key]["impulse_magnitude"] = max(deduplicated[key]["impulse_magnitude"], event["impulse_magnitude"])
        metrics.update({"red_success": all(self.reached_target.get(r, False) for r in self.red_ids),
                        "valid": self._mut_valid, "invalid_reason": self._mut_invalid_reason,
                        "end_reason": self.episode_end_reason,
                        "agent_outcomes": dict(self._mut_outcomes),
                        "agent_results": results,
                        "red_results": {rid: results[rid] for rid in self.red_ids},
                        "task_stall": deepcopy(self._task_stall_evidence),
                        "case_key": self._case["case_key"] if self._case else None,
                        "family": self._case["family"] if self._case else None,
                        "contract_status": "candidate",
                        "physical_contacts": contacts,
                        "contact_events": list(deduplicated.values())})
        metrics["route_progress"] = dict(getattr(self, "_route_progress", {}))
        if getattr(self, "team_reward", None) is not None:
            metrics.update(self.team_reward.metrics(self.mut_events.blue_eligible))
        reward_residual = max((abs(value.get("reward_residual", 0.0))
                               for value in self._last_mut_reward_details.values()), default=0.0)
        return {"agent_terminated": dict(self.last_terminated),
                "agent_truncated": dict(self.last_truncated),
                "active_mask": dict(self.active_mask), "mut_metrics": metrics,
                "world_tick_counts": dict(self._world_tick_counts),
                "world_tick_scope": "explicit_client_ticks_including_reset_and_cleanup",
                "case_geometry": deepcopy(self._case_geometry_evidence),
                "post_tick_kinematics": deepcopy(getattr(self, "_post_tick_kinematics", {})),
                "blue_reward_audit": deepcopy(self._last_mut_reward_details),
                "blue_reward_formula": ("shared C1 progress utility + driving - risk - team safety"
                    if getattr(self, "team_reward", None) is not None
                    else "team_impact + team_threat + team_timeout + driving - lambda * safety_cost"),
                "blue_reward_max_abs_residual": reward_residual}

    def step(self, actions: dict):
        if self._case is None:
            raise RuntimeError("reset_case must precede policy actions")
        if self.episode_end_reason != "running":
            raise RuntimeError("cannot tick a finished episode")
        for aid in range(self.num_agents):
            if self.active_mask.get(aid, False):
                if aid not in actions:
                    raise ValueError(f"active physical agent {aid} lacks a control")
                value = np.asarray(actions[aid], dtype=np.float32)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError("environment controls must be finite [throttle, steer, brake]")
                if not (0.0 <= value[0] <= 1.0 and -1.0 <= value[1] <= 1.0 and 0.0 <= value[2] <= 1.0):
                    raise ValueError("environment control outside physical bounds")
        active_before = dict(self.active_mask)
        obs, rewards, dones, info = super().step(actions)
        # The parent captures observations before goal/failed-car despawn.
        # Keep that exact physical state for agent-specific truncation
        # bootstrap even though the next-action observations use empty slots.
        bootstrap_state = np.concatenate([np.asarray(obs[aid], dtype=np.float32)
                                          for aid in range(self.num_agents)]).copy()
        if bootstrap_state.shape != (self.num_agents * 20,) or not np.isfinite(bootstrap_state).all():
            raise RuntimeError("Mut terminal bootstrap global-state contract mismatch")
        info["bootstrap_global_state"] = bootstrap_state
        if self.episode_end_reason != "running":
            bonus = self.mut_events.finalize(end_reason=self.episode_end_reason,
                    unfinished_red_ids=[r for r in self.red_ids if not self.reached_target.get(r, False)],
                    step=self.current_step)
            if bonus:
                for bid in self.blue_ids:
                    if bid in self._last_mut_reward_details:
                        rewards[bid] += bonus
                        info["reward_details"][bid]["team_timeout"] = bonus
                        self._last_mut_reward_details[bid]["team_timeout"] = bonus
        for bid, audit in self._last_mut_reward_details.items():
            expected = math.fsum([audit["team_impact"], audit["team_threat"], audit["team_timeout"],
                                  audit["driving"], -audit["lambda"] * audit["safety_cost"]])
            audit["expected_reward"] = expected
            audit["actual_reward"] = float(rewards[bid])
            audit["reward_residual"] = float(rewards[bid]) - expected
        for aid, snapshot in getattr(self, "_post_tick_kinematics", {}).items():
            if aid in getattr(self, "_route_axes", {}):
                x, y, fx, fy, length = self._route_axes[aid]
                loc = snapshot["location"]
                self._route_progress[aid] = min(1.0, max(0.0, ((loc["x"] - x) * fx + (loc["y"] - y) * fy) / length))
                if self.reached_target.get(aid, False):
                    self._route_progress[aid] = 1.0
        if getattr(self, "team_reward", None) is not None:
            costs = {b: audit["costs"] for b, audit in self._last_mut_reward_details.items()}
            utility, components = self.team_reward.step(step=self.current_step,
                progress=self._route_progress, reached=self.reached_target,
                eligible=self.mut_events.blue_eligible and not self._last_mut_step_events.get("birth_protected", False),
                safety=self.mut_events.safety,
                costs=costs, base_details={b: self._team_base_details.get(b, {}) for b in self.blue_ids},
                reference=self._team_reference)
            self._last_mut_reward_details = {}
            for bid in self.blue_ids:
                if active_before.get(bid, False):
                    rewards[bid] = utility
                    info["reward_details"][bid] = dict(components)
                    self._last_mut_reward_details[bid] = {"team_utility": utility,
                        "expected_reward": utility, "actual_reward": utility, "reward_residual": 0.0}
            info["team_step_reward"] = utility
        if self.follow_spectator:
            self._update_overhead_spectator()
        info["episode_end_reason"] = self.episode_end_reason
        info.update(self._mut_info())
        # Departures return zero observations in physical slots; retained
        # post-tick snapshots provide terminal diagnostics before despawn.
        obs = self._get_observations()
        return obs, rewards, dones, info

    def get_global_state(self):
        state = super().get_global_state()
        if state.shape != (self.num_agents * 20,):
            raise RuntimeError("Mut global-state observation contract mismatch")
        return state
