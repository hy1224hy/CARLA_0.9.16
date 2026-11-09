"""Short offline checks for reward evidence and four-car lifecycle boundaries."""
from pathlib import Path
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from environment.mut_mappo_events import (
    BLUE_BASE_COMPONENTS, BLUE_BASE_RISK_COMPONENTS,
    MutEpisodeEvents, MutRewardConfig, blue_driving_reward,
)
from utils.adversarial_metrics import AgentKinematics


def kin(x, speed):
    return AgentKinematics(loc=(x, 0.0), fwd=(1.0, 0.0), right=(0.0, -1.0),
                           fwd_speed=speed, speed=abs(speed))


class MutEventTests(unittest.TestCase):
    def detector(self, **values):
        return MutEpisodeEvents(MutRewardConfig.from_dict(values))

    def respond(self, detector, start=31, overlapping_blues=True):
        impact = 0.0
        for step in range(start, start + 8):
            kinematics = {0: kin(0, 5.0 if step == start else 2.0), 2: kin(8, 2.0)}
            if overlapping_blues:
                kinematics[3] = kin(10, 2.0)
            result = detector.step(step=step, kinematics=kinematics)
            impact += result["team_impact"]
        return impact

    def test_birth_protection_covers_all_attack_positive_terms(self):
        events = self.detector(birth_protect_secs=3.0)
        for step in range(1, 31):
            result = events.step(step=step, kinematics={0: kin(0, 5 if step == 1 else 1), 2: kin(8, 1)})
            self.assertEqual(result["team_impact"] + result["team_threat"], 0.0)
        self.assertEqual(events.finalize(end_reason="mission_timeout", unfinished_red_ids=(0,), step=30), 0)
        self.assertFalse(events.metrics()["response"])

    def test_response_persists_and_red_event_is_not_double_counted(self):
        events = self.detector()
        self.assertEqual(self.respond(events), 5.0)
        self.assertEqual(events.metrics()["response_event_count"], 1)
        self.assertEqual(events.response_events[0]["blue_ids"], [2, 3])
        # Sustained slowing may shape threat, but cannot reissue sparse events.
        for step in range(39, 44):
            self.assertEqual(events.step(step=step, kinematics={0: kin(0, 2), 2: kin(8, 2)})["team_impact"], 0)
        self.assertLessEqual(events.threat_total, events.config.threat_cap_per_episode)

    def test_contacts_disable_current_and_future_team_attack_credit(self):
        events = self.detector()
        self.respond(events)
        result = events.step(step=39, kinematics={0: kin(0, 1), 2: kin(8, 1)},
                             safety_flags={3: {"collision": True}})
        self.assertEqual(result["team_impact"] + result["team_threat"], 0)
        self.assertFalse(events.metrics()["blue_safe"])
        self.assertTrue(events.metrics()["response"])
        self.assertFalse(events.metrics()["qualified_response"])
        self.assertEqual(events.finalize(end_reason="mission_timeout", unfinished_red_ids=(0,), step=500), 0)

    def test_only_evidence_linked_real_timeout_gets_one_team_bonus(self):
        for reason in ("cooperative_collision", "environment_fault", "budget_interrupt"):
            events = self.detector()
            self.respond(events)
            self.assertEqual(events.finalize(end_reason=reason, unfinished_red_ids=(0,), step=500), 0)
        events = self.detector()
        self.respond(events)
        self.assertEqual(events.finalize(end_reason="mission_timeout", unfinished_red_ids=(1,), step=500), 0)
        events = self.detector()
        self.respond(events)
        self.assertEqual(events.finalize(end_reason="mission_timeout", unfinished_red_ids=(0,), step=500), 15)
        self.assertEqual(events.finalize(end_reason="mission_timeout", unfinished_red_ids=(0,), step=500), 0)

    def test_costs_are_nonnegative_and_overspeed_has_no_plateau(self):
        cfg = MutRewardConfig.from_dict()
        common = dict(base_details={"progress": 0.1, "collision": 100}, route_speed_mps=5,
                      front_blocker=False, near_goal=False, terminal=True,
                      safety_flags={"collision": True}, config=cfg)
        drive, costs = blue_driving_reward(speed_mps=10, **common)
        _, fast_costs = blue_driving_reward(speed_mps=20, **common)
        self.assertAlmostEqual(drive, 0.03)
        self.assertTrue(all(value >= 0 for value in costs.values()))
        self.assertGreater(fast_costs["overspeed"], costs["overspeed"])
        self.assertEqual(costs["collision"], 50)

    def test_driving_fields_match_frozen_v6_source_and_keep_nonzero_terms(self):
        from environment.base_reward_v6 import REWARD_COMPONENTS
        self.assertTrue(set(BLUE_BASE_COMPONENTS + BLUE_BASE_RISK_COMPONENTS) <= set(REWARD_COMPONENTS))
        drive, _ = blue_driving_reward(
            base_details={"progress": 1.0, "success": 2.0,
                          "longitudinal_rate": -0.1, "steer_rate": -0.2},
            speed_mps=3, route_speed_mps=3, front_blocker=False,
            near_goal=False, terminal=False, safety_flags={}, config=MutRewardConfig.from_dict())
        self.assertAlmostEqual(drive, 0.81)

    def test_candidate_split_initial_states_are_distinct(self):
        path = Path(__file__).resolve().parents[1] / "data/mut_mappo/cases_v1.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cases = {case["case_key"]: case for case in manifest["cases"]}
        self.assertEqual(len(cases), len(manifest["cases"]))
        seen = set()
        for split, keys in manifest["splits"].items():
            for key in keys:
                self.assertNotIn(key, seen)
                seen.add(key)
            if split == "gate":
                for family in ("blocking", "cut_in"):
                    self.assertEqual(sum(cases[k]["family"] == family for k in keys), 8)
        states = [json.dumps({k: c[k] for k in ("base_scenario", "red_overrides", "blue_spawn_offsets")}, sort_keys=True)
                  for c in cases.values()]
        self.assertEqual(len(states), len(set(states)))


class MutLifecycleTests(unittest.TestCase):
    @staticmethod
    def env():
        # Actual lifecycle code, offline actors only; no Client/world/tick.
        from environment.mut_mappo_env import MutMAPPOEnv
        env = MutMAPPOEnv.__new__(MutMAPPOEnv)
        env.num_agents = 4
        env.red_ids, env.blue_ids, env.adversarial_agent_ids = (0, 1), (2, 3), [2, 3]
        env.adversary_collision_ends_episode = False
        env.reward_version = "v6"
        env.current_step, env.max_steps = 1, 500
        env.timestep, env._last_world_frame = 0.1, 1
        env._agent_result_evidence = {}
        env._last_mut_reward_details = {}
        env.vehicles = [SimpleNamespace(is_alive=True, get_transform=lambda: SimpleNamespace(location=object())) for _ in range(4)]
        env.active_mask = {i: True for i in range(4)}
        env.last_terminated = {i: False for i in range(4)}
        env.last_truncated = {i: False for i in range(4)}
        env.reached_target = {i: False for i in range(4)}
        env.collision_events = {i: False for i in range(4)}
        env.crosswalk_invasion_events = {i: False for i in range(4)}
        env.offroad_terminal_events = {i: False for i in range(4)}
        env.road_violation_terminal_events = {i: False for i in range(4)}
        env.road_legality_decisions = {i: {"success_eligible": True} for i in range(4)}
        env.last_road_occupancy = {}
        env._task_violations = {2: False, 3: False}
        env._mut_outcomes = {i: "running" for i in range(4)}
        env._post_tick_kinematics = {i: {"alive": True} for i in range(4)}
        env._goal_zone_for_agent = lambda aid: {}
        env.goal_hits = set()
        env._is_agent_in_goal_zone = lambda loc, aid: aid in env.goal_hits
        env._despawn_reached_vehicle = lambda aid: setattr(env.vehicles[aid], "is_alive", False)
        return env

    def test_both_blue_failures_do_not_end_the_red_task(self):
        env = self.env()
        env.collision_events[2] = env.collision_events[3] = True
        dones = env._check_dones()
        self.assertEqual(env.episode_end_reason, "running")
        self.assertFalse(dones[0])
        self.assertFalse(dones[1])
        self.assertEqual(env._mut_outcomes[2], "collision")
        self.assertFalse(env.active_mask[2])

    def test_red_failure_externally_truncates_only_unfinished_other_red(self):
        env = self.env()
        env.collision_events[0] = True
        env._check_dones()
        self.assertTrue(env.last_terminated[0])
        self.assertTrue(env.last_truncated[1])
        self.assertFalse(env.last_terminated[1])
        self.assertTrue(env.last_terminated[2])
        self.assertEqual(env._mut_outcomes[1], "external_truncation")

    def test_blue_task_stall_only_removes_that_blue(self):
        env = self.env()
        env._task_violations[2] = True
        env._check_dones()
        self.assertEqual(env.episode_end_reason, "running")
        self.assertFalse(env.active_mask[2])
        self.assertTrue(env.active_mask[3])
        self.assertFalse(env.vehicles[2].is_alive)
        self.assertEqual(env._mut_outcomes[2], "task_violation")

    def test_real_timeout_is_natural_terminal_for_red_and_blue(self):
        env = self.env()
        env.current_step = 500
        env._check_dones()
        self.assertEqual(env.episode_end_reason, "mission_timeout")
        self.assertTrue(all(env.last_terminated.values()))
        self.assertFalse(any(env.last_truncated.values()))

    def test_red_success_is_team_end_without_requiring_blue_arrival(self):
        env = self.env()
        env.goal_hits = {0, 1}
        env._check_dones()
        self.assertEqual(env.episode_end_reason, "cooperative_success")
        self.assertTrue(env.reached_target[0] and env.reached_target[1])
        self.assertTrue(all(env.last_terminated.values()))
        self.assertEqual(env._mut_outcomes[2], "adversarial_task_end")

    def test_reset_and_decision_ticks_both_enter_the_actual_cost_audit(self):
        env = self.env()
        records = []
        env._world_tick_counts = {"total": 0, "decision": 0, "setup": 0}
        env._tick_audit_callback = records.append
        env.world = SimpleNamespace(tick=lambda: 100)
        env._tick_world()
        env._tick_world("decision")
        self.assertEqual(env._world_tick_counts, {"total": 2, "decision": 1, "setup": 1})
        self.assertEqual([(r["kind"], r["ticks"]) for r in records], [("setup", 1), ("decision", 1)])

    def test_contact_callback_preserves_actor_frame_and_impulse_evidence(self):
        env = self.env()
        sensor = SimpleNamespace(listen=lambda cb: setattr(sensor, "callback", cb))
        env.blueprint_library = SimpleNamespace(find=lambda name: name)
        env.world = SimpleNamespace(spawn_actor=lambda *args, **kwargs: sensor)
        env.collision_sensors = {}
        env.raw_collision_events = {}
        vehicle = SimpleNamespace(id=123)
        env._attach_collision_sensor(2, vehicle)
        sensor.callback(SimpleNamespace(other_actor=SimpleNamespace(id=456, type_id="vehicle.tesla.model3"),
                                        frame=789, timestamp=12.3, normal_impulse=SimpleNamespace(x=3, y=4, z=0)))
        event = env.raw_collision_events[2][0]
        self.assertEqual((event["actor_id"], event["other_actor_id"], event["frame"]), (123, 456, 789))
        self.assertEqual(event["impulse_magnitude"], 5)
        self.assertEqual(event["responsibility"], "unknown")

    def test_truncation_bootstrap_keeps_pre_cleanup_physical_observations(self):
        from environment.carla_multi_agent_env import CarlaMultiAgentEnv
        env = self.env()
        env._case = {"case_key": "offline_bootstrap"}
        env.episode_end_reason = "running"
        env.follow_spectator = False
        before = {aid: np.full(20, aid + 1.0, dtype=np.float32) for aid in range(4)}
        after = {aid: np.zeros(20, dtype=np.float32) for aid in range(4)}
        env._get_observations = lambda: after
        env._mut_info = lambda: {}
        info = {"truncated": {aid: True for aid in range(4)}}
        with patch.object(CarlaMultiAgentEnv, "step", return_value=(before, {}, {}, info)):
            observations, _, _, result = env.step({aid: np.zeros(3) for aid in range(4)})
        np.testing.assert_array_equal(result["bootstrap_global_state"], np.concatenate(list(before.values())))
        self.assertTrue(all(np.count_nonzero(value) == 0 for value in observations.values()))


if __name__ == "__main__":
    unittest.main()
