"""Fake-parent checks for reward replacement and mutation-free net-gap logging.

The fake superclass avoids importing CARLA and never creates a client/world.
"""
import copy
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from environment.base_driving_reward import reward_spec, startup_efficiency


class FakeParent:
    def _compute_rewards_v6(self):
        rewards, details, diagnostics = copy.deepcopy(self.fixture)
        self.last_reward_diagnostics = diagnostics
        return rewards, details


fake_module = types.ModuleType("environment.carla_multi_agent_env")
fake_module.CarlaMultiAgentEnv = FakeParent
spec = importlib.util.spec_from_file_location(
    "fake_base_driving_env_under_test",
    Path(__file__).resolve().parents[1] / "environment/base_driving_env.py",
)
adapter = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"environment.carla_multi_agent_env": fake_module}):
    spec.loader.exec_module(adapter)


@dataclass
class Vector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Rotation:
    yaw: float = 0.0


class Transform:
    def __init__(self, x, y=0.0, yaw=0.0):
        self.location = Vector(x, y)
        self.rotation = Rotation(yaw)

    def transform(self, point):
        # Deliberately mutate input and return None: both CARLA API behaviours
        # are handled without contaminating the actor-owned bbox point.
        angle = math.radians(self.rotation.yaw)
        x = self.location.x + point.x * math.cos(angle) - point.y * math.sin(angle)
        y = self.location.y + point.x * math.sin(angle) + point.y * math.cos(angle)
        point.x, point.y = x, y


class Vehicle:
    is_alive = True

    def __init__(self, x, speed, y=0.0):
        self.transform = Transform(x, y)
        self.velocity = Vector(speed)
        self.bounding_box = types.SimpleNamespace(location=Vector(.5), extent=Vector(2.0, 1.0), rotation=Rotation())

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity


def reward_env(profile):
    env = adapter.BaseDrivingEnv.__new__(adapter.BaseDrivingEnv)
    env.base_driving_reward_profile = profile
    env.current_step, env.timestep = 10, .1
    env.current_scenario_family = "free_flow"
    d = {"closing_speed_mps": 0.0, "distance_to_goal_m": 10.0,
         "pace_free_flow_tag": 1.0, "pace_non_interactive_tag": 1.0,
         "pace_front_blocker_exempt": 0.0, "pace_complex_interaction_exempt": 0.0,
         "pace_terminal_exempt": 0.0}
    terminal = {**d, "pace_terminal_exempt": 1.0}
    env.fixture = ({0: .95, 1: 0.0},
                   {0: {"progress": 1.0, "free_flow_pace_deficit": -.05, "lane_deviation": 0.0},
                    1: {"progress": 0.0, "free_flow_pace_deficit": 0.0, "lane_deviation": 0.0}},
                   {0: d, 1: terminal})
    return env


class RewardAdapterTests(unittest.TestCase):
    def test_baseline_exactly_preserves_parent_and_startup_changes_only_pace(self):
        env = reward_env("v6")
        rewards, details = env._compute_rewards_v6()
        self.assertEqual(rewards, env.fixture[0])
        self.assertEqual(details, env.fixture[1])
        env = reward_env("startup_v1")
        rewards, details = env._compute_rewards_v6()
        self.assertAlmostEqual(details[0]["free_flow_pace_deficit"], -.15)
        self.assertEqual(details[0]["progress"], 1.0)
        self.assertAlmostEqual(rewards[0], .85)
        self.assertEqual(details[1]["free_flow_pace_deficit"], 0.0)
        self.assertEqual(rewards[1], 0.0)
        for aid in rewards:
            self.assertEqual(rewards[aid], math.fsum(details[aid].values()))

    def test_own_terminal_blocker_and_complex_traffic_remain_exempt(self):
        for field in ("pace_terminal_exempt", "pace_front_blocker_exempt", "pace_complex_interaction_exempt"):
            env = reward_env("startup_v1")
            env.fixture[2][0][field] = 1.0
            rewards, details = env._compute_rewards_v6()
            self.assertEqual(details[0]["free_flow_pace_deficit"], 0.0)
            self.assertEqual(rewards[0], 1.0)

    def test_pace_grace_near_goal_and_profile_identity_are_explicit(self):
        args = dict(distance_to_goal_m=10.0, elapsed_seconds=.8, timestep=.1,
                    eligible=True, front_blocker=False, complex_interaction=False, terminal_event=False)
        self.assertEqual(startup_efficiency(0, **args).reward, 0.0)
        args["elapsed_seconds"] = .9
        self.assertAlmostEqual(startup_efficiency(0, **args).reward, -.15)
        args["distance_to_goal_m"] = 4.0
        self.assertEqual(startup_efficiency(.5, **args).reward, 0.0)
        self.assertEqual(reward_spec("v6")["changed_components"], [])
        self.assertEqual(reward_spec("startup_v1")["changed_components"], ["free_flow_pace_deficit"])
        with self.assertRaises(ValueError):
            reward_spec("unknown")

    def test_bbox_and_road_tangent_diagnostics_do_not_mutate_actor_geometry(self):
        for mapping in (False, True):
            env = adapter.BaseDrivingEnv.__new__(adapter.BaseDrivingEnv)
            ego, front = Vehicle(0.0, 5.0), Vehicle(10.0, 3.0)
            env.vehicles = {0: ego, 1: front} if mapping else [ego, front]
            env.road_reference_waypoints = {}
            first = env.get_longitudinal_safety_state()
            second = env.get_longitudinal_safety_state()
            self.assertEqual(first, second)
            self.assertEqual(ego.bounding_box.location, Vector(.5))
            self.assertEqual(front.bounding_box.location, Vector(.5))
            self.assertAlmostEqual(first[0]["front_center_distance_m"], 10.0)
            self.assertAlmostEqual(first[0]["front_gap_m"], 6.0)
            self.assertAlmostEqual(first[0]["relative_speed_mps"], 2.0)
            self.assertAlmostEqual(first[0]["ttc_s"], 3.0)
            self.assertEqual(first[1]["has_front_vehicle"], 0.0)


if __name__ == "__main__":
    unittest.main()
