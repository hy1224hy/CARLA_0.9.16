import types
import unittest

import numpy as np

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from training.adversarial_reward import AdversarialRewardComputer, RewardConfig
from training.lifecycle_semantics import (
    bootstrap_state_for_boundary,
    validate_lifecycle_flags,
)
from training.train_no_rllib_mappo_robust_v2 import compute_gae
from utils.adversarial_metrics import extract_kinematics


class _Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0, in_goal=False):
        self.x = x
        self.y = y
        self.z = z
        self.in_goal = in_goal


class _Rotation:
    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Vector()
        self.rotation = rotation or _Rotation()


class _Control:
    throttle = 0.2
    steer = 0.1
    brake = 0.0


class _Vehicle:
    def __init__(self, *, alive=True, in_goal=False, x=0.0, y=0.0):
        self.is_alive = alive
        self._transform = _Transform(_Vector(x, y, in_goal=in_goal))

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return _Vector(1.0, 0.0, 0.0)

    def get_angular_velocity(self):
        return _Vector()

    def get_control(self):
        return _Control()


def _make_env(
    *,
    current_step=1,
    num_agents=3,
    adversarial_agent_ids=None,
    adversary_collision_ends_episode=False,
):
    env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
    env.num_agents = num_agents
    if adversarial_agent_ids is None:
        adversarial_agent_ids = [2] if num_agents >= 3 else []
    env.adversarial_agent_ids = list(adversarial_agent_ids)
    env.adversary_collision_ends_episode = adversary_collision_ends_episode
    env.vehicles = [_Vehicle() for _ in range(num_agents)]
    env.reached_target = {i: False for i in range(num_agents)}
    env.collision_events = {i: False for i in range(num_agents)}
    env.crosswalk_invasion_events = {i: False for i in range(num_agents)}
    env.lane_invasion_events = {i: False for i in range(num_agents)}
    env.goal_zone = {}
    env.current_step = current_step
    env.max_steps = 10
    env._is_in_goal_zone = lambda location: location.in_goal

    def _despawn(self, agent_id):
        self.vehicles[agent_id].is_alive = False

    env._despawn_reached_vehicle = types.MethodType(_despawn, env)
    return env


def _snapshot(x, y, *, yaw=0.0, vx=0.0, reached=False):
    return {
        "location": {"x": x, "y": y, "z": 0.0},
        "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
        "velocity": {"x": vx, "y": 0.0, "z": 0.0},
        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "control": {"throttle": 0.0, "steer": 0.0, "brake": 0.0},
        "speed": abs(vx),
        "alive": True,
        "reached": reached,
    }


class EpisodeLifecycleTests(unittest.TestCase):
    def test_step_publishes_snapshot_before_blue_despawn(self):
        env = _make_env()
        env.vehicles[2]._transform.location.x = 2.0
        env._total_env_steps = 0
        env.use_tensorboard = False
        env.prev_controls = {}
        env._vehicle_on_crosswalk = lambda vehicle: False
        env._get_observations = lambda: {i: [0.0] * 19 for i in range(3)}
        env._compute_rewards = lambda: (
            {0: 0.0, 1: 0.0, 2: -300.0},
            {0: {}, 1: {}, 2: {"collision": -300.0}},
        )
        env._log_to_tensorboard = lambda *args: None

        def _tick():
            # 模拟本 tick 内车辆移动后触发碰撞；info 应保留移动前 x=2 的快照。
            env.vehicles[2]._transform.location.x = 99.0
            env.collision_events[2] = True

        env.world = types.SimpleNamespace(tick=_tick)

        observations, rewards, dones, info = env.step({})

        self.assertEqual(info["episode_end_reason"], "running")
        self.assertEqual(info["active_mask"], {0: True, 1: True, 2: False})
        self.assertTrue(info["collisions"][2])
        self.assertEqual(
            info["pre_collision_kinematics"][2]["location"]["x"], 2.0
        )
        self.assertTrue(dones[2])
        self.assertFalse(env.vehicles[2].is_alive)

    def test_blue_collision_only_terminates_blue(self):
        env = _make_env()
        env.collision_events[2] = True

        dones = env._check_dones()

        self.assertEqual(dones, {0: False, 1: False, 2: True})
        self.assertEqual(env.active_mask, {0: True, 1: True, 2: False})
        self.assertTrue(env.last_terminated[2])
        self.assertFalse(env.last_truncated[2])
        self.assertEqual(env.episode_end_reason, "running")

    def test_all_blue_dead_does_not_end_red_episode(self):
        env = _make_env()
        env.vehicles[2].is_alive = False

        dones = env._check_dones()

        self.assertEqual(dones, {0: False, 1: False, 2: True})
        self.assertEqual(env.active_mask, {0: True, 1: True, 2: False})
        self.assertEqual(env.episode_end_reason, "running")

    def test_single_red_goal_terminates_only_that_red(self):
        env = _make_env()
        env.vehicles[0]._transform.location.in_goal = True

        dones = env._check_dones()

        self.assertEqual(dones, {0: True, 1: False, 2: False})
        self.assertEqual(env.episode_end_reason, "running")
        self.assertEqual(env.last_terminated, {0: True, 1: False, 2: False})
        self.assertEqual(env.last_truncated, {0: False, 1: False, 2: False})
        self.assertEqual(env.active_mask, {0: False, 1: True, 2: True})

    def test_cooperative_success_terminates_live_blue(self):
        env = _make_env()
        env.vehicles[0]._transform.location.in_goal = True
        env.vehicles[1]._transform.location.in_goal = True

        dones = env._check_dones()

        self.assertTrue(all(dones.values()))
        self.assertEqual(env.episode_end_reason, "cooperative_success")
        self.assertTrue(env.last_terminated[0])
        self.assertTrue(env.last_terminated[1])
        self.assertTrue(env.last_terminated[2])
        self.assertFalse(any(env.last_truncated.values()))
        self.assertFalse(any(env.active_mask.values()))

    def test_cooperative_collision_truncates_other_red_and_terminates_blue(self):
        env = _make_env()
        env.collision_events[0] = True

        dones = env._check_dones()

        self.assertTrue(all(dones.values()))
        self.assertEqual(env.episode_end_reason, "cooperative_collision")
        self.assertTrue(env.last_terminated[0])
        self.assertFalse(env.last_terminated[1])
        self.assertTrue(env.last_truncated[1])
        self.assertTrue(env.last_terminated[2])
        self.assertFalse(env.last_truncated[2])
        self.assertFalse(any(env.active_mask.values()))

    def test_base_cooperative_collision_truncates_unaffected_red(self):
        env = _make_env(num_agents=2, adversarial_agent_ids=[])
        env.collision_events[0] = True

        dones = env._check_dones()

        self.assertEqual(dones, {0: True, 1: True})
        self.assertEqual(env.episode_end_reason, "cooperative_collision")
        self.assertEqual(env.last_terminated, {0: True, 1: False})
        self.assertEqual(env.last_truncated, {0: False, 1: True})

    def test_base_all_agents_reached_terminates_all_reds(self):
        env = _make_env(num_agents=2, adversarial_agent_ids=[])
        env.vehicles[0]._transform.location.in_goal = True
        env.vehicles[1]._transform.location.in_goal = True

        dones = env._check_dones()

        self.assertEqual(dones, {0: True, 1: True})
        self.assertEqual(env.episode_end_reason, "all_agents_reached")
        self.assertEqual(env.last_terminated, {0: True, 1: True})
        self.assertEqual(env.last_truncated, {0: False, 1: False})

    def test_enabled_adversary_global_collision_keeps_legacy_semantics(self):
        env = _make_env(adversary_collision_ends_episode=True)
        env.collision_events[2] = True

        dones = env._check_dones()

        self.assertTrue(all(dones.values()))
        self.assertEqual(env.episode_end_reason, "adversary_collision")
        self.assertEqual(env.last_terminated, {0: False, 1: False, 2: True})
        self.assertEqual(env.last_truncated, {0: True, 1: True, 2: False})

    def test_max_steps_is_truncation(self):
        env = _make_env(current_step=10)

        dones = env._check_dones()

        self.assertTrue(all(dones.values()))
        self.assertEqual(env.episode_end_reason, "max_steps")
        self.assertFalse(any(env.last_terminated.values()))
        self.assertTrue(all(env.last_truncated.values()))
        self.assertFalse(any(env.active_mask.values()))

    def test_natural_termination_is_not_overwritten_by_later_timeout(self):
        env = _make_env(current_step=10)
        env.vehicles[0]._transform.location.in_goal = True

        env._check_dones()

        self.assertEqual(env.episode_end_reason, "max_steps")
        self.assertEqual(env.last_terminated, {0: True, 1: False, 2: False})
        self.assertEqual(env.last_truncated, {0: False, 1: True, 2: True})

    def test_collision_snapshot_restores_despawned_blue_reward(self):
        env = types.SimpleNamespace(
            vehicles=[_Vehicle(alive=False) for _ in range(3)],
            reached_target={0: False, 1: False, 2: False},
        )
        snapshots = {
            0: _snapshot(0.0, 0.0, vx=1.0),
            1: _snapshot(-5.0, 0.0, vx=1.0),
            2: _snapshot(1.0, 0.0, vx=1.0),
        }

        kinematics = extract_kinematics(env, [0, 1, 2], {}, snapshots)
        self.assertIn(2, kinematics)

        computer = AdversarialRewardComputer(
            [0, 1], [2], RewardConfig(self_collision_coef_init=1.0)
        )
        rewards, details = computer.step(
            env,
            {2: {"collision": -300.0}},
            {},
            step_idx=5,
            collisions={0: False, 1: False, 2: True},
            kinematic_snapshots=snapshots,
        )

        self.assertIn(2, rewards)
        self.assertEqual(details[2]["c_self"], -300.0)
        self.assertEqual(details[2]["self_coll_term"], -300.0)


class LifecycleBootstrapTests(unittest.TestCase):
    def test_lifecycle_flags_are_exclusive_and_cover_done(self):
        self.assertEqual(validate_lifecycle_flags(False, False, False), (False, False, False))
        self.assertEqual(validate_lifecycle_flags(True, True, False), (True, True, False))
        self.assertEqual(validate_lifecycle_flags(True, False, True), (True, False, True))
        with self.assertRaises(ValueError):
            validate_lifecycle_flags(True, True, True)
        with self.assertRaises(ValueError):
            validate_lifecycle_flags(True, False, False)

    def test_bootstrap_state_exists_only_for_truncation(self):
        state = np.array([1.0, 2.0], dtype=np.float32)

        self.assertIsNone(bootstrap_state_for_boundary(True, True, False, state))
        bootstrap = bootstrap_state_for_boundary(True, False, True, state)

        np.testing.assert_array_equal(bootstrap, state)
        self.assertIsNot(bootstrap, state)

    def test_gae_bootstraps_truncation_but_not_termination(self):
        rewards = np.array([1.0], dtype=np.float32)
        values = np.array([2.0], dtype=np.float32)
        dones = np.array([1.0], dtype=np.float32)

        _, terminal_return = compute_gae(
            rewards, values, dones, np.array([1.0], dtype=np.float32),
            gamma=0.9, gae_lambda=0.95, bootstrap_value=10.0,
        )
        _, truncated_return = compute_gae(
            rewards, values, dones, np.array([0.0], dtype=np.float32),
            gamma=0.9, gae_lambda=0.95, bootstrap_value=10.0,
        )

        self.assertAlmostEqual(float(terminal_return[0]), 1.0, places=6)
        self.assertAlmostEqual(float(truncated_return[0]), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
