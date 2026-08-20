import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    METRICS_PROTOCOL_VERSION,
    MetricsProtocolError,
    env_action_diagnostics,
    reward_components,
    reward_diagnostics,
    reward_schema_document,
    validate_active_reward_payload,
)
from training.action_protocol import action_tanh_to_env
from utils.training_metrics_recorder import TrainingMetricsRecorder
from utils.validate_tensorboard_run import validate_metrics_run
from environment.carla_multi_agent_env import CarlaMultiAgentEnv


def _reward_details(**overrides):
    values = {name: 0.0 for name in reward_components("v2")}
    values.update(overrides)
    return values


def _reward_diagnostics(**overrides):
    values = {name: 0.0 for name in reward_diagnostics("v2")}
    values.update(
        {
            "lidar_min_normalized": 1.0,
            "ttc_min_seconds_capped": 3.0,
        }
    )
    values.update(overrides)
    return values


def _reward(details):
    return math.fsum(details.values())


class MetricsProtocolTests(unittest.TestCase):
    def test_action_diagnostics_separate_zero_saturation_and_modes(self):
        throttle = env_action_diagnostics([0.5, 0.0, 0.0], context="throttle")
        self.assertEqual(throttle["throttle_mode"], 1.0)
        self.assertEqual(throttle["brake_zero"], 1.0)
        self.assertEqual(throttle["brake_saturation"], 0.0)
        self.assertEqual(throttle["longitudinal_conflict"], 0.0)

        saturated = env_action_diagnostics([1.0, -1.0, 0.0], context="saturated")
        self.assertEqual(saturated["throttle_saturation"], 1.0)
        self.assertEqual(saturated["steer_saturation"], 1.0)
        self.assertEqual(saturated["throttle_boundary"], 1.0)

        coast = env_action_diagnostics([0.0, 0.0, 0.0], context="coast")
        self.assertEqual(coast["coast_mode"], 1.0)

        conflict = env_action_diagnostics([0.2, 0.0, 0.3], context="conflict")
        self.assertEqual(conflict["longitudinal_conflict"], 1.0)
        self.assertEqual(
            sum(
                conflict[name]
                for name in (
                    "throttle_mode",
                    "brake_mode",
                    "coast_mode",
                    "longitudinal_conflict",
                )
            ),
            1.0,
        )
        self.assertEqual(set(conflict), set(ACTION_RATE_NAMES))

        for action in ([0.0, 0.0], [1.1, 0.0, 0.0], [0.0, float("nan"), 0.0]):
            with self.subTest(action=action):
                with self.assertRaises(MetricsProtocolError):
                    env_action_diagnostics(action, context="invalid")

    def test_schema_distinguishes_disabled_and_sparse_zero_components(self):
        schema = reward_schema_document("v2")
        self.assertFalse(schema["reward_components"]["progress"]["enabled"])
        self.assertFalse(schema["reward_components"]["alive"]["enabled"])
        self.assertTrue(schema["reward_components"]["collision"]["sparse"])
        self.assertTrue(schema["reward_components"]["safety_lidar"]["sparse"])
        self.assertEqual(schema["metrics_protocol_version"], METRICS_PROTOCOL_VERSION)

        v3_schema = reward_schema_document("v3")
        self.assertTrue(v3_schema["reward_components"]["progress"]["enabled"])
        self.assertFalse(v3_schema["reward_components"]["overspeed"]["sparse"])
        self.assertEqual(
            v3_schema["reward_protocol"]["reward_protocol_version"],
            "base_route_completion_reward_v3_offroad_v1",
        )

    def test_reward_conservation_and_exact_schema_pass(self):
        details = _reward_details(speed=1.0, step=-0.1, collision=-2.0)
        rewards, normalized, diagnostics, residuals = validate_active_reward_payload(
            active_agent_ids=[0],
            rewards={0: _reward(details)},
            reward_details_by_agent={0: details},
            reward_diagnostics_by_agent={0: _reward_diagnostics(collision_event=1.0)},
            reward_version="v2",
            context="unit",
        )
        self.assertEqual(rewards[0], -1.1)
        self.assertEqual(normalized[0]["collision"], -2.0)
        self.assertEqual(diagnostics[0]["collision_event"], 1.0)
        self.assertLess(abs(residuals[0]), 1e-12)

    def test_reward_v3_payload_uses_exact_versioned_schema(self):
        details = {name: 0.0 for name in reward_components("v3")}
        details["progress"] = 0.25
        diagnostics = {name: 0.0 for name in reward_diagnostics("v3")}
        rewards, normalized, normalized_diagnostics, residuals = (
            validate_active_reward_payload(
                active_agent_ids=[0],
                rewards={0: 0.25},
                reward_details_by_agent={0: details},
                reward_diagnostics_by_agent={0: diagnostics},
                reward_version="v3",
                context="reward-v3-unit",
            )
        )
        self.assertEqual(rewards[0], 0.25)
        self.assertEqual(normalized[0], details)
        self.assertEqual(normalized_diagnostics[0], diagnostics)
        self.assertLess(abs(residuals[0]), 1e-12)

    def test_missing_unknown_nonfinite_and_bad_sum_fail_fast(self):
        good_details = _reward_details(speed=1.0)
        good_diagnostics = _reward_diagnostics()

        cases = []
        missing = dict(good_details)
        missing.pop("speed")
        cases.append(({0: missing}, {0: good_diagnostics}, {0: 1.0}, "missing"))
        unknown = dict(good_details, mystery=1.0)
        cases.append(({0: unknown}, {0: good_diagnostics}, {0: 2.0}, "unknown"))
        nonfinite = dict(good_details, speed=float("nan"))
        cases.append(({0: nonfinite}, {0: good_diagnostics}, {0: 1.0}, "NaN/Inf"))
        cases.append(({0: good_details}, {0: good_diagnostics}, {0: 2.0}, "conservation"))

        for details, diagnostics, rewards, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(MetricsProtocolError, message):
                    validate_active_reward_payload(
                        active_agent_ids=[0],
                        rewards=rewards,
                        reward_details_by_agent=details,
                        reward_diagnostics_by_agent=diagnostics,
                        reward_version="v2",
                        context="unit",
                    )


class EnvironmentRewardDiagnosticsTests(unittest.TestCase):
    def test_v2_diagnostics_explain_disabled_and_lidar_components(self):
        class FakeLocation:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

            def distance(self, other):
                return math.sqrt(
                    (self.x - other.x) ** 2
                    + (self.y - other.y) ** 2
                    + (self.z - other.z) ** 2
                )

        location = FakeLocation(9.0, 0.0)
        transform = SimpleNamespace(
            location=location,
            rotation=SimpleNamespace(yaw=0.0),
        )
        velocity = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        control = SimpleNamespace(throttle=0.5, steer=0.0, brake=0.0)

        class FakeVehicle:
            is_alive = True

            @staticmethod
            def get_transform():
                return transform

            @staticmethod
            def get_velocity():
                return velocity

            @staticmethod
            def get_control():
                return control

        waypoint_transform = SimpleNamespace(
            location=FakeLocation(9.0, 0.0),
            get_right_vector=lambda: SimpleNamespace(x=0.0, y=1.0),
        )
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.vehicles = [FakeVehicle()]
        env.reached_target = {0: False}
        env.reward_gamma = 0.997
        env.timestep = 0.1
        env.max_steps = 1000
        env.current_step = 10
        env.prev_distances = {0: 2.0}
        env.initial_distances = {0: 20.0}
        env.prev_controls = {}
        env.collision_events = {0: False}
        env.crosswalk_invasion_events = {0: False}
        env.lane_invasion_events = {0: False}
        env.lane_invasion_details = {}
        env.goal_zone = {}
        env.adversarial_agent_ids = []
        env.map = SimpleNamespace(
            get_waypoint=lambda _location: SimpleNamespace(
                transform=waypoint_transform
            )
        )
        env._get_target_location = lambda _agent_id: FakeLocation(10.0, 0.0)
        env._is_in_goal_zone = lambda _location: False
        env._get_ray_detection = lambda *_args, **_kwargs: np.asarray(
            [0.25] + [1.0] * 10, dtype=np.float32
        )
        env._get_relative_position_to_agents = lambda _agent_id: {}

        rewards, details = env._compute_rewards_v2()
        diagnostics = env.last_reward_diagnostics[0]

        self.assertAlmostEqual(
            rewards[0], math.fsum(details[0].values()), delta=1e-6
        )
        self.assertEqual(details[0]["progress"], 0.0)
        self.assertEqual(details[0]["alive"], 0.0)
        self.assertAlmostEqual(diagnostics["progress_m"], 1.0)
        self.assertAlmostEqual(diagnostics["closing_speed_mps"], 1.0)
        self.assertAlmostEqual(diagnostics["lidar_min_normalized"], 0.25)
        self.assertEqual(diagnostics["lidar_trigger_ray_count"], 1.0)
        self.assertLess(details[0]["safety_lidar"], 0.0)
        self.assertEqual(set(diagnostics), set(reward_diagnostics("v2")))

    def test_v3_environment_uses_bounded_progress_and_symmetric_brake_rate(self):
        class FakeLocation:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

            def distance(self, other):
                return math.sqrt(
                    (self.x - other.x) ** 2
                    + (self.y - other.y) ** 2
                    + (self.z - other.z) ** 2
                )

        location = FakeLocation(9.0, 0.0)
        transform = SimpleNamespace(
            location=location,
            rotation=SimpleNamespace(yaw=0.0),
        )
        velocity = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        control = SimpleNamespace(throttle=0.0, steer=0.0, brake=0.5)

        class FakeVehicle:
            is_alive = True

            @staticmethod
            def get_transform():
                return transform

            @staticmethod
            def get_velocity():
                return velocity

            @staticmethod
            def get_control():
                return control

        waypoint_transform = SimpleNamespace(
            location=FakeLocation(9.0, 0.0),
            get_right_vector=lambda: SimpleNamespace(x=0.0, y=1.0),
        )
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.reward_version = "v3"
        env.vehicles = [FakeVehicle()]
        env.reached_target = {0: False}
        env.reward_gamma = 0.997
        env.timestep = 0.1
        env.max_steps = 1000
        env.current_step = 10
        env.prev_distances = {0: 2.0}
        env.initial_distances = {0: 20.0}
        env.prev_controls = {
            0: {"throttle": 0.5, "steer": 0.0, "brake": 0.0}
        }
        env.collision_events = {0: False}
        env.crosswalk_invasion_events = {0: False}
        env.lane_invasion_events = {0: False}
        env.lane_invasion_details = {}
        env.goal_zone = {}
        env.adversarial_agent_ids = []
        env.map = SimpleNamespace(
            get_waypoint=lambda _location: SimpleNamespace(
                transform=waypoint_transform
            )
        )
        env._get_target_location = lambda _agent_id: FakeLocation(10.0, 0.0)
        env._is_in_goal_zone = lambda _location: False
        env._get_ray_detection = lambda *_args, **_kwargs: np.asarray(
            [0.25] + [1.0] * 10, dtype=np.float32
        )
        env._get_relative_position_to_agents = lambda _agent_id: {}

        rewards, details = env._compute_rewards()
        diagnostics = env.last_reward_diagnostics[0]

        self.assertAlmostEqual(
            rewards[0], math.fsum(details[0].values()), delta=1e-9
        )
        self.assertEqual(set(details[0]), set(reward_components("v3")))
        self.assertEqual(set(diagnostics), set(reward_diagnostics("v3")))
        self.assertGreater(details[0]["progress"], 0.0)
        self.assertEqual(details[0]["longitudinal_rate"], -0.005)
        self.assertEqual(diagnostics["longitudinal_command"], -0.5)
        self.assertEqual(diagnostics["longitudinal_rate_normalized"], 1.0)
        self.assertGreater(diagnostics["lidar_risk"], 0.0)
        self.assertEqual(details[0]["overspeed"], 0.0)

    def test_v3_collision_has_priority_over_goal_success(self):
        class FakeLocation:
            x = 0.0
            y = 0.0
            z = 0.0

            @staticmethod
            def distance(_other):
                return 0.0

        transform = SimpleNamespace(
            location=FakeLocation(),
            rotation=SimpleNamespace(yaw=0.0),
        )
        velocity = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        control = SimpleNamespace(throttle=0.0, steer=0.0, brake=0.0)

        class FakeVehicle:
            is_alive = True

            get_transform = staticmethod(lambda: transform)
            get_velocity = staticmethod(lambda: velocity)
            get_control = staticmethod(lambda: control)

        waypoint_transform = SimpleNamespace(
            location=FakeLocation(),
            get_right_vector=lambda: SimpleNamespace(x=0.0, y=1.0),
        )
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.reward_version = "v3"
        env.vehicles = [FakeVehicle()]
        env.reached_target = {0: False}
        env.reward_gamma = 0.997
        env.timestep = 0.1
        env.prev_distances = {0: 1.0}
        env.initial_distances = {0: 10.0}
        env.prev_controls = {}
        env.collision_events = {0: True}
        env.crosswalk_invasion_events = {0: False}
        env.lane_invasion_events = {0: False}
        env.lane_invasion_details = {}
        env.goal_zone = {}
        env.adversarial_agent_ids = []
        env.map = SimpleNamespace(
            get_waypoint=lambda _location: SimpleNamespace(
                transform=waypoint_transform
            )
        )
        env._get_target_location = lambda _agent_id: FakeLocation()
        env._is_in_goal_zone = lambda _location: True
        env._get_ray_detection = lambda *_args, **_kwargs: np.ones(11)
        env._get_relative_position_to_agents = lambda _agent_id: {}

        _, details = env._compute_rewards_v3()
        self.assertEqual(details[0]["collision"], -20.0)
        self.assertEqual(details[0]["success"], 0.0)

        env.num_agents = 1
        env.adversary_collision_ends_episode = True
        env.current_step = 1
        env.max_steps = 100
        env._despawn_reached_vehicle = lambda _agent_id: self.fail(
            "collision-priority outcome must not despawn as success"
        )
        dones = env._check_dones()
        self.assertEqual(dones, {0: True})
        self.assertEqual(env.episode_end_reason, "cooperative_collision")
        self.assertEqual(env.reached_target, {0: False})
        self.assertEqual(env.last_terminated, {0: True})
        self.assertEqual(env.last_truncated, {0: False})

        # During the 3-tick debounce window, a raw offroad footprint inside
        # the goal must wait rather than being registered as success.
        env.collision_events[0] = False
        env.offroad_terminal_events = {0: False}
        env.last_road_occupancy = {0: {"offroad_raw": True}}
        _, raw_offroad_details = env._compute_rewards_v3()
        self.assertEqual(raw_offroad_details[0]["success"], 0.0)
        self.assertEqual(raw_offroad_details[0]["offroad"], 0.0)
        self.assertEqual(env._check_dones(), {0: False})
        self.assertEqual(env.episode_end_reason, "running")

        env.offroad_terminal_events[0] = True
        self.assertEqual(env._check_dones(), {0: True})
        self.assertEqual(env.episode_end_reason, "cooperative_offroad")
        self.assertFalse(env.reached_target[0])

    def test_v4_environment_emits_exact_mission_schema_and_timeout_failure(self):
        class FakeLocation:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

            def distance(self, other):
                return math.sqrt(
                    (self.x - other.x) ** 2
                    + (self.y - other.y) ** 2
                    + (self.z - other.z) ** 2
                )

        location = FakeLocation(1.0, 0.0)
        transform = SimpleNamespace(
            location=location,
            rotation=SimpleNamespace(yaw=0.0),
        )
        velocity = SimpleNamespace(x=0.5, y=0.0, z=0.0)
        control = SimpleNamespace(throttle=0.2, steer=0.0, brake=0.0)

        class FakeVehicle:
            is_alive = True

            get_transform = staticmethod(lambda: transform)
            get_velocity = staticmethod(lambda: velocity)
            get_control = staticmethod(lambda: control)

        waypoint_transform = SimpleNamespace(
            location=FakeLocation(1.0, 0.0),
            get_right_vector=lambda: SimpleNamespace(x=0.0, y=1.0),
        )
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.reward_version = "v4"
        env.vehicles = [FakeVehicle()]
        env.reached_target = {0: False}
        env.reward_gamma = 0.999
        env.timestep = 0.1
        env.current_step = 500
        env.max_steps = 500
        env.prev_distances = {0: 10.0}
        env.initial_distances = {0: 10.0}
        env.prev_controls = {}
        env.previous_lidar_distances = {0: np.ones(11, dtype=np.float32)}
        env.legal_lane_change_grace_ticks = {0: 0}
        env.stall_states = {}
        env.current_scenario_family = "free_flow"
        env.lidar_range_m = 20.0
        env.ttc_max_range_m = 20.0
        env.collision_events = {0: False}
        env.crosswalk_invasion_events = {0: False}
        env.lane_invasion_events = {0: False}
        env.lane_invasion_details = {}
        env.offroad_terminal_events = {0: False}
        env.offroad_consecutive_ticks = {0: 0}
        env.last_road_occupancy = {
            0: {
                "offroad_raw": False,
                "driving_center_valid": True,
                "footprint_outside_count": 0,
                "footprint_outside_fraction": 0.0,
            }
        }
        env.goal_zone = {}
        env.agent_goal_zones = {}
        env.adversarial_agent_ids = []
        env.map = SimpleNamespace(
            name="Carla/Maps/Town05",
            get_waypoint=lambda _location: SimpleNamespace(
                transform=waypoint_transform
            ),
        )
        env._get_target_location = lambda _agent_id: FakeLocation(10.0, 0.0)
        env._is_in_goal_zone = lambda _location: False
        env._get_ray_detection = lambda *_args, **_kwargs: np.ones(11)
        env._get_relative_position_to_agents = lambda _agent_id: {}

        rewards, details = env._compute_rewards_v4()
        diagnostics = env.last_reward_diagnostics[0]

        self.assertAlmostEqual(rewards[0], math.fsum(details[0].values()))
        self.assertEqual(set(details[0]), set(reward_components("v4")))
        self.assertEqual(set(diagnostics), set(reward_diagnostics("v4")))
        self.assertEqual(details[0]["mission_timeout"], -19.0)
        self.assertEqual(diagnostics["remaining_time_fraction"], 0.0)
        self.assertEqual(diagnostics["mission_timeout_event"], 1.0)
        self.assertEqual(details[0]["lane_deviation"], 0.0)
        self.assertEqual(details[0]["safety_lidar"], 0.0)


class TrainingMetricsRecorderTests(unittest.TestCase):
    def _new_recorder(self, run_dir, **overrides):
        kwargs = {
            "run_dir": run_dir,
            "run_id": "metrics_s42",
            "agent_ids": [0, 1],
            "reward_version": "v2",
            "start_update": 1,
            "completed_update": 0,
            "histogram_interval": 0,
        }
        kwargs.update(overrides)
        return TrainingMetricsRecorder(**kwargs)

    def _record_ragged_episode(self, recorder):
        details_0 = _reward_details(speed=1.0, step=-0.1)
        details_1_a = _reward_details(speed=2.0, step=-0.1, collision=-4.0)
        recorder.record_env_step(
            active_agent_ids=[0, 1],
            rewards={0: _reward(details_0), 1: _reward(details_1_a)},
            reward_details={0: details_0, 1: details_1_a},
            reward_diagnostics_by_agent={
                0: _reward_diagnostics(speed_mps=1.0),
                1: _reward_diagnostics(speed_mps=2.0, collision_event=1.0),
            },
            policy_actions={0: [0.525, 0.0], 1: [1.0, -1.0]},
            environment_actions={
                0: action_tanh_to_env([0.525, 0.0]),
                1: action_tanh_to_env([1.0, -1.0]),
            },
            context="episode step=0",
        )
        details_1_b = _reward_details(speed=3.0, step=-0.1)
        recorder.record_env_step(
            active_agent_ids=[1],
            rewards={1: _reward(details_1_b)},
            reward_details={1: details_1_b},
            reward_diagnostics_by_agent={1: _reward_diagnostics(speed_mps=3.0)},
            policy_actions={1: [0.43, 0.2]},
            environment_actions={1: action_tanh_to_env([0.43, 0.2])},
            context="episode step=1",
        )
        recorder.finish_episode(
            update=1,
            episode_index=0,
            episode_returns={
                0: _reward(details_0),
                1: _reward(details_1_a) + _reward(details_1_b),
            },
            episode_length=2,
            active_decision_steps={0: 1, 1: 2},
            transition_counts={0: 1, 1: 2},
            dones={0: True, 1: True},
            terminated={0: True, 1: True},
            truncated={0: False, 1: False},
            bootstrap_enabled={0: False, 1: False},
            success=False,
            collision=True,
            timeout=False,
            end_reason="cooperative_collision",
            scenario_name="scenario_close.json",
            context="episode",
        )

    def _record_model_and_skipped_ppo(self, recorder):
        actors = {0: nn.Linear(2, 2), 1: nn.Linear(2, 2)}
        critic = nn.Linear(3, 1)
        recorder.capture_model_snapshot(actors, critic)
        with torch.no_grad():
            actors[0].weight.add_(0.1)
            critic.bias.add_(0.2)
        recorder.record_model_update(actors, critic)
        recorder.record_ppo_update(
            ppo_metrics=None,
            advantage_stats={},
            preupdate_ratio_errors={},
            agent_stop_epoch={},
            transition_counts={0: 1, 1: 2},
            actor_lrs={0: 3e-4, 1: 3e-4},
            critic_lr=3e-4,
            skipped=True,
            max_grad_norm=0.5,
        )

    def test_policy_and_environment_action_mapping_mismatch_fails_fast(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            details = _reward_details(speed=1.0)
            with self.assertRaisesRegex(MetricsProtocolError, "mapping mismatch"):
                recorder.record_env_step(
                    active_agent_ids=[0],
                    rewards={0: _reward(details)},
                    reward_details={0: details},
                    reward_diagnostics_by_agent={0: _reward_diagnostics()},
                    policy_actions={0: [0.5, 0.0]},
                    environment_actions={0: [0.5, 0.0, 0.0]},
                    context="mapping audit",
                )
            recorder.close()

    def test_agent_isolation_sparse_events_and_aggregation_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            self._record_ragged_episode(recorder)
            self._record_model_and_skipped_ppo(recorder)
            payload = recorder.finalize_update(1)
            scalars = payload["scalars"]

            self.assertAlmostEqual(
                scalars["reward/agent_0/speed/mean_per_active_step"], 1.0
            )
            self.assertAlmostEqual(
                scalars["reward/agent_1/speed/mean_per_active_step"], 2.5
            )
            self.assertAlmostEqual(
                scalars["reward/pooled/speed/mean_per_active_step"], 2.0
            )
            self.assertEqual(
                scalars["events/agent_1/collision/nonzero_count_per_update"], 1.0
            )
            self.assertEqual(
                scalars["rollout/agent_0/active_transition_count"], 1.0
            )
            self.assertEqual(
                scalars["rollout/agent_1/active_transition_count"], 2.0
            )
            self.assertEqual(scalars["reward_status/progress/enabled"], 0.0)
            self.assertEqual(
                scalars["action/agent_0/brake_zero_rate"], 1.0
            )
            self.assertEqual(
                scalars["action/agent_0/brake_saturation_rate"], 0.0
            )
            self.assertEqual(
                scalars["action/agent_1/throttle_saturation_rate"], 0.5
            )
            self.assertEqual(
                scalars["action/agent_1/steer_saturation_rate"], 0.5
            )
            self.assertEqual(
                scalars["action/agent_1/throttle_mode_rate"], 1.0
            )
            self.assertEqual(
                scalars["action/agent_1/longitudinal_conflict_rate"], 0.0
            )
            self.assertLess(
                scalars["reward_audit/max_abs_episode_residual"], 1e-12
            )
            self.assertGreater(
                scalars["model/actor_agent_0/update_norm"], 0.0
            )
            recorder.close()

    def test_tensorboard_and_jsonl_receive_the_same_payload(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            self._record_ragged_episode(recorder)
            self._record_model_and_skipped_ppo(recorder)
            payload = recorder.finalize_update(1)
            recorder.commit_payload(payload)
            tb_dir = recorder.tb_dir
            recorder.close()

            records = [
                json.loads(line)
                for line in (Path(temporary_dir) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["scalars"], payload["scalars"])

            episode_records = [
                json.loads(line)
                for line in (Path(temporary_dir) / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(episode_records), 1)
            self.assertEqual(episode_records[0]["update"], 1)
            self.assertEqual(episode_records[0]["episode_index"], 0)
            self.assertEqual(
                episode_records[0]["agents"]["0"]["active_decision_steps"], 1
            )
            self.assertEqual(
                episode_records[0]["agents"]["1"]["transition_count"], 2
            )
            self.assertFalse(
                episode_records[0]["agents"]["1"]["bootstrap_enabled"]
            )

            event_file = next(tb_dir.glob("events.out.tfevents.*"))
            accumulator = EventAccumulator(
                str(event_file), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
            tag = "reward/agent_1/speed/mean_per_active_step"
            self.assertIn(tag, accumulator.Tags()["scalars"])
            event = accumulator.Scalars(tag)[-1]
            self.assertEqual(event.step, 1)
            self.assertAlmostEqual(event.value, payload["scalars"][tag])

            report = validate_metrics_run(temporary_dir)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["updates"], [1])
            self.assertEqual(report["episode_record_count"], 1)

    def test_validator_rejects_tampered_longitudinal_action_modes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            self._record_ragged_episode(recorder)
            self._record_model_and_skipped_ppo(recorder)
            payload = recorder.finalize_update(1)
            recorder.commit_payload(payload)
            recorder.close()

            metrics_path = Path(temporary_dir) / "metrics.jsonl"
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            scalars = record["scalars"]
            scalars["action/agent_0/throttle_mode_rate"] = 0.0
            scalars["action/agent_0/longitudinal_conflict_rate"] = 1.0
            metrics_path.write_text(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MetricsProtocolError, "mutually exclusive"):
                validate_metrics_run(temporary_dir)

    def test_checkpoint_payload_replays_missing_metrics_after_crash(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = self._new_recorder(temporary_dir)
            self._record_ragged_episode(first)
            self._record_model_and_skipped_ppo(first)
            payload = first.finalize_update(1)
            # Simulate: checkpoint containing payload was saved, then process
            # died before TensorBoard/JSONL commit.
            first.close()

            resumed = self._new_recorder(
                temporary_dir,
                start_update=2,
                completed_update=1,
                resume_metrics_payload=payload,
            )
            resumed.close()
            records = [
                json.loads(line)
                for line in (Path(temporary_dir) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual([record["update"] for record in records], [1])
            self.assertEqual(records[0]["scalars"], payload["scalars"])

            mismatched_payload = dict(payload)
            mismatched_payload["scalars"] = dict(payload["scalars"])
            mismatched_payload["scalars"]["episode/count"] = 999.0
            with self.assertRaisesRegex(
                MetricsProtocolError, "differs from safe checkpoint"
            ):
                self._new_recorder(
                    temporary_dir,
                    start_update=2,
                    completed_update=1,
                    resume_metrics_payload=mismatched_payload,
                )

    def test_episode_return_or_active_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            details = _reward_details(speed=1.0)
            recorder.record_env_step(
                active_agent_ids=[0, 1],
                rewards={0: 1.0, 1: 1.0},
                reward_details={0: details, 1: details},
                reward_diagnostics_by_agent={
                    0: _reward_diagnostics(),
                    1: _reward_diagnostics(),
                },
                policy_actions={0: [0.0, 0.0], 1: [0.0, 0.0]},
                environment_actions={
                    0: action_tanh_to_env([0.0, 0.0]),
                    1: action_tanh_to_env([0.0, 0.0]),
                },
                context="step",
            )
            with self.assertRaisesRegex(MetricsProtocolError, "active-step count"):
                recorder.finish_episode(
                    update=1,
                    episode_index=0,
                    episode_returns={0: 1.0, 1: 1.0},
                    episode_length=1,
                    active_decision_steps={0: 0, 1: 1},
                    transition_counts={0: 0, 1: 1},
                    dones={0: True, 1: True},
                    terminated={0: True, 1: True},
                    truncated={0: False, 1: False},
                    bootstrap_enabled={0: False, 1: False},
                    success=False,
                    collision=False,
                    timeout=True,
                    end_reason="max_steps",
                    scenario_name="scenario.json",
                    context="episode",
                )
            recorder.close()

    def test_episode_lifecycle_record_rejects_incorrect_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            details = _reward_details(speed=1.0)
            recorder.record_env_step(
                active_agent_ids=[0, 1],
                rewards={0: 1.0, 1: 1.0},
                reward_details={0: details, 1: details},
                reward_diagnostics_by_agent={
                    0: _reward_diagnostics(),
                    1: _reward_diagnostics(),
                },
                policy_actions={0: [0.0, 0.0], 1: [0.0, 0.0]},
                environment_actions={
                    0: action_tanh_to_env([0.0, 0.0]),
                    1: action_tanh_to_env([0.0, 0.0]),
                },
                context="step",
            )
            with self.assertRaisesRegex(
                MetricsProtocolError, "bootstrap_enabled must equal truncated"
            ):
                recorder.finish_episode(
                    update=1,
                    episode_index=0,
                    episode_returns={0: 1.0, 1: 1.0},
                    episode_length=1,
                    active_decision_steps={0: 1, 1: 1},
                    transition_counts={0: 1, 1: 1},
                    dones={0: True, 1: True},
                    terminated={0: False, 1: False},
                    truncated={0: True, 1: True},
                    bootstrap_enabled={0: False, 1: True},
                    success=False,
                    collision=False,
                    timeout=True,
                    end_reason="max_steps",
                    scenario_name="scenario.json",
                    context="episode",
                )
            recorder.close()

    def test_resume_discards_episode_records_ahead_of_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = self._new_recorder(temporary_dir)
            self._record_ragged_episode(first)
            first.close()
            self.assertTrue(
                (Path(temporary_dir) / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

            resumed = self._new_recorder(
                temporary_dir,
                start_update=1,
                completed_update=0,
            )
            resumed.close()
            self.assertEqual(
                (Path(temporary_dir) / "episodes.jsonl").read_text(
                    encoding="utf-8"
                ),
                "",
            )

    def test_non_skipped_ppo_health_metrics_are_namespaced(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = self._new_recorder(temporary_dir)
            self._record_ragged_episode(recorder)
            actors = {0: nn.Linear(2, 2), 1: nn.Linear(2, 2)}
            critic = nn.Linear(3, 1)
            recorder.capture_model_snapshot(actors, critic)
            with torch.no_grad():
                actors[0].weight.add_(0.01)
                actors[1].weight.sub_(0.02)
                critic.bias.add_(0.03)
            recorder.record_model_update(actors, critic)
            ppo_metrics = {
                "critic_loss": [2.0],
                "grad_norm_critic": [0.7],
                "critic_samples_seen": 3,
                "explained_variance": 0.4,
                "value_prediction_mean": 1.0,
                "value_prediction_std": 0.5,
                "return_target_mean": 1.2,
                "return_target_std": 0.6,
                "value_error_abs_mean": 0.3,
                "normalized_value_prediction_mean": 0.0,
                "normalized_value_prediction_std": 0.8,
                "normalized_return_target_mean": 0.0,
                "normalized_return_target_std": 1.0,
                "value_normalizer_mean": 1.2,
                "value_normalizer_std": 0.6,
                "value_normalizer_variance": 0.36,
                "value_normalizer_debiasing_term": 0.001,
                "value_normalizer_update_count": 1.0,
            }
            for aid in (0, 1):
                ppo_metrics.update(
                    {
                        f"actor_loss_{aid}": [0.2],
                        f"surrogate_loss_{aid}": [0.3],
                        f"entropy_{aid}": [1.5],
                        f"entropy_bonus_{aid}": [0.045],
                        f"approx_kl_{aid}": [0.01],
                        f"clip_frac_{aid}": [0.2],
                        f"ratio_mean_{aid}": [1.01],
                        f"ratio_std_{aid}": [0.1],
                        f"ratio_min_{aid}": [0.85, 0.8],
                        f"ratio_max_{aid}": [1.2, 1.25],
                        f"grad_norm_actor_{aid}": [0.6],
                        f"actor_samples_seen_{aid}": 3,
                    }
                )
            recorder.record_ppo_update(
                ppo_metrics=ppo_metrics,
                advantage_stats={
                    0: {"raw_mean": 1.0, "raw_std": 2.0},
                    1: {"raw_mean": 3.0, "raw_std": 4.0},
                },
                preupdate_ratio_errors={0: 0.0, 1: 0.0},
                agent_stop_epoch={0: 2, 1: 3},
                transition_counts={0: 1, 1: 2},
                actor_lrs={0: 3e-4, 1: 3e-4},
                critic_lr=3e-4,
                skipped=False,
                max_grad_norm=0.5,
            )
            payload = recorder.finalize_update(1)
            scalars = payload["scalars"]
            self.assertAlmostEqual(scalars["policy/agent_0/ratio_mean"], 1.01)
            self.assertAlmostEqual(scalars["policy/agent_0/ratio_min"], 0.8)
            self.assertAlmostEqual(scalars["policy/agent_0/ratio_max"], 1.25)
            self.assertAlmostEqual(scalars["policy/agent_1/entropy_bonus"], 0.045)
            self.assertEqual(
                scalars["policy/agent_0/grad_clip_activation_rate"], 1.0
            )
            self.assertEqual(scalars["critic/grad_clip_activation_rate"], 1.0)
            recorder.commit_payload(payload)
            recorder.close()
            report = validate_metrics_run(
                temporary_dir,
                expected_updates=[1],
                expected_episodes_per_update=1,
                require_ppo_updates=True,
                max_preupdate_ratio_error=1e-6,
            )
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
