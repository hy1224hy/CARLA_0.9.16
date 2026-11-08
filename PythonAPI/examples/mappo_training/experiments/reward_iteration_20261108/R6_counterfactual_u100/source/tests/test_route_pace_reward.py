"""Offline behavioral checks; never creates a CARLA client."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

from environment.base_reward_v6 import free_flow_pace_deficit
from environment.route_pace_reward import route_pace_deficit
from tests.test_base_reward_v6 import RewardV6IntegrationTests
from training.checkpoint_protocol import CheckpointProtocolError
from utils.training_metrics_recorder import TrainingMetricsRecorder
from training.train_no_rllib_mappo_robust_v2 import (
    apply_resume_configuration, build_parser, build_training_environment_kwargs,
    checkpoint_training_config,
)


class RoutePaceTests(unittest.TestCase):
    def test_waiting_cost_and_on_schedule_neutrality(self):
        for distance in (15.0, 40.0):
            cost = route_pace_deficit(1.0, initial_distance_m=distance,
                                     free_seconds=2.0, timestep=0.1)
            self.assertAlmostEqual(cost.reward, -0.03)
            on_time = route_pace_deficit(1 - 5 / distance,
                initial_distance_m=distance, free_seconds=2.0, timestep=0.1)
            self.assertEqual(on_time.reward, 0)
        old = free_flow_pace_deficit(1.0, 0.8, current_step=40,
                                    free_flow=True, non_interactive=True)
        self.assertLess(cost.reward, old.reward)
        long_wait = route_pace_deficit(1.0, initial_distance_m=40,
                                      free_seconds=100, timestep=0.1)
        self.assertEqual(long_wait.reward, -0.05)
        self.assertEqual(route_pace_deficit(1, initial_distance_m=40,
            free_seconds=100, timestep=0.1, terminal_event=True).reward, 0)

    def environment(self):
        env = RewardV6IntegrationTests()._environment()
        env.pace_profile = "route_pace_v1"
        env._route_pace_clocks = {}
        env.reached_target = {}
        env.vehicles = [SimpleNamespace(is_alive=True) for _ in range(2)]
        env.initial_distances = {0: 40.0, 1: 40.0}
        env.timestep = 0.1
        return env

    def test_all_twelve_scenarios_receive_pace_cost(self):
        pool = Path(__file__).resolve().parents[1] / "data/training_scenarios_town05_r8a_v1"
        paths = list(pool.glob("scenario_*.json"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            scenario = json.loads(path.read_text(encoding="utf-8"))
            env = self.environment()
            env.current_scenario_tags = scenario["tags"]
            env.current_scenario_family = scenario["family"]
            rewards, details = env._compute_rewards_v6()
            self.assertLess(rewards[0], 0, path.name)
            self.assertEqual(rewards[1], 0, "blocked car must be exempt")
            self.assertAlmostEqual(rewards[0], sum(details[0].values()))

    def test_blocker_pauses_clock_and_finished_car_has_no_cost(self):
        env = self.environment()
        env._compute_rewards_v6()
        self.assertEqual(env._route_pace_clocks[1][1], 0)
        env.current_step = 101
        env._pace_has_front_blocker = lambda _: False
        rewards, _ = env._compute_rewards_v6()
        self.assertAlmostEqual(env._route_pace_clocks[1][1], 0.1)
        self.assertEqual(rewards[1], 0)
        env.reached_target[0] = True
        env.current_step = 102
        rewards, _ = env._compute_rewards_v6()
        self.assertEqual(rewards[0], 0)
        env.current_scenario_family = "required_merge"
        env.current_scenario_tags = []
        self.assertEqual(env._compute_rewards_v6()[0][1], 0)

    def test_profile_reaches_environment_and_cannot_change_on_resume(self):
        args = build_parser().parse_args([
            "--reward-version", "v6", "--pace-profile", "route_pace_v1"])
        self.assertEqual(build_training_environment_kwargs(args, num_agents=2)["pace_profile"],
                         "route_pace_v1")
        self.assertEqual(checkpoint_training_config(args)["pace_profile"], "route_pace_v1")
        args._explicit_cli_options = {"--pace-profile"}
        with self.assertRaises(CheckpointProtocolError):
            apply_resume_configuration(args, {"training_config": {}})
        config = checkpoint_training_config(args)
        restored = build_parser().parse_args(["--reward-version", "v6"])
        apply_resume_configuration(restored, {
            "training_config": config,
            "run_metadata": {"seed": 42, "run_id": "pace_test"},
            "lr_schedule": {"decay_horizon": 600},
        })
        self.assertEqual(restored.pace_profile, "route_pace_v1")
        self.assertEqual(checkpoint_training_config(restored), config)

    def test_log_schema_records_actual_reward_instead_of_old_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrainingMetricsRecorder(run_dir=directory, run_id="pace_test",
                agent_ids=[0, 1], reward_version="v6", start_update=1,
                completed_update=0, pace_profile="route_pace_v1")
            try:
                schema = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                self.assertEqual(schema["reward_protocol"]["pace_intervention"]["profile"],
                                 "route_pace_v1")
                self.assertIn("metres", schema["reward_diagnostics"]["pace_deficit_raw"]["value_semantics"])
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
