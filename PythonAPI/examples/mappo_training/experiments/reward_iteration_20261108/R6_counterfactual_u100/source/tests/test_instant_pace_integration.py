"""Mock integration of candidate speed costs; never creates a CARLA client."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from environment.base_reward_v5 import (
    REWARD_COMPONENTS as V5_COMPONENTS,
    REWARD_DIAGNOSTICS as V5_DIAGNOSTICS,
)
from environment.base_reward_v6 import (
    PACE_COMPONENT, REWARD_COMPONENTS as V6_COMPONENTS,
    REWARD_DIAGNOSTICS as V6_DIAGNOSTICS, free_flow_pace_deficit,
)
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.instant_pace_reward import PROFILE, instant_pace_spec
from environment.instant_pace_reward_v3 import instant_pace_spec as instant_pace_v3_spec
from environment.instant_pace_reward_v4 import instant_pace_spec as instant_pace_v4_spec
from environment.instant_pace_reward_v5 import instant_pace_spec as instant_pace_v5_spec
from environment.route_pace_reward import route_pace_spec
from training.checkpoint_protocol import CheckpointProtocolError
from training.metrics_protocol import reward_schema_document
from training.train_no_rllib_mappo_robust_v2 import (
    apply_resume_configuration, build_parser, build_training_environment_kwargs,
    checkpoint_training_config, runtime_metadata_with_carla,
)
from utils.training_metrics_recorder import TrainingMetricsRecorder


PROJECT_DIR = Path(__file__).resolve().parents[1]


class InstantPaceIntegrationTests(unittest.TestCase):
    def environment(self, profile=PROFILE):
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.num_agents = 2
        env.adversarial_agent_ids = []
        env.reward_version = "v6"
        env.reward_gamma = 0.999
        env.pace_profile = profile
        env.current_step = 100
        env.timestep = 0.1
        env.current_scenario_tags = {"free_flow", "non_interactive"}
        env.current_scenario_family = "free_flow"
        env.vehicles = [SimpleNamespace(is_alive=True) for _ in range(2)]
        env.reached_target = {0: False, 1: False}
        env.initial_distances = {0: 40.0, 1: 40.0}
        env._route_pace_clocks = {}
        env._pace_has_front_blocker = lambda _: False
        env.client = SimpleNamespace(
            get_client_version=lambda: "mock-client",
            get_server_version=lambda: "mock-server",
        )
        env.parent_details = {
            aid: {name: 0.0 for name in V5_COMPONENTS}
            for aid in range(2)
        }
        env.parent_diagnostics = {
            aid: {name: 0.0 for name in V5_DIAGNOSTICS}
            for aid in range(2)
        }
        for aid in range(2):
            env.parent_details[aid].update({
                "progress": 0.4 + 0.05 * aid,
                "overspeed": -0.005,
                "longitudinal_rate": -0.001,
                "steer_rate": -0.002,
                "safety_lidar": -0.003,
                "ttc": -0.004,
                "lane_deviation": -0.006,
                "lane_invasion": -0.25,
            })
            env.parent_diagnostics[aid].update({
                "remaining_route_fraction": 0.75,
                "remaining_time_fraction": 0.5,
                "distance_to_goal_m": 30.0,
                "speed_mps": 50.0 / 3.6 if aid == 0 else 2.5,
                "closing_speed_mps": 1.0 if aid == 0 else 2.5,
                "overspeed_risk": 1.0,
            })

        def frozen_v5():
            details = copy.deepcopy(env.parent_details)
            env.last_reward_diagnostics = copy.deepcopy(env.parent_diagnostics)
            return {aid: math.fsum(values.values())
                    for aid, values in details.items()}, details

        env._compute_rewards_v5 = frozen_v5
        return env

    def test_replaces_only_two_components_and_uses_forward_not_total_speed(self):
        env = self.environment()
        original = copy.deepcopy(env.parent_details)
        rewards, details = env._compute_rewards_v6()
        # Lateral/high-speed movement does not clear a low forward speed cost.
        self.assertAlmostEqual(details[0][PACE_COMPONENT], -0.03)
        self.assertAlmostEqual(details[0]["overspeed"], -0.02 * (35 / 15) ** 2)
        self.assertEqual(details[1][PACE_COMPONENT], 0.0)
        self.assertEqual(details[1]["overspeed"], 0.0)
        for aid in range(2):
            self.assertEqual(set(details[aid]), set(V6_COMPONENTS))
            self.assertEqual(set(env.last_reward_diagnostics[aid]), set(V6_DIAGNOSTICS))
            for name in V5_COMPONENTS:
                if name != "overspeed":
                    self.assertEqual(details[aid][name], original[aid][name])
            self.assertAlmostEqual(rewards[aid], math.fsum(details[aid].values()), places=12)
        self.assertEqual(env.parent_details, original)
        self.assertEqual(env.last_reward_diagnostics[0]["pace_deficit_raw"], 1.5)
        self.assertGreater(env.last_reward_diagnostics[0]["overspeed_risk"], 1.0)
        self.assertEqual(env._route_pace_clocks, {}, "instant profile must not accumulate debt")

    def test_old_profiles_keep_their_speed_cost_and_frozen_formulas(self):
        for profile in ("frozen_v6", "route_pace_v1"):
            with self.subTest(profile=profile):
                env = self.environment(profile)
                rewards, details = env._compute_rewards_v6()
                if profile == "frozen_v6":
                    expected = free_flow_pace_deficit(
                        0.75, 0.5, current_step=100,
                        free_flow=True, non_interactive=True,
                    ).reward
                else:
                    # After 2s grace: expected 20m, travelled 10m, deadzone 2m.
                    expected = -0.05
                for aid in range(2):
                    self.assertAlmostEqual(details[aid][PACE_COMPONENT], expected)
                    for name in V5_COMPONENTS:
                        self.assertEqual(details[aid][name], env.parent_details[aid][name])
                    self.assertAlmostEqual(rewards[aid],
                        math.fsum(env.parent_details[aid].values()) + expected, places=12)

    def test_v3_environment_changes_only_speed_cost_at_thirty_kmh(self):
        older = self.environment(PROFILE)
        newer = self.environment("instant_pace_v3")
        for env in (older, newer):
            env.parent_diagnostics[0]["speed_mps"] = 30.0 / 3.6
        old_rewards, old_details = older._compute_rewards_v6()
        new_rewards, new_details = newer._compute_rewards_v6()
        self.assertAlmostEqual(old_details[0]["overspeed"], -0.02)
        self.assertAlmostEqual(new_details[0]["overspeed"], -0.1)
        self.assertAlmostEqual(new_details[0][PACE_COMPONENT], -0.03)
        self.assertAlmostEqual(new_rewards[0] - old_rewards[0], -0.08)
        for aid in range(2):
            for name in V6_COMPONENTS:
                if name != "overspeed":
                    self.assertEqual(new_details[aid][name], old_details[aid][name])
            self.assertAlmostEqual(new_rewards[aid], math.fsum(new_details[aid].values()), places=12)
            # The diagnostic remains squared excess, rather than total cost.
            self.assertEqual(newer.last_reward_diagnostics[aid], older.last_reward_diagnostics[aid])
        self.assertEqual(newer._route_pace_clocks, {})

    def test_v3_runtime_and_schema_identify_new_coefficients_without_changing_v2(self):
        temporary_root = (PROJECT_DIR / "diagnostics").resolve()
        self.assertTrue(temporary_root.is_relative_to(PROJECT_DIR.resolve()))
        specs = {PROFILE: instant_pace_spec(), "instant_pace_v3": instant_pace_v3_spec()}
        schemas = {}
        for profile, expected in specs.items():
            env = self.environment(profile)
            with patch("training.train_no_rllib_mappo_robust_v2.runtime_metadata", return_value={}):
                metadata = runtime_metadata_with_carla(env)
            self.assertEqual(metadata["pace_reward_protocol"], expected)
            with tempfile.TemporaryDirectory(prefix="instant_pace_test_", dir=temporary_root) as directory:
                self.assertEqual(Path(directory).resolve().parent, temporary_root)
                with patch("utils.training_metrics_recorder.SummaryWriter"):
                    recorder = TrainingMetricsRecorder(
                        run_dir=directory, run_id="instant_identity_test", agent_ids=[0, 1],
                        reward_version="v6", start_update=1, completed_update=0,
                        pace_profile=profile,
                    )
                    try:
                        schemas[profile] = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                    finally:
                        recorder.close()
            self.assertEqual(schemas[profile]["pace_profile"], profile)
            self.assertEqual(schemas[profile]["reward_protocol"]["pace_intervention"], expected)
        v3_spec = schemas["instant_pace_v3"]["reward_protocol"]["pace_intervention"]
        self.assertEqual(v3_spec["parameters"]["overspeed_linear_cost_per_second"], 0.5)
        self.assertEqual(v3_spec["parameters"]["overspeed_quadratic_cost_per_second"], 0.5)
        self.assertNotIn("overspeed_cost_per_second", v3_spec["parameters"])
        self.assertEqual(schemas[PROFILE]["reward_protocol"]["pace_intervention"], instant_pace_spec())
        self.assertEqual(schemas[PROFILE]["reward_protocol"]["pace_intervention"]["parameters"]["overspeed_cost_per_second"], 0.2)
        self.assertEqual(schemas[PROFILE]["reward_diagnostics"], schemas["instant_pace_v3"]["reward_diagnostics"])

    def test_v4_unfinished_goal_approach_charges_stop_but_terminal_stays_exempt(self):
        older = self.environment("instant_pace_v3")
        newer = self.environment("instant_pace_v4")
        for env in (older, newer):
            env.parent_diagnostics[0].update({
                "distance_to_goal_m": 3.18, "closing_speed_mps": 0.0,
                "speed_mps": 0.0,
            })
            env.parent_details[0]["progress"] = 0.0
        old_rewards, old_details = older._compute_rewards_v6()
        new_rewards, new_details = newer._compute_rewards_v6()
        self.assertEqual(old_details[0][PACE_COMPONENT], 0.0)
        self.assertEqual(older.last_reward_diagnostics[0]["pace_near_goal_exempt"], 1.0)
        self.assertAlmostEqual(new_details[0][PACE_COMPONENT], -0.05)
        self.assertEqual(newer.last_reward_diagnostics[0]["pace_near_goal_exempt"], 0.0)
        self.assertEqual(newer.last_reward_diagnostics[0]["pace_deficit_raw"], 0.5)
        self.assertEqual(newer.last_reward_diagnostics[0]["pace_deficit_eligible"], 1.0)
        self.assertAlmostEqual(new_rewards[0] - old_rewards[0], -0.05)
        for name in V5_COMPONENTS:
            self.assertEqual(new_details[0][name], old_details[0][name])
        newer.parent_diagnostics[0]["success_event"] = 1.0
        newer.parent_details[0]["success"] = 20.0
        rewards, details = newer._compute_rewards_v6()
        self.assertEqual(details[0][PACE_COMPONENT], 0.0)
        self.assertEqual(details[0]["success"], 20.0)
        self.assertEqual(newer.last_reward_diagnostics[0]["pace_terminal_exempt"], 1.0)
        self.assertEqual(newer.last_reward_diagnostics[0]["pace_near_goal_exempt"], 0.0)
        self.assertAlmostEqual(rewards[0], math.fsum(details[0].values()), places=12)

    def test_v4_runtime_and_schema_record_slow_approach_reference_and_default_is_frozen(self):
        expected = instant_pace_v4_spec()
        newer = self.environment("instant_pace_v4")
        default_profile = build_parser().parse_args([]).pace_profile
        self.assertEqual(default_profile, "frozen_v6")
        default_env = self.environment(default_profile)
        with patch("training.train_no_rllib_mappo_robust_v2.runtime_metadata", side_effect=lambda: {}):
            self.assertEqual(runtime_metadata_with_carla(newer)["pace_reward_protocol"], expected)
            self.assertNotIn("pace_reward_protocol", runtime_metadata_with_carla(default_env))
        temporary_root = (PROJECT_DIR / "diagnostics").resolve()
        self.assertTrue(temporary_root.is_relative_to(PROJECT_DIR.resolve()))
        with tempfile.TemporaryDirectory(prefix="instant_pace_test_", dir=temporary_root) as directory:
            self.assertEqual(Path(directory).resolve().parent, temporary_root)
            with patch("utils.training_metrics_recorder.SummaryWriter"):
                recorder = TrainingMetricsRecorder(
                    run_dir=directory, run_id="v4_identity_test", agent_ids=[0, 1],
                    reward_version="v6", start_update=1, completed_update=0,
                    pace_profile="instant_pace_v4",
                )
                try:
                    schema = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                finally:
                    recorder.close()
        self.assertEqual(schema["pace_profile"], "instant_pace_v4")
        self.assertEqual(schema["reward_protocol"]["pace_intervention"], expected)
        self.assertEqual(expected["parameters"]["approach_reference_speed_mps"], 0.5)
        self.assertEqual(expected["parameters"]["overspeed_linear_cost_per_second"], 0.5)
        self.assertEqual(expected["parameters"]["overspeed_quadratic_cost_per_second"], 0.5)
        self.assertFalse(expected["pace"]["near_goal_exemption"])
        self.assertIn("m/s", schema["reward_diagnostics"]["pace_deficit_raw"]["value_semantics"])

    def test_v5_lane_feedback_grows_and_preserves_only_valid_grace(self):
        previous_cost = None
        for lateral in (1.65, 2.5):
            with self.subTest(lateral=lateral):
                older = self.environment("instant_pace_v4")
                newer = self.environment("instant_pace_v5")
                for env in (older, newer):
                    env.parent_diagnostics[0]["lateral_distance_m"] = lateral
                    env.parent_diagnostics[0]["road_current_to_reference_cosine"] = 1.0
                old_rewards, old_details = older._compute_rewards_v6()
                new_rewards, new_details = newer._compute_rewards_v6()
                normalized = (lateral - 0.35) / 0.9
                expected = -0.01 * (normalized + normalized ** 2)
                self.assertAlmostEqual(new_details[0]["lane_deviation"], expected)
                self.assertGreater(newer.last_reward_diagnostics[0]["lane_deviation_risk"], 1.0)
                for name in V6_COMPONENTS:
                    if name != "lane_deviation":
                        self.assertEqual(new_details[0][name], old_details[0][name])
                self.assertAlmostEqual(new_rewards[0] - old_rewards[0],
                    expected - old_details[0]["lane_deviation"], places=12)
                self.assertAlmostEqual(new_rewards[0], math.fsum(new_details[0].values()), places=12)
                if previous_cost is not None:
                    self.assertLess(expected, previous_cost)
                previous_cost = expected
        for road, cosine, junction, remains_exempt in (
            (False, 1.0, False, True),
            (True, 1.0, False, False),
            (False, -1.0, False, False),
            (True, -1.0, True, True),
        ):
            with self.subTest(road=road, cosine=cosine, junction=junction):
                newer = self.environment("instant_pace_v5")
                newer.parent_details[0]["lane_deviation"] = 0.0
                newer.parent_diagnostics[0].update({
                    "lateral_distance_m": 2.5,
                    "lane_deviation_exempted": 1.0,
                    "road_violation_raw": float(road),
                    "road_current_to_reference_cosine": cosine,
                    "road_in_junction": float(junction),
                })
                _, details = newer._compute_rewards_v6()
                self.assertEqual(newer.last_reward_diagnostics[0]["lane_deviation_exempted"],
                                 float(remains_exempt))
                if remains_exempt:
                    self.assertEqual(details[0]["lane_deviation"], 0.0)
                else:
                    self.assertLess(details[0]["lane_deviation"], 0.0)

    def test_v5_runtime_and_schema_record_unclipped_reference_lane_replacement(self):
        expected = instant_pace_v5_spec()
        newer = self.environment("instant_pace_v5")
        with patch("training.train_no_rllib_mappo_robust_v2.runtime_metadata", side_effect=lambda: {}):
            self.assertEqual(runtime_metadata_with_carla(newer)["pace_reward_protocol"], expected)
        temporary_root = (PROJECT_DIR / "diagnostics").resolve()
        self.assertTrue(temporary_root.is_relative_to(PROJECT_DIR.resolve()))
        schemas = {}
        for profile in ("instant_pace_v4", "instant_pace_v5"):
            with tempfile.TemporaryDirectory(prefix="instant_pace_test_", dir=temporary_root) as directory:
                self.assertEqual(Path(directory).resolve().parent, temporary_root)
                with patch("utils.training_metrics_recorder.SummaryWriter"):
                    recorder = TrainingMetricsRecorder(
                        run_dir=directory, run_id="v5_identity_test", agent_ids=[0, 1],
                        reward_version="v6", start_update=1, completed_update=0,
                        pace_profile=profile,
                    )
                    try:
                        schemas[profile] = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                    finally:
                        recorder.close()
        schema = schemas["instant_pace_v5"]
        self.assertEqual(schema["pace_profile"], "instant_pace_v5")
        self.assertEqual(schema["reward_protocol"]["pace_intervention"], expected)
        self.assertEqual(schema["reward_protocol"]["replaced_base_components"],
                         [PACE_COMPONENT, "overspeed", "lane_deviation"])
        self.assertIn("unclipped", schema["reward_diagnostics"]["lane_deviation_risk"]["value_semantics"])
        self.assertIn("effective exemption", schema["reward_diagnostics"]["lane_deviation_exempted"]["value_semantics"])
        self.assertEqual(schemas["instant_pace_v4"]["reward_protocol"]["pace_intervention"], instant_pace_v4_spec())
        self.assertEqual(schemas["instant_pace_v4"]["reward_protocol"]["replaced_base_components"],
                         [PACE_COMPONENT, "overspeed"])

    def test_v6_replaces_three_costs_and_preserves_reward_sum(self):
        env = self.environment("instant_pace_v6")
        env.prev_controls = {0: {"steer": 0.0}, 1: {"steer": 0.5}}
        for aid, steer in ((0, .8), (1, -.5)):
            env.vehicles[aid].get_control = lambda steer=steer: SimpleNamespace(steer=steer)
            env.parent_diagnostics[aid]["lateral_distance_m"] = .3
        rewards, details = env._compute_rewards_v6()
        self.assertAlmostEqual(details[0]["steer_rate"], -.016)
        self.assertAlmostEqual(details[1]["steer_rate"], -.025)
        self.assertLess(details[0]["overspeed"], -.8)
        self.assertAlmostEqual(details[0]["lane_deviation"], -.01*(.3/.9)**2)
        self.assertAlmostEqual(env.last_reward_diagnostics[0]["steer_rate_normalized"], 16)
        for aid in range(2):
            self.assertAlmostEqual(rewards[aid], math.fsum(details[aid].values()))
            for name in V5_COMPONENTS:
                if name not in {"overspeed", "lane_deviation", "steer_rate"}:
                    self.assertEqual(details[aid][name], env.parent_details[aid][name])
        self.assertEqual(env.prev_controls[0]["steer"], 0, "reward must not mutate step history")

    def test_v6_first_control_and_inactive_car_do_not_get_stale_steer_cost(self):
        env = self.environment("instant_pace_v6")
        env.prev_controls = {1: {"steer": .9}}
        env.vehicles[0].get_control = lambda: SimpleNamespace(steer=.5)
        env.reached_target[1] = True
        _, details = env._compute_rewards_v6()
        self.assertEqual(details[0]["steer_rate"], 0)
        self.assertEqual(details[1]["steer_rate"], 0)
        self.assertEqual(env.last_reward_diagnostics[1]["steer_rate_normalized"], 0)

    def test_v6_runtime_schema_and_evaluator_identity(self):
        from environment.instant_pace_reward_v6 import instant_pace_spec as spec_v6
        from evaluation.evaluate_reward_fork import checkpoint_environment
        env = self.environment("instant_pace_v6")
        with patch("training.train_no_rllib_mappo_robust_v2.runtime_metadata", side_effect=lambda: {}):
            self.assertEqual(runtime_metadata_with_carla(env)["pace_reward_protocol"], spec_v6())
        self.assertIn("instant_pace_v6", next(a for a in build_parser()._actions if a.dest == "pace_profile").choices)
        with tempfile.TemporaryDirectory(prefix="v6_schema_", dir=PROJECT_DIR / "diagnostics") as directory:
            with patch("utils.training_metrics_recorder.SummaryWriter"):
                recorder = TrainingMetricsRecorder(run_dir=directory, run_id="v6_test", agent_ids=[0, 1],
                    reward_version="v6", start_update=1, completed_update=0, pace_profile="instant_pace_v6")
                try:
                    schema = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                finally:
                    recorder.close()
        self.assertEqual(schema["reward_protocol"]["pace_intervention"], spec_v6())
        self.assertEqual(schema["reward_protocol"]["replaced_base_components"],
                         [PACE_COMPONENT, "overspeed", "lane_deviation", "steer_rate"])
        self.assertIn("unclipped", schema["reward_diagnostics"]["steer_rate_normalized"]["value_semantics"])
        self.assertIn("center", schema["reward_diagnostics"]["lane_deviation_risk"]["value_semantics"])

    def test_live_exemptions_do_not_disable_the_speed_safety_cost(self):
        for gate in ("front", "complex", "unlisted", "startup", "goal"):
            with self.subTest(gate=gate):
                env = self.environment()
                if gate == "front":
                    env._pace_has_front_blocker = lambda _: True
                elif gate == "complex":
                    env.current_scenario_family = "car_following_steady"
                elif gate == "unlisted":
                    env.current_scenario_tags = set()
                    env.current_scenario_family = "unregistered"
                elif gate == "startup":
                    env.current_step = 20
                else:
                    env.parent_diagnostics[0]["distance_to_goal_m"] = 5.0
                _, details = env._compute_rewards_v6()
                self.assertEqual(details[0][PACE_COMPONENT], 0.0)
                self.assertFalse(env.last_reward_diagnostics[0]["pace_deficit_eligible"])
                self.assertLess(details[0]["overspeed"], 0.0)

    def test_terminal_events_preserve_parent_event_once_and_disable_pace(self):
        for event, component, amount in (
            ("success_event", "success", 20.0),
            ("collision_event", "collision", -50.0),
            ("offroad_terminated", "offroad", -50.0),
            ("road_violation_terminated", "road_rule", -50.0),
            ("mission_timeout_event", "mission_timeout", -15.0),
        ):
            with self.subTest(event=event):
                env = self.environment()
                env.parent_diagnostics[0][event] = 1.0
                env.parent_details[0][component] = amount
                rewards, details = env._compute_rewards_v6()
                self.assertEqual(details[0][PACE_COMPONENT], 0.0)
                self.assertEqual(details[0][component], amount)
                self.assertTrue(env.last_reward_diagnostics[0]["pace_terminal_exempt"])
                self.assertAlmostEqual(rewards[0], math.fsum(details[0].values()), places=12)

    def test_finished_or_removed_zero_parent_rows_stay_zero_without_repeat_bonus(self):
        for inactive in ("reached", "removed"):
            with self.subTest(inactive=inactive):
                env = self.environment()
                # This is the frozen parent reward's post-success inactive output.
                env.parent_details[0] = {name: 0.0 for name in V5_COMPONENTS}
                env.parent_diagnostics[0] = {name: 0.0 for name in V5_DIAGNOSTICS}
                env.parent_diagnostics[0]["success_event"] = 1.0
                if inactive == "reached":
                    env.reached_target[0] = True
                else:
                    env.vehicles[0].is_alive = False
                for _ in range(2):
                    rewards, details = env._compute_rewards_v6()
                    self.assertEqual(rewards[0], 0.0)
                    self.assertTrue(all(value == 0.0 for value in details[0].values()))
                    self.assertTrue(env.last_reward_diagnostics[0]["pace_terminal_exempt"])

    def test_parser_environment_and_resume_retain_profile_identity(self):
        args = build_parser().parse_args(["--reward-version", "v6", "--pace-profile", PROFILE])
        self.assertEqual(build_training_environment_kwargs(args, num_agents=2)["pace_profile"], PROFILE)
        config = checkpoint_training_config(args)
        self.assertEqual(config["pace_profile"], PROFILE)
        restored = build_parser().parse_args(["--reward-version", "v6"])
        saved = {"training_config": config,
                 "run_metadata": {"seed": 42, "run_id": "instant_test"},
                 "lr_schedule": {"decay_horizon": 600}}
        apply_resume_configuration(restored, saved)
        self.assertEqual(restored.pace_profile, PROFILE)
        restored._explicit_cli_options = {"--pace-profile"}
        restored.pace_profile = "route_pace_v1"
        with self.assertRaises(CheckpointProtocolError):
            apply_resume_configuration(restored, saved)

    def test_runtime_metadata_identifies_actual_profile_with_no_real_client(self):
        for profile, expected in ((PROFILE, instant_pace_spec()),
                                  ("route_pace_v1", route_pace_spec()),
                                  ("frozen_v6", None)):
            with self.subTest(profile=profile):
                env = self.environment(profile)
                with patch("training.train_no_rllib_mappo_robust_v2.runtime_metadata",
                           return_value={"runtime": "mock"}):
                    observed = runtime_metadata_with_carla(env)
                if expected is None:
                    self.assertNotIn("pace_reward_protocol", observed)
                else:
                    self.assertEqual(observed["pace_reward_protocol"], expected)
                self.assertEqual(observed["environment_reward_gamma"], 0.999)
                self.assertEqual(observed["carla_server"], "mock-server")

    def test_metrics_schema_uses_correct_profile_units_and_keeps_legacy_schema(self):
        temporary_root = (PROJECT_DIR / "diagnostics").resolve()
        self.assertTrue(temporary_root.is_relative_to(PROJECT_DIR.resolve()))
        for profile in (PROFILE, "route_pace_v1", "frozen_v6"):
            with self.subTest(profile=profile):
                with tempfile.TemporaryDirectory(prefix="instant_pace_test_", dir=temporary_root) as directory:
                    self.assertEqual(Path(directory).resolve().parent, temporary_root)
                    with patch("utils.training_metrics_recorder.SummaryWriter"):
                        recorder = TrainingMetricsRecorder(
                            run_dir=directory, run_id="instant_test", agent_ids=[0, 1],
                            reward_version="v6", start_update=1, completed_update=0,
                            pace_profile=profile,
                        )
                        try:
                            schema = json.loads(recorder.schema_path.read_text(encoding="utf-8"))
                        finally:
                            recorder.close()
                    if profile == PROFILE:
                        self.assertEqual(schema["pace_profile"], PROFILE)
                        self.assertEqual(schema["reward_protocol"]["pace_intervention"], instant_pace_spec())
                        self.assertEqual(schema["reward_protocol"]["replaced_base_components"],
                                         [PACE_COMPONENT, "overspeed"])
                        self.assertIn("m/s", schema["reward_diagnostics"]["pace_deficit_raw"]["value_semantics"])
                        self.assertIn("not clipped", schema["reward_diagnostics"]["overspeed_risk"]["value_semantics"])
                    elif profile == "route_pace_v1":
                        self.assertEqual(schema["reward_protocol"]["pace_intervention"], route_pace_spec())
                        self.assertIn("metres", schema["reward_diagnostics"]["pace_deficit_raw"]["value_semantics"])
                        self.assertNotIn("replaced_base_components", schema["reward_protocol"])
                    else:
                        original = reward_schema_document("v6")
                        self.assertEqual(schema["reward_protocol"], original["reward_protocol"])
                        self.assertEqual(schema["reward_diagnostics"], original["reward_diagnostics"])
                        self.assertNotIn("pace_profile", schema)


if __name__ == "__main__":
    unittest.main()
