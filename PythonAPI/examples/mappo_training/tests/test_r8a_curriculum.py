import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from training.checkpoint_protocol import scenario_pool_fingerprint
from training.experiment_contract import (
    ExperimentContractError,
    apply_experiment_contract,
    load_experiment_contract,
    validate_experiment_contract,
)
from training.mastery_curriculum import (
    MasteryWindow,
    R8AMasteryGate,
    r8a_mastery_protocol_spec,
)
from training.r8a_curriculum import (
    R8A_MASTERY_PROTOCOL_VERSION,
    R8A_SAMPLING_PROTOCOL_VERSION,
    R8A_STAGE_NEW_SCENARIOS,
    r8a_cumulative_stages,
    r8a_sampler_state,
    sample_r8a_balanced,
    validate_r8a_sampler_state,
)
from training.train_no_rllib_mappo_robust_v2 import (
    CurriculumLearning,
    build_parser,
    checkpoint_protocol_versions,
)
from utils.build_town05_r8a_scenarios import write_payloads
from utils.validate_r8a_scenarios import validate_r8a_scenario_pool
from utils.validate_tensorboard_run import SUPPORTED_METRICS_PROTOCOL_VERSIONS
from training.metrics_protocol import R7_METRICS_PROTOCOL_VERSION


PROJECT_DIR = Path(__file__).resolve().parents[1]
R7_CONTRACT = PROJECT_DIR / "configs/base_mappo/p0_base_town05_r7_v5.json"
R8A_CONTRACT = PROJECT_DIR / "configs/base_mappo/p0_base_town05_r8a_v1.json"
R7_SCENARIOS = PROJECT_DIR / "data/training_scenarios_town05_v2"
R8A_SCENARIOS = PROJECT_DIR / "data/training_scenarios_town05_r8a_v1"


def _passing_window(update, *, episode_count=8, speed=1.0):
    return MasteryWindow(
        update=update,
        episode_count=episode_count,
        agent_mean_speed_mps={0: speed, 1: speed},
        agent_mean_completion={0: 0.8, 1: 0.8},
        agent_brake_mode_rate={0: 0.0, 1: 0.0},
        team_success_rate=0.8,
        team_timeout_rate=0.0,
        team_collision_rate=0.0,
        team_offroad_rate=0.0,
        team_road_violation_rate=0.0,
    )


class R8AContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r7 = load_experiment_contract(
            R7_CONTRACT, project_dir=PROJECT_DIR
        )["payload"]
        cls.r8a = load_experiment_contract(
            R8A_CONTRACT, project_dir=PROJECT_DIR
        )["payload"]

    def test_contract_registers_u150_but_retains_r7_schedules(self):
        self.assertEqual(self.r8a["planned_total_updates"], 150)
        self.assertEqual(
            self.r8a["segment_stop_updates"],
            [1, 5, 10, 20, 40, 60, 80, 100, 120, 150],
        )
        self.assertEqual(self.r8a["trainer_args"]["lr_decay_horizon"], 600)
        self.assertEqual(self.r8a["trainer_args"]["entropy_anneal_updates"], 600)
        self.assertEqual(self.r8a["trainer_args"]["critic_lr_hold_updates"], 300)

    def test_contract_changes_only_curriculum_owned_trainer_arguments(self):
        r7_args = dict(self.r7["trainer_args"])
        r8a_args = dict(self.r8a["trainer_args"])
        changed = {
            name for name in r7_args if r7_args[name] != r8a_args[name]
        }
        self.assertEqual(
            changed,
            {"scenario_dir", "scenario_sampling_mode", "curriculum_protocol"},
        )
        for protocol in (
            "reward_protocol",
            "offroad_protocol",
            "road_legality_protocol",
            "stall_protocol",
            "policy_distribution_protocol",
            "critic_diagnostics_protocol",
            "learning_rate_protocol",
        ):
            self.assertEqual(self.r8a[protocol], self.r7[protocol])
        self.assertEqual(
            self.r8a["mastery_protocol"]["parameters"],
            self.r7["mastery_protocol"]["parameters"],
        )
        self.assertEqual(
            self.r8a["mastery_protocol"], r8a_mastery_protocol_spec()
        )

    def test_formal_u1_is_registered_without_changing_periodic_interval(self):
        run_id = "P0_BASE_MAPPO_Town05R8A_s42_20260828_120000"
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / run_id
            argv = [
                "--experiment-contract", str(R8A_CONTRACT),
                "--updates", "1",
                "--run-id", run_id,
                "--save-dir", str(save_dir),
            ]
            args = build_parser().parse_args(argv)
            args._explicit_cli_options = {
                token.split("=", 1)[0]
                for token in argv if token.startswith("--")
            }
            apply_experiment_contract(args, project_dir=PROJECT_DIR)
            self.assertEqual(args.updates, 1)
            self.assertEqual(args.save_interval, 5)
            self.assertEqual(args.scenario_sampling_mode, "r8a_balanced")

    def test_only_update_one_may_use_the_latest_checkpoint_between_intervals(self):
        drifted = copy.deepcopy(self.r8a)
        drifted["segment_stop_updates"] = [1, 5, 7, 10, 20, 40, 60, 80, 100, 150]
        drifted["evaluation_contract"]["milestone_updates"] = list(
            drifted["segment_stop_updates"]
        )
        with self.assertRaisesRegex(ExperimentContractError, "update 1"):
            validate_experiment_contract(drifted, project_dir=PROJECT_DIR)

    def test_r8a_protocol_versions_do_not_relabel_r7_history(self):
        r7 = checkpoint_protocol_versions("v5", 20.0, "mastery_v2")
        r8a = checkpoint_protocol_versions("v5", 20.0, "r8a_mastery_v1")
        self.assertNotIn("scenario_sampling", r7)
        self.assertEqual(r8a["scenario_sampling"], R8A_SAMPLING_PROTOCOL_VERSION)
        self.assertEqual(r8a["curriculum"], R8A_MASTERY_PROTOCOL_VERSION)
        self.assertEqual(r8a["reward"], r7["reward"])

    def test_live_validator_covers_all_short_scenarios_and_cleanup(self):
        source = (
            PROJECT_DIR / "utils/validate_r8a_curriculum_live.py"
        ).read_text(encoding="utf-8")
        self.assertIn("R8A_STAGE_NEW_SCENARIOS[:2]", source)
        self.assertIn("max_abs_reward_residual", source)
        self.assertIn("CARLA cleanup invariant failed", source)

    def test_strict_metrics_validator_registers_the_frozen_r7_schema(self):
        self.assertIn(
            R7_METRICS_PROTOCOL_VERSION,
            SUPPORTED_METRICS_PROTOCOL_VERSIONS,
        )
        self.assertIn(
            "utils/validate_tensorboard_run.py",
            self.r8a["critical_source_paths"],
        )

    def test_r7_scenario_pool_is_unchanged_and_r8a_is_distinct(self):
        self.assertEqual(
            scenario_pool_fingerprint(R7_SCENARIOS),
            "fd7ea9605b60a7f22fa40e4491399cd76700cc04334c7054eccbdc69983e82ae",
        )
        self.assertNotEqual(
            scenario_pool_fingerprint(R8A_SCENARIOS),
            scenario_pool_fingerprint(R7_SCENARIOS),
        )
        self.assertEqual(
            scenario_pool_fingerprint(R8A_SCENARIOS),
            "1bc0013442f2f6107089fdd5c5829d991f9cdd3bd1e862c6a657cc8eb0965232",
        )


class R8AScenarioAndSamplingTests(unittest.TestCase):
    def test_generated_pool_is_current_and_strictly_valid(self):
        write_payloads(check=True)
        self.assertEqual(validate_r8a_scenario_pool(R8A_SCENARIOS), [])

    def test_metadata_reconstructs_exact_cumulative_stages(self):
        curriculum = CurriculumLearning(
            str(R8A_SCENARIOS),
            mode="metadata",
            mastery_enabled=True,
            r8a_enabled=True,
            maximum_team_road_violation_rate=0.10,
        )
        self.assertEqual(curriculum.curriculum_stages, r8a_cumulative_stages())

    def test_initial_update_is_exactly_eight_eight_for_every_phase(self):
        scenarios = R8A_STAGE_NEW_SCENARIOS[0]
        for phase in range(20):
            selected = sample_r8a_balanced(scenarios, 16, phase=phase)
            self.assertEqual(Counter(selected), Counter({
                scenarios[0]: 8,
                scenarios[1]: 8,
            }))

    def test_later_stage_remainders_rotate_without_count_bias(self):
        stage = r8a_cumulative_stages()[-1]
        cumulative = Counter()
        for phase in range(len(stage)):
            selected = sample_r8a_balanced(stage, 16, phase=phase)
            counts = Counter(selected)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            cumulative.update(selected)
        self.assertEqual(len(set(cumulative.values())), 1)

    def test_sampler_resume_phase_depends_only_on_completed_update(self):
        scenarios = R8A_STAGE_NEW_SCENARIOS[0]
        uninterrupted = sample_r8a_balanced(scenarios, 16, phase=5)
        resumed = sample_r8a_balanced(scenarios, 16, phase=5)
        wrong_phase = sample_r8a_balanced(scenarios, 16, phase=6)
        self.assertEqual(uninterrupted, resumed)
        self.assertNotEqual(uninterrupted, wrong_phase)
        state = r8a_sampler_state(5)
        validate_r8a_sampler_state(state, expected_completed_update=5)
        with self.assertRaisesRegex(ValueError, "sampler state mismatch"):
            validate_r8a_sampler_state(state, expected_completed_update=6)

    def test_scenario_episode_limit_is_non_sticky_and_contract_bounded(self):
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.map = SimpleNamespace(name="Town05")
        env.default_max_steps = 500
        env.max_steps = 500
        env.set_goal_zone = lambda value: setattr(env, "goal_zone", value)
        env.set_agent_goal_zones = lambda value: setattr(env, "agent_goal_zones", value)
        env.set_custom_start_positions = lambda value: setattr(env, "starts", value)
        env.apply_scenario_config({"town": "Town05", "episode_max_steps": 200})
        self.assertEqual(env.max_steps, 200)
        env.apply_scenario_config({"town": "Town05"})
        self.assertEqual(env.max_steps, 500)
        with self.assertRaisesRegex(ValueError, "contract max_steps"):
            env.apply_scenario_config({"town": "Town05", "episode_max_steps": 501})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            env.apply_scenario_config({"town": "Town05", "episode_max_steps": 200.5})

    def test_sampler_rejects_fractional_integer_fields(self):
        scenarios = R8A_STAGE_NEW_SCENARIOS[0]
        with self.assertRaisesRegex(ValueError, "positive integer"):
            sample_r8a_balanced(scenarios, 16.5, phase=0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            sample_r8a_balanced(scenarios, 16, phase=0.5)


class R8AStratifiedMasteryTests(unittest.TestCase):
    def test_pooled_success_cannot_hide_one_mirror_failure(self):
        gate = R8AMasteryGate(stage_count=4, agent_ids=[0, 1])
        north, south = R8A_STAGE_NEW_SCENARIOS[0]
        evidence = gate.evaluate(
            _passing_window(1, episode_count=16),
            scenario_windows={
                north: _passing_window(1),
                south: _passing_window(1, speed=0.0),
            },
        )
        self.assertFalse(evidence["passed"])
        self.assertIn(
            f"scenario.{south}.agent0.mean_speed", evidence["failures"]
        )
        self.assertEqual(gate.stage_index, 0)

    def test_eight_episode_stratum_makes_one_safety_failure_fail_closed(self):
        gate = R8AMasteryGate(stage_count=4, agent_ids=[0, 1])
        north, south = R8A_STAGE_NEW_SCENARIOS[0]
        unsafe = _passing_window(1)
        unsafe = MasteryWindow(
            **{
                **unsafe.__dict__,
                "team_success_rate": 0.875,
                "team_collision_rate": 0.125,
            }
        )
        evidence = gate.evaluate(
            _passing_window(1, episode_count=16),
            scenario_windows={
                north: unsafe,
                south: _passing_window(1),
            },
        )
        self.assertFalse(evidence["passed"])
        self.assertIn(
            f"scenario.{north}.team.collision_rate", evidence["failures"]
        )

    def test_two_stratified_windows_promote_and_resume_exactly(self):
        gate = R8AMasteryGate(stage_count=4, agent_ids=[0, 1])
        north, south = R8A_STAGE_NEW_SCENARIOS[0]
        for update in (1, 2):
            evidence = gate.evaluate(
                _passing_window(update, episode_count=16),
                scenario_windows={
                    north: _passing_window(update),
                    south: _passing_window(update),
                },
            )
        self.assertTrue(evidence["promoted"])
        self.assertEqual(gate.stage_index, 1)
        restored = R8AMasteryGate(stage_count=4, agent_ids=[0, 1])
        restored.load_state_dict(copy.deepcopy(gate.state_dict()))
        self.assertEqual(restored.state_dict(), gate.state_dict())


if __name__ == "__main__":
    unittest.main()
