import math
import unittest
from pathlib import Path

from environment.base_reward_v5 import (
    REWARD_COMPONENTS as REWARD_V5_COMPONENTS,
    REWARD_DIAGNOSTICS as REWARD_V5_DIAGNOSTICS,
)
from environment.base_reward_v6 import (
    DEFAULT_REWARD_V6_CONFIG,
    PACE_COMPONENT,
    REWARD_COMPONENTS as REWARD_V6_COMPONENTS,
    REWARD_DIAGNOSTICS as REWARD_V6_DIAGNOSTICS,
    RewardV6Config,
    free_flow_pace_deficit,
    reward_v6_spec,
    validate_reward_v6_config,
)
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from training.checkpoint_protocol import (
    CheckpointProtocolError,
    R8C1_CHECKPOINT_SCHEMA_VERSION,
    R8C1_TRAINING_PROTOCOL_VERSION,
    checkpoint_schema_version_for_reward,
    training_protocol_version_for_reward,
    validate_checkpoint_header,
    validate_checkpoint_payload,
)
from training.critic_diagnostics import build_critic_probe_batch
from training.experiment_contract import load_experiment_contract
from training.metrics_protocol import (
    R7_EPISODE_RECORD_VERSION,
    R8C1_METRICS_PROTOCOL_VERSION,
    episode_record_version,
    metrics_protocol_version,
    reward_components,
    reward_diagnostics,
    reward_schema_document,
)
from training.train_no_rllib_mappo_robust_v2 import (
    build_parser,
    checkpoint_protocol_versions,
    observation_dim_for_reward,
)
from tests.test_checkpoint_resume import _checkpoint_payload


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_DIR
    / "configs"
    / "base_mappo"
    / "p0_base_town05_r8c1_pace_v1.json"
)


class RewardV6MathTests(unittest.TestCase):
    def test_v6_is_v5_plus_one_bounded_dense_cost(self):
        self.assertEqual(REWARD_V6_COMPONENTS[:-1], REWARD_V5_COMPONENTS)
        self.assertEqual(REWARD_V6_COMPONENTS[-1], PACE_COMPONENT)
        self.assertEqual(
            REWARD_V6_DIAGNOSTICS[:len(REWARD_V5_DIAGNOSTICS)],
            REWARD_V5_DIAGNOSTICS,
        )
        spec = reward_v6_spec()
        self.assertEqual(spec["positive_dense_components"], ["progress"])
        self.assertFalse(
            spec["pace_deficit"]["policy_invariance_guarantee"]
        )

    def test_on_schedule_or_ahead_is_neutral(self):
        on_schedule = free_flow_pace_deficit(
            0.5,
            0.5,
            current_step=100,
            free_flow=True,
            non_interactive=True,
        )
        ahead = free_flow_pace_deficit(
            0.3,
            0.5,
            current_step=100,
            free_flow=True,
            non_interactive=True,
        )
        self.assertEqual(on_schedule.reward, 0.0)
        self.assertEqual(on_schedule.raw_deficit, 0.0)
        self.assertEqual(ahead.reward, 0.0)

    def test_stationary_cost_grows_and_remains_bounded(self):
        middle = free_flow_pace_deficit(
            1.0,
            0.5,
            current_step=100,
            free_flow=True,
            non_interactive=True,
        )
        late = free_flow_pace_deficit(
            1.0,
            0.1,
            current_step=180,
            free_flow=True,
            non_interactive=True,
        )
        self.assertAlmostEqual(middle.reward, -0.01 * 0.45 / 0.95)
        self.assertLess(late.reward, middle.reward)
        self.assertGreaterEqual(late.reward, -DEFAULT_REWARD_V6_CONFIG.pace_deficit_weight)

    def test_grace_traffic_complex_near_goal_and_terminal_are_exempt(self):
        common = dict(
            remaining_route_fraction=1.0,
            remaining_time_fraction=0.2,
            free_flow=True,
            non_interactive=True,
        )
        self.assertEqual(
            free_flow_pace_deficit(current_step=19, **common).reward, 0.0
        )
        self.assertEqual(
            free_flow_pace_deficit(
                current_step=100, front_blocker=True, **common
            ).reward,
            0.0,
        )
        self.assertEqual(
            free_flow_pace_deficit(
                current_step=100, complex_interaction=True, **common
            ).reward,
            0.0,
        )
        self.assertEqual(
            free_flow_pace_deficit(
                0.1,
                0.0,
                current_step=199,
                free_flow=True,
                non_interactive=True,
            ).reward,
            0.0,
        )
        self.assertEqual(
            free_flow_pace_deficit(
                current_step=100, terminal_event=True, **common
            ).reward,
            0.0,
        )

    def test_both_required_tags_are_strict(self):
        for free_flow, non_interactive in ((False, True), (True, False)):
            decision = free_flow_pace_deficit(
                1.0,
                0.2,
                current_step=100,
                free_flow=free_flow,
                non_interactive=non_interactive,
            )
            self.assertFalse(decision.eligible)
            self.assertEqual(decision.reward, 0.0)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            free_flow_pace_deficit(
                math.nan,
                0.5,
                current_step=100,
                free_flow=True,
                non_interactive=True,
            )
        with self.assertRaises(ValueError):
            validate_reward_v6_config(RewardV6Config(pace_deficit_weight=0.0))


class RewardV6IntegrationTests(unittest.TestCase):
    def _environment(self):
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.num_agents = 2
        env.adversarial_agent_ids = []
        env.current_step = 100
        env.current_scenario_tags = frozenset({"free_flow", "non_interactive"})
        env.current_scenario_family = "parallel_free_flow"

        details = {
            agent_id: {name: 0.0 for name in REWARD_V5_COMPONENTS}
            for agent_id in range(2)
        }
        diagnostics = {
            agent_id: {name: 0.0 for name in REWARD_V5_DIAGNOSTICS}
            for agent_id in range(2)
        }
        for values in diagnostics.values():
            values["remaining_route_fraction"] = 1.0
            values["remaining_time_fraction"] = 0.5
        env._pace_has_front_blocker = lambda agent_id: agent_id == 1

        def frozen_v5():
            env.last_reward_diagnostics = diagnostics
            return {0: 0.0, 1: 0.0}, details

        env._compute_rewards_v5 = frozen_v5
        return env

    def test_environment_adds_only_pace_component_and_exempts_blocker(self):
        env = self._environment()
        rewards, details = env._compute_rewards_v6()
        expected = -0.01 * 0.45 / 0.95
        self.assertAlmostEqual(rewards[0], expected)
        self.assertAlmostEqual(details[0][PACE_COMPONENT], expected)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(set(details[0]), set(REWARD_V6_COMPONENTS))
        diagnostics = env.last_reward_diagnostics
        self.assertEqual(set(diagnostics[0]), set(REWARD_V6_DIAGNOSTICS))
        self.assertEqual(diagnostics[0]["pace_deficit_eligible"], 1.0)
        self.assertEqual(diagnostics[1]["pace_front_blocker_exempt"], 1.0)
        self.assertAlmostEqual(
            rewards[0], math.fsum(details[0].values()), places=12
        )

    def test_protocol_identities_are_distinct_and_parser_accepts_v6(self):
        self.assertEqual(observation_dim_for_reward("v6"), 20)
        self.assertEqual(
            build_parser().parse_args(["--reward-version", "v6"]).reward_version,
            "v6",
        )
        self.assertEqual(
            checkpoint_schema_version_for_reward("v6"),
            R8C1_CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(
            training_protocol_version_for_reward("v6"),
            R8C1_TRAINING_PROTOCOL_VERSION,
        )
        self.assertEqual(metrics_protocol_version("v6"), R8C1_METRICS_PROTOCOL_VERSION)
        self.assertEqual(episode_record_version("v6"), R7_EPISODE_RECORD_VERSION)
        self.assertEqual(reward_components("v6"), REWARD_V6_COMPONENTS)
        self.assertEqual(reward_diagnostics("v6"), REWARD_V6_DIAGNOSTICS)
        self.assertEqual(
            reward_schema_document("v6")["reward_protocol"], reward_v6_spec()
        )

    def test_formal_u100_contract_is_loadable_and_reward_only(self):
        contract = load_experiment_contract(
            CONTRACT_PATH, project_dir=PROJECT_DIR
        )["payload"]
        self.assertEqual(contract["planned_total_updates"], 100)
        self.assertEqual(
            contract["segment_stop_updates"],
            [1, 5, 10, 20, 40, 60, 80, 100],
        )
        self.assertEqual(contract["trainer_args"]["reward_version"], "v6")
        self.assertEqual(
            contract["trainer_args"]["actor_init_mode"], "independent"
        )
        self.assertEqual(contract["reward_protocol"], reward_v6_spec())

    def test_v5_checkpoint_cannot_be_relabelled_as_v6(self):
        checkpoint = _checkpoint_payload()
        checkpoint["checkpoint_schema_version"] = R8C1_CHECKPOINT_SCHEMA_VERSION
        checkpoint["training_protocol_version"] = R8C1_TRAINING_PROTOCOL_VERSION
        checkpoint["protocol_versions"] = {
            **checkpoint["protocol_versions"],
            "reward": "v5",
        }
        with self.assertRaises(CheckpointProtocolError):
            validate_checkpoint_header(checkpoint)

    def test_v6_checkpoint_payload_uses_its_own_schema(self):
        checkpoint = _checkpoint_payload()
        checkpoint["checkpoint_schema_version"] = R8C1_CHECKPOINT_SCHEMA_VERSION
        checkpoint["training_protocol_version"] = R8C1_TRAINING_PROTOCOL_VERSION
        checkpoint["protocol_versions"] = checkpoint_protocol_versions(
            "v6", 20.0, "r8a_mastery_v1"
        )
        checkpoint["last_metrics"]["metrics_protocol_version"] = (
            R8C1_METRICS_PROTOCOL_VERSION
        )
        checkpoint["training_config"] = {
            **checkpoint["training_config"],
            "reward_version": "v6",
        }
        checkpoint["total_road_violations"] = 0
        state_dim = int(checkpoint["model_spec"]["state_dim"])
        checkpoint["critic_probe_batch"] = build_critic_probe_batch(
            states=[
                [float(row + column) for column in range(state_dim)]
                for row in range(4)
            ],
            agent_ids=[0, 1, 0, 1],
            targets=[0.0, 1.0, -1.0, 2.0],
            maximum_rows=4,
        )
        validate_checkpoint_payload(
            checkpoint,
            expected_protocol_versions=checkpoint["protocol_versions"],
            expected_action_spec=checkpoint["action_spec"],
            expected_model_spec=checkpoint["model_spec"],
            expected_training_config=checkpoint["training_config"],
            expected_critic_spec=checkpoint["critic_spec"],
        )


if __name__ == "__main__":
    unittest.main()
