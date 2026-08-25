import unittest
from pathlib import Path
import json
import tempfile

from environment.base_reward_v5 import reward_v5_spec
from environment.road_legality_protocol import road_legality_protocol_spec
from tests.test_episode_lifecycle import _make_env
from training.checkpoint_protocol import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointProtocolError,
    R7_CHECKPOINT_SCHEMA_VERSION,
    R7_TRAINING_PROTOCOL_VERSION,
    checkpoint_schema_version_for_reward,
    training_protocol_version_for_reward,
    validate_checkpoint_payload,
)
from training.critic_diagnostics import (
    build_critic_probe_batch,
    critic_diagnostics_protocol_spec,
)
from training.experiment_contract import load_experiment_contract
from training.learning_rate_protocol import learning_rate_protocol_spec
from training.mastery_curriculum import MasteryGate, MasteryWindow, mastery_protocol_spec
from training.metrics_protocol import (
    R6_EPISODE_RECORD_VERSION,
    R7_EPISODE_RECORD_VERSION,
    R7_METRICS_PROTOCOL_VERSION,
    metrics_protocol_version,
    validate_episode_record,
    reward_components,
    reward_diagnostics,
)
from training.train_no_rllib_mappo_robust_v2 import (
    build_parser,
    checkpoint_protocol_versions,
    classify_base_episode_outcome_r7,
    observation_dim_for_reward,
)
from utils.training_metrics_recorder import TrainingMetricsRecorder
from tests.test_checkpoint_resume import _checkpoint_payload


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_town05_r7_v5.json"
)


def _episode_record(*, road_violation=True):
    return {
        "episode_record_version": R7_EPISODE_RECORD_VERSION,
        "run_id": "r7",
        "update": 1,
        "episode_index": 0,
        "created_at": "2026-08-23T00:00:00+00:00",
        "scenario_name": "scenario.json",
        "episode_length": 5,
        "episode_end_reason": "cooperative_road_violation",
        "success": False,
        "collision": False,
        "timeout": False,
        "offroad": False,
        "road_violation": road_violation,
        "agents": {
            "0": {
                "episode_return": -50.0,
                "active_decision_steps": 5,
                "transition_count": 5,
                "done": True,
                "terminated": True,
                "truncated": False,
                "bootstrap_enabled": False,
                "offroad_terminated": False,
                "first_offroad_evidence": None,
                "max_offroad_consecutive_ticks": 0,
                "max_footprint_outside_fraction": 0.0,
                "road_rule_terminated": road_violation,
            },
            "1": {
                "episode_return": 0.0,
                "active_decision_steps": 5,
                "transition_count": 5,
                "done": True,
                "terminated": False,
                "truncated": True,
                "bootstrap_enabled": True,
                "offroad_terminated": False,
                "first_offroad_evidence": None,
                "max_offroad_consecutive_ticks": 0,
                "max_footprint_outside_fraction": 0.0,
                "road_rule_terminated": False,
            },
        },
    }


class R7ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_experiment_contract(
            CONTRACT_PATH, project_dir=PROJECT_DIR
        )["payload"]

    def test_contract_freezes_new_math_and_600_update_horizon(self):
        contract = self.contract
        self.assertEqual(contract["planned_total_updates"], 600)
        self.assertEqual(contract["trainer_args"]["reward_version"], "v5")
        self.assertEqual(contract["trainer_args"]["critic_lr_hold_updates"], 300)
        self.assertEqual(contract["reward_protocol"], reward_v5_spec())
        self.assertEqual(
            contract["road_legality_protocol"], road_legality_protocol_spec()
        )
        self.assertEqual(
            contract["critic_diagnostics_protocol"],
            critic_diagnostics_protocol_spec(),
        )
        self.assertEqual(
            contract["learning_rate_protocol"], learning_rate_protocol_spec()
        )
        self.assertEqual(
            contract["mastery_protocol"],
            mastery_protocol_spec(maximum_team_road_violation_rate=0.10),
        )

    def test_r7_keeps_observation_dimension_and_has_distinct_identities(self):
        self.assertEqual(build_parser().parse_args(["--reward-version", "v5"]).reward_version, "v5")
        self.assertEqual(observation_dim_for_reward("v4"), 20)
        self.assertEqual(observation_dim_for_reward("v5"), 20)
        self.assertEqual(checkpoint_schema_version_for_reward("v4"), CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(checkpoint_schema_version_for_reward("v5"), R7_CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(training_protocol_version_for_reward("v5"), R7_TRAINING_PROTOCOL_VERSION)
        versions = checkpoint_protocol_versions("v5", 20.0)
        self.assertEqual(versions["metrics"], R7_METRICS_PROTOCOL_VERSION)
        self.assertIn("road_legality", versions)
        self.assertIn("critic_diagnostics", versions)
        self.assertNotEqual(R6_EPISODE_RECORD_VERSION, R7_EPISODE_RECORD_VERSION)

    def test_r7_outcome_and_episode_record_are_explicit(self):
        self.assertEqual(
            classify_base_episode_outcome_r7("cooperative_road_violation"),
            (False, False, False, False, True),
        )
        normalized = validate_episode_record(
            _episode_record(), run_id="r7", agent_ids=[0, 1], context="test"
        )
        self.assertTrue(normalized["road_violation"])
        tampered = _episode_record(road_violation=False)
        tampered["timeout"] = True
        tampered["agents"]["0"]["road_rule_terminated"] = True
        with self.assertRaisesRegex(Exception, "road_violation must match"):
            validate_episode_record(
                tampered, run_id="r7", agent_ids=[0, 1], context="test"
            )

    def test_r7_checkpoint_requires_real_replayable_critic_probe(self):
        checkpoint = _checkpoint_payload()
        checkpoint["checkpoint_schema_version"] = R7_CHECKPOINT_SCHEMA_VERSION
        checkpoint["training_protocol_version"] = R7_TRAINING_PROTOCOL_VERSION
        checkpoint["protocol_versions"] = checkpoint_protocol_versions("v5", 20.0)
        checkpoint["last_metrics"]["metrics_protocol_version"] = (
            checkpoint["protocol_versions"]["metrics"]
        )
        checkpoint["training_config"] = {
            **checkpoint["training_config"],
            "reward_version": "v5",
        }
        checkpoint["model_spec"] = {
            **checkpoint["model_spec"],
            "obs_dim": 20,
            "state_dim": 12,
        }
        checkpoint["total_road_violations"] = 2
        checkpoint["critic_probe_batch"] = build_critic_probe_batch(
            states=[[float(row + col) for col in range(12)] for row in range(4)],
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
        checkpoint.pop("critic_probe_batch")
        with self.assertRaisesRegex(
            CheckpointProtocolError, "missing critic_probe_batch"
        ):
            validate_checkpoint_payload(
                checkpoint,
                expected_protocol_versions=checkpoint["protocol_versions"],
                expected_action_spec=checkpoint["action_spec"],
                expected_model_spec=checkpoint["model_spec"],
                expected_training_config=checkpoint["training_config"],
                expected_critic_spec=checkpoint["critic_spec"],
            )

    def test_persistent_road_violation_terminates_offender_and_truncates_peer(self):
        env = _make_env(num_agents=2, adversarial_agent_ids=[])
        env.reward_version = "v5"
        env.road_violation_terminal_events = {0: True, 1: False}
        env.road_legality_decisions = {
            0: {"violation_raw": True, "success_eligible": False},
            1: {"violation_raw": False, "success_eligible": True},
        }
        dones = env._check_dones()
        self.assertEqual(dones, {0: True, 1: True})
        self.assertEqual(env.episode_end_reason, "cooperative_road_violation")
        self.assertEqual(env.last_terminated, {0: True, 1: False})
        self.assertEqual(env.last_truncated, {0: False, 1: True})

    def test_road_violation_has_its_own_mastery_gate(self):
        gate = MasteryGate(
            stage_count=2,
            agent_ids=[0, 1],
            maximum_team_road_violation_rate=0.10,
        )
        evidence = gate.evaluate(MasteryWindow(
            update=1,
            episode_count=16,
            agent_mean_speed_mps={0: 1.0, 1: 1.0},
            agent_mean_completion={0: 0.5, 1: 0.5},
            agent_brake_mode_rate={0: 0.0, 1: 0.0},
            team_success_rate=0.5,
            team_timeout_rate=0.0,
            team_collision_rate=0.0,
            team_offroad_rate=0.0,
            team_road_violation_rate=0.25,
        ))
        self.assertFalse(evidence["passed"])
        self.assertIn("team.road_violation_rate", evidence["failures"])
        self.assertNotIn("team.offroad_rate", evidence["failures"])

    def test_raw_road_violation_blocks_illegal_goal_success(self):
        env = _make_env(num_agents=2, adversarial_agent_ids=[])
        env.reward_version = "v5"
        env.road_violation_terminal_events = {0: False, 1: False}
        env.road_legality_decisions = {
            0: {"violation_raw": True, "success_eligible": False},
            1: {"violation_raw": False, "success_eligible": True},
        }
        env.vehicles[0]._transform.location.in_goal = True
        dones = env._check_dones()
        self.assertFalse(dones[0])
        self.assertFalse(env.reached_target[0])

    def test_r7_recorder_keeps_full_critic_audit_out_of_tensorboard_scalars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TrainingMetricsRecorder(
                run_dir=temp_dir,
                run_id="r7",
                agent_ids=[0],
                reward_version="v5",
                start_update=1,
                completed_update=0,
                histogram_interval=0,
            )
            components = {name: 0.0 for name in reward_components("v5")}
            components["road_rule"] = -50.0
            diagnostics = {name: 0.0 for name in reward_diagnostics("v5")}
            diagnostics["road_violation_raw"] = 1.0
            diagnostics["road_violation_terminated"] = 1.0
            recorder.record_env_step(
                active_agent_ids=[0],
                rewards={0: -50.0},
                reward_details={0: components},
                reward_diagnostics_by_agent={0: diagnostics},
                policy_actions={0: [0.0, 0.0]},
                environment_actions={0: [0.0, 0.0, 0.0]},
                context="r7 recorder",
            )
            recorder.finish_episode(
                update=1,
                episode_index=0,
                episode_returns={0: -50.0},
                episode_length=1,
                active_decision_steps={0: 1},
                transition_counts={0: 1},
                dones={0: True},
                terminated={0: True},
                truncated={0: False},
                bootstrap_enabled={0: False},
                success=False,
                collision=False,
                timeout=False,
                offroad=False,
                road_violation=True,
                road_rule_terminated={0: True},
                end_reason="cooperative_road_violation",
                scenario_name="scenario.json",
                context="r7 recorder",
            )
            recorder.record_audit(
                "critic_groups", {"per_agent": {"agent_0": {"count": 1.0}}}
            )
            payload = recorder.finalize_update(1)
            self.assertEqual(
                payload["metrics_protocol_version"], R7_METRICS_PROTOCOL_VERSION
            )
            self.assertIn("critic_groups", payload["audits"])
            self.assertFalse(
                any("critic_groups" in tag for tag in payload["scalars"])
            )
            recorder.commit_payload(payload)
            recorder.close()
            episode = json.loads(
                (Path(temp_dir) / "episodes.jsonl").read_text(encoding="utf-8")
            )
            self.assertTrue(episode["road_violation"])


if __name__ == "__main__":
    unittest.main()
