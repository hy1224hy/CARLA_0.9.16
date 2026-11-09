import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from environment.base_reward_v4 import reward_v4_spec
from environment.offroad_protocol import r6_offroad_protocol_spec
from environment.stall_protocol import stall_protocol_spec
from evaluation.base_c0_protocol import (
    expand_profile_cases,
    load_and_validate_c0_checkpoint,
    load_c0_contract,
)
from evaluation.base_go_hold_protocol import load_go_hold_contract
from evaluation.evaluate_base_policy_c0 import (
    C0ProtocolError,
    _observation,
    _run_case,
)
from models.actor_critic import Actor
from training.experiment_contract import load_experiment_contract
from training.action_protocol import action_protocol_spec
from training.checkpoint_protocol import (
    CheckpointProtocolError,
    OFFROAD_CHECKPOINT_SCHEMA_VERSION,
    OFFROAD_TRAINING_PROTOCOL_VERSION,
    capture_rng_state,
    save_checkpoint_file,
)
from training.mastery_curriculum import mastery_protocol_spec
from training.policy_distribution_protocol import policy_distribution_spec
from training.metrics_protocol import (
    OFFROAD_METRICS_PROTOCOL_VERSION,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
)
from training.value_normalization import ValueNormalizer
from training.train_no_rllib_mappo_robust_v2 import (
    checkpoint_protocol_versions,
    observation_dim_for_reward,
    runtime_metadata_with_carla,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAINING_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_base_town05_r6_v4.json"
)
DIAG50_PATH = (
    PROJECT_DIR / "configs" / "base_mappo" / "p0_diag_town05_r6_u50_v1.json"
)
C0_PATH = (
    PROJECT_DIR / "configs" / "evaluation" / "p0_base_c0_town05_r6_v4.json"
)
GO_HOLD_PATH = (
    PROJECT_DIR
    / "configs"
    / "evaluation"
    / "p0_base_go_hold_town05_r6_v4.json"
)


class _TrafficManager:
    def set_random_device_seed(self, seed):
        self.seed = seed


class _OneStepR6MissionTimeoutEnv:
    def __init__(self):
        self.traffic_manager = _TrafficManager()
        self.reached_target = {0: False, 1: False}

    def apply_scenario_config(self, config):
        self.config = config

    def reset(self):
        return (
            {0: np.zeros(20, dtype=np.float32), 1: np.zeros(20, dtype=np.float32)},
            {
                "active_mask": {0: True, 1: True},
                "terminated": {0: False, 1: False},
                "truncated": {0: False, 1: False},
                "episode_end_reason": "running",
            },
        )

    def step(self, actions):
        details = {}
        diagnostics = {}
        rewards = {}
        for agent_id in (0, 1):
            components = {name: 0.0 for name in REWARD_COMPONENTS["v4"]}
            components["mission_timeout"] = -20.0
            details[agent_id] = components
            values = {name: 0.0 for name in REWARD_DIAGNOSTICS["v4"]}
            values["mission_timeout_event"] = 1.0
            diagnostics[agent_id] = values
            rewards[agent_id] = -20.0
        return (
            {0: np.zeros(20, dtype=np.float32), 1: np.zeros(20, dtype=np.float32)},
            rewards,
            {0: True, 1: True},
            {
                "reward_details": details,
                "reward_diagnostics": diagnostics,
                "terminated": {0: True, 1: True},
                "truncated": {0: False, 1: False},
                "active_mask": {0: False, 1: False},
                "episode_end_reason": "mission_timeout",
                "offroad_terminated": {0: False, 1: False},
                "first_offroad_evidence": {0: None, 1: None},
                "max_offroad_consecutive_ticks": {0: 0, 1: 0},
                "max_footprint_outside_fraction": {0: 0.0, 1: 0.0},
            },
        )


class R6ContractBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_experiment_contract(
            TRAINING_PATH, project_dir=PROJECT_DIR
        )
        cls.diag50 = load_experiment_contract(
            DIAG50_PATH, project_dir=PROJECT_DIR
        )
        cls.c0 = load_c0_contract(C0_PATH, project_dir=PROJECT_DIR)
        cls.go_hold = load_go_hold_contract(
            GO_HOLD_PATH, project_dir=PROJECT_DIR
        )

    def test_contract_chain_is_frozen_and_hash_bound(self):
        training = self.training["payload"]
        c0 = self.c0["payload"]
        go_hold = self.go_hold["payload"]
        self.assertEqual(training["source_mode"], "fresh")
        self.assertEqual(
            c0["checkpoint"]["required_training_contract_sha256"],
            self.training["sha256"],
        )
        self.assertEqual(
            go_hold["bindings"]["training_contract_sha256"],
            self.training["sha256"],
        )
        self.assertEqual(
            go_hold["bindings"]["c0_contract_sha256"], self.c0["sha256"]
        )
        self.assertEqual(go_hold["behavioral_health"]["collapse_scope"], "any_agent")

    def test_r6_scientific_specs_are_exact_and_time_aware(self):
        training = self.training["payload"]
        args = training["trainer_args"]
        self.assertEqual(args["reward_version"], "v4")
        self.assertEqual(args["gamma"], 0.999)
        self.assertEqual(args["max_steps"], 500)
        self.assertEqual(args["curriculum_protocol"], "mastery_v2")
        self.assertEqual(training["reward_protocol"], reward_v4_spec())
        self.assertEqual(training["stall_protocol"], stall_protocol_spec())
        self.assertEqual(training["mastery_protocol"], mastery_protocol_spec())
        self.assertEqual(
            training["policy_distribution_protocol"],
            policy_distribution_spec(),
        )
        self.assertEqual(training["offroad_protocol"], r6_offroad_protocol_spec())
        self.assertEqual(observation_dim_for_reward("v4"), 20)
        versions = checkpoint_protocol_versions("v4", 20.0)
        self.assertEqual(versions["reward"], "v4")
        self.assertIn("policy_distribution", versions)
        self.assertIn("curriculum", versions)
        self.assertIn("stall", versions)

    def test_diag50_contract_is_fresh_r6_and_cannot_enter_formal_selection(self):
        formal = self.training["payload"]
        diagnostic = self.diag50["payload"]
        self.assertEqual(diagnostic["source_mode"], "fresh")
        self.assertEqual(diagnostic["contract_id"], "P0_DIAG_MAPPO_TOWN05_R6_U50_V1")
        self.assertIn(50, diagnostic["segment_stop_updates"])
        self.assertEqual(
            diagnostic["trainer_args"], formal["trainer_args"]
        )
        for protocol in (
            "offroad_protocol",
            "reward_protocol",
            "stall_protocol",
            "mastery_protocol",
            "policy_distribution_protocol",
        ):
            self.assertEqual(diagnostic[protocol], formal[protocol])
        self.assertNotEqual(
            self.c0["payload"]["checkpoint"]["required_training_contract_id"],
            diagnostic["contract_id"],
        )
        self.assertIn(
            "configs/base_mappo/p0_diag_town05_r6_u50_v1.json",
            diagnostic["critical_source_paths"],
        )
        self.assertIn(
            "utils/validate_r6_live.py", diagnostic["critical_source_paths"]
        )

    def test_failed_u50_run_is_not_a_source_or_selection_candidate(self):
        combined = json.dumps(
            [
                self.training["payload"],
                self.c0["payload"],
                self.go_hold["payload"],
            ],
            sort_keys=True,
        )
        self.assertNotIn("P0_BASE_MAPPO_Town05V3_s42_20260814_125226", combined)
        self.assertNotIn("ckpt_update_000050", combined)

    def test_c0_observation_boundary_accepts_20_and_rejects_19(self):
        actual = _observation(
            np.zeros(20, dtype=np.float32),
            agent_id=0,
            expected_dim=20,
            context="R6 test",
        )
        self.assertEqual(actual.shape, (20,))
        with self.assertRaises(C0ProtocolError):
            _observation(
                np.zeros(19, dtype=np.float32),
                agent_id=0,
                expected_dim=20,
                context="R6 test",
            )

    def test_runtime_metadata_uses_r6_offroad_identity(self):
        env = types.SimpleNamespace(
            reward_version="v4",
            reward_gamma=0.999,
            client=types.SimpleNamespace(
                get_client_version=lambda: "0.9.16",
                get_server_version=lambda: "0.9.16",
            ),
        )
        metadata = runtime_metadata_with_carla(env)
        self.assertEqual(
            metadata["offroad_protocol"], r6_offroad_protocol_spec()
        )

    def test_r6_c0_runner_records_terminal_mission_timeout(self):
        contract = self.c0["payload"]
        case = expand_profile_cases(
            contract, "smoke", project_dir=PROJECT_DIR
        )[0]
        record = _run_case(
            env=_OneStepR6MissionTimeoutEnv(),
            actors={0: Actor(obs_dim=20, act_dim=2), 1: Actor(obs_dim=20, act_dim=2)},
            case=case,
            contract=contract,
            eval_id="P0_BASE_C0_smoke_u000005_20260818_000000",
            profile="smoke",
            checkpoint_sha256="c" * 64,
        )
        self.assertEqual(record["outcome"], "timeout")
        self.assertEqual(record["episode_end_reason"], "mission_timeout")
        for agent in record["agents"].values():
            self.assertTrue(agent["final_terminated"])
            self.assertFalse(agent["final_truncated"])

    def test_schema_v6_checkpoint_is_formally_eligible_for_only_r6_contract(self):
        contract = self.c0["payload"]
        run_id = "P0_BASE_MAPPO_Town05R6_s42_20260818_000000"
        versions = checkpoint_protocol_versions("v4", 20.0)
        normalizer = ValueNormalizer()
        payload = {
            "checkpoint_schema_version": contract["checkpoint"]["schema_version"],
            "training_protocol_version": contract["checkpoint"][
                "training_protocol_version"
            ],
            "protocol_versions": versions,
            "action_spec": action_protocol_spec(),
            "checkpoint_kind": "periodic",
            "update": 5,
            "actors": {
                agent_id: Actor(
                    obs_dim=20,
                    act_dim=2,
                    hidden=128,
                    init_log_std=-0.5,
                    min_log_std=-2.0,
                    max_log_std=0.0,
                ).state_dict()
                for agent_id in (0, 1)
            },
            "model_spec": {
                "obs_dim": 20,
                "action_dim": 2,
                "state_dim": 40,
                "num_agents": 2,
                "agent_ids": [0, 1],
                "hidden_dim": 128,
                "critic_type": "AgentConditionedCritic",
                "critic_output_semantics": "normalized_value",
            },
            "value_normalizer": normalizer.state_dict(),
            "critic_spec": normalizer.spec(),
            "training_config": {"gamma": 0.999},
            "lr_schedule": {"decay_horizon": 160},
            "rng_state": capture_rng_state(),
            "last_metrics": {
                "update": 5,
                "metrics_protocol_version": versions["metrics"],
                "run_id": run_id,
                "scalars": {"loss": 0.1},
            },
            "run_metadata": {
                "run_id": run_id,
                "seed": 42,
                "origin_git_sha": "a" * 40,
                "last_git_sha": "a" * 40,
                "code_fingerprint": "b" * 64,
                "experiment_contract_id": contract["checkpoint"][
                    "required_training_contract_id"
                ],
                "experiment_contract_sha256": contract["checkpoint"][
                    "required_training_contract_sha256"
                ],
                "runtime": {
                    "environment_reward_gamma": 0.999,
                    "offroad_protocol": r6_offroad_protocol_spec(),
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "r6_ckpt_000005.pt"
            save_checkpoint_file(payload, path)
            result = load_and_validate_c0_checkpoint(path, contract=contract)
            historical = copy.deepcopy(payload)
            historical["checkpoint_schema_version"] = (
                OFFROAD_CHECKPOINT_SCHEMA_VERSION
            )
            historical["training_protocol_version"] = (
                OFFROAD_TRAINING_PROTOCOL_VERSION
            )
            historical["protocol_versions"]["metrics"] = (
                OFFROAD_METRICS_PROTOCOL_VERSION
            )
            historical["last_metrics"]["metrics_protocol_version"] = (
                OFFROAD_METRICS_PROTOCOL_VERSION
            )
            historical_path = Path(temp_dir) / "historical_u50_shape.pt"
            save_checkpoint_file(historical, historical_path)
            with self.assertRaisesRegex(CheckpointProtocolError, "schema mismatch"):
                load_and_validate_c0_checkpoint(
                    historical_path, contract=contract
                )
        self.assertTrue(result["formal_eligible"])
        self.assertEqual(result["update"], 5)


if __name__ == "__main__":
    unittest.main()
