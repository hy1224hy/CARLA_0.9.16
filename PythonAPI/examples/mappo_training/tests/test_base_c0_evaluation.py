import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.actor_critic import Actor
from evaluation.base_c0_protocol import (
    C0_EPISODE_RECORD_VERSION,
    C0_MANIFEST_VERSION,
    C0_RESOLVED_CONFIG_VERSION,
    C0ProtocolError,
    append_jsonl,
    atomic_write_csv,
    build_summary,
    compare_repeatability_records,
    ensure_completed_prefix,
    expand_profile_cases,
    load_and_validate_c0_checkpoint,
    load_c0_contract,
    summary_csv_rows,
    validate_c0_contract,
    validate_episode_record,
)
from evaluation.validate_base_c0_run import validate_c0_run
from evaluation.evaluate_base_policy_c0 import _run_case
from training.checkpoint_protocol import (
    ACTION_PROTOCOL_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LIFECYCLE_PROTOCOL_VERSION,
    OBSERVATION_PROTOCOL_VERSION,
    TRAINING_PROTOCOL_VERSION,
    atomic_write_json,
    capture_rng_state,
    save_checkpoint_file,
    sha256_file,
)
from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    METRICS_PROTOCOL_VERSION,
    POLICY_ACTION_RATE_NAMES,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
    env_action_diagnostics,
)
from training.action_protocol import (
    action_protocol_spec,
    policy_action_diagnostics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_DIR / "configs" / "evaluation" / "p0_base_c0_eval_v2.json"


def _checkpoint_payload(contract, *, formal=True):
    run_id = "P0_BASE_MAPPO_Town02_s42_20260804_210000"
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "protocol_versions": {
            "observation": OBSERVATION_PROTOCOL_VERSION,
            "action": ACTION_PROTOCOL_VERSION,
            "lifecycle": LIFECYCLE_PROTOCOL_VERSION,
            "reward": "v2",
            "metrics": METRICS_PROTOCOL_VERSION,
        },
        "checkpoint_kind": "periodic" if formal else "latest",
        "update": 5,
        "actors": {
            0: Actor(act_dim=2).state_dict(),
            1: Actor(act_dim=2).state_dict(),
        },
        "action_spec": action_protocol_spec(),
        "model_spec": {
            "obs_dim": 19,
            "action_dim": 2,
            "state_dim": 38,
            "num_agents": 2,
            "agent_ids": [0, 1],
            "hidden_dim": 128,
            "critic_type": "AgentConditionedCritic",
        },
        "run_metadata": {
            "run_id": run_id,
            "seed": 42,
            "origin_git_sha": "a" * 40,
            "last_git_sha": "a" * 40,
            "code_fingerprint": "b" * 64,
            "experiment_contract_id": (
                contract["checkpoint"]["required_training_contract_id"] if formal else ""
            ),
            "experiment_contract_sha256": (
                contract["checkpoint"]["required_training_contract_sha256"] if formal else ""
            ),
        },
        "training_config": {},
        "lr_schedule": {"decay_horizon": 100},
        "rng_state": capture_rng_state(),
        "last_metrics": {
            "update": 5,
            "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
            "run_id": run_id,
            "scalars": {"loss": 0.1},
        },
    }


def _stats(value=0.0):
    return {"count": 1, "min": value, "max": value, "mean": value, "last": value}


def _agent_record(*, return_value=-0.05):
    components = {name: 0.0 for name in REWARD_COMPONENTS["v2"]}
    components["step"] = return_value
    action_rates = env_action_diagnostics([0.5, 0.0, 0.0], context="fixture")
    policy_action = [0.525, 0.0]
    policy_rates = policy_action_diagnostics(policy_action)
    return {
        "episode_return": return_value,
        "active_decision_steps": 1,
        "reached_target": False,
        "collision": False,
        "final_done": True,
        "final_terminated": False,
        "final_truncated": True,
        "reward_component_sums": components,
        "reward_residual_max_abs": 0.0,
        "diagnostics": {name: _stats() for name in REWARD_DIAGNOSTICS["v2"]},
        "actions": {
            "count": 1,
            "throttle_mean": 0.5,
            "steer_mean": 0.0,
            "brake_mean": 0.0,
            **{f"{name}_rate": action_rates[name] for name in ACTION_RATE_NAMES},
            "policy_signed_longitudinal_mean": policy_action[0],
            "policy_steer_mean": policy_action[1],
            **{
                f"{name}_rate": policy_rates[name]
                for name in POLICY_ACTION_RATE_NAMES
            },
        },
    }


def _episode_record(case, *, eval_id, checkpoint_sha):
    agents = {"0": _agent_record(), "1": _agent_record()}
    return {
        "record_version": C0_EPISODE_RECORD_VERSION,
        "eval_id": eval_id,
        "profile": "smoke",
        "case_id": case["case_id"],
        "case_ordinal": case["case_ordinal"],
        "scenario_name": case["scenario_name"],
        "scenario_sha256": case["scenario_sha256"],
        "scenario_family": case["scenario_family"],
        "scenario_difficulty": case["scenario_difficulty"],
        "episode_seed": case["episode_seed"],
        "checkpoint_sha256": checkpoint_sha,
        "started_at": "2026-08-04T00:00:00+00:00",
        "finished_at": "2026-08-04T00:00:01+00:00",
        "episode_length": 1,
        "episode_end_reason": "max_steps",
        "outcome": "timeout",
        "agents": agents,
        "team": {"mean_episode_return": -0.05, "total_active_decision_steps": 2},
    }


class _TrafficManager:
    def set_random_device_seed(self, seed):
        self.seed = seed


class _OneStepTimeoutEnv:
    def __init__(self):
        self.traffic_manager = _TrafficManager()
        self.reached_target = {0: False, 1: False}

    def apply_scenario_config(self, config):
        self.config = config

    def reset(self):
        return (
            {0: np.zeros(19, dtype=np.float32), 1: np.zeros(19, dtype=np.float32)},
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
            components = {name: 0.0 for name in REWARD_COMPONENTS["v2"]}
            components["step"] = -0.05
            details[agent_id] = components
            diagnostics[agent_id] = {name: 0.0 for name in REWARD_DIAGNOSTICS["v2"]}
            rewards[agent_id] = -0.05
        return (
            {0: np.zeros(19, dtype=np.float32), 1: np.zeros(19, dtype=np.float32)},
            rewards,
            {0: True, 1: True},
            {
                "reward_details": details,
                "reward_diagnostics": diagnostics,
                "terminated": {0: False, 1: False},
                "truncated": {0: True, 1: True},
                "active_mask": {0: False, 1: False},
                "episode_end_reason": "max_steps",
            },
        )


class C0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_c0_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)

    def test_frozen_profiles_have_expected_cardinality_and_holdout_hashes(self):
        contract = self.record["payload"]
        counts = {
            name: len(expand_profile_cases(contract, name, project_dir=PROJECT_DIR))
            for name in ("smoke", "gate", "validation")
        }
        self.assertEqual(counts, {"smoke": 1, "gate": 16, "validation": 24})
        gate = expand_profile_cases(contract, "gate", project_dir=PROJECT_DIR)
        validation = expand_profile_cases(contract, "validation", project_dir=PROJECT_DIR)
        self.assertFalse(
            {case["scenario_sha256"] for case in gate}
            & {case["scenario_sha256"] for case in validation}
        )
        drifted = copy.deepcopy(contract)
        drifted["profiles"]["validation"]["scenario_pool_sha256"] = "0" * 64
        with self.assertRaisesRegex(C0ProtocolError, "fingerprint drift"):
            validate_c0_contract(drifted, project_dir=PROJECT_DIR)

    def test_resume_evidence_must_be_an_exact_case_prefix(self):
        cases = expand_profile_cases(self.record["payload"], "validation", project_dir=PROJECT_DIR)
        ensure_completed_prefix([{"case_id": cases[0]["case_id"]}], cases)
        with self.assertRaisesRegex(C0ProtocolError, "exact prefix"):
            ensure_completed_prefix([{"case_id": cases[1]["case_id"]}], cases)


class C0CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_c0_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)["payload"]

    def test_formal_checkpoint_requires_training_contract_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "ckpt_000005.pt"
            save_checkpoint_file(_checkpoint_payload(self.contract), checkpoint_path)
            result = load_and_validate_c0_checkpoint(checkpoint_path, contract=self.contract)
            self.assertTrue(result["formal_eligible"])
            self.assertEqual(result["update"], 5)

            diagnostic_path = Path(temp_dir) / "latest.pt"
            save_checkpoint_file(_checkpoint_payload(self.contract, formal=False), diagnostic_path)
            with self.assertRaisesRegex(C0ProtocolError, "not eligible for formal"):
                load_and_validate_c0_checkpoint(diagnostic_path, contract=self.contract)
            diagnostic = load_and_validate_c0_checkpoint(
                diagnostic_path,
                contract=self.contract,
                allow_diagnostic_checkpoint=True,
            )
            self.assertFalse(diagnostic["formal_eligible"])
            self.assertGreaterEqual(len(diagnostic["formal_ineligibility_reasons"]), 1)

            malformed = _checkpoint_payload(self.contract)
            malformed["actors"][0].pop("mu_head.bias")
            malformed_path = Path(temp_dir) / "malformed.pt"
            save_checkpoint_file(malformed, malformed_path)
            with self.assertRaisesRegex(C0ProtocolError, "actor state"):
                load_and_validate_c0_checkpoint(malformed_path, contract=self.contract)


class C0EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.contract_record = load_c0_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)
        self.contract = self.contract_record["payload"]
        self.case = expand_profile_cases(self.contract, "smoke", project_dir=PROJECT_DIR)[0]

    def test_episode_reward_components_must_reconstruct_return(self):
        record = _episode_record(self.case, eval_id="eval", checkpoint_sha="c" * 64)
        validate_episode_record(
            record,
            eval_id="eval",
            profile="smoke",
            checkpoint_sha256="c" * 64,
            expected_case=self.case,
        )
        broken = copy.deepcopy(record)
        broken["agents"]["0"]["reward_component_sums"]["step"] = 0.0
        with self.assertRaisesRegex(C0ProtocolError, "conserve"):
            validate_episode_record(
                broken,
                eval_id="eval",
                profile="smoke",
                checkpoint_sha256="c" * 64,
                expected_case=self.case,
            )

    def test_action_rate_evidence_enforces_decomposition_and_exclusivity(self):
        record = _episode_record(self.case, eval_id="eval", checkpoint_sha="c" * 64)

        broken_boundary = copy.deepcopy(record)
        broken_boundary["agents"]["0"]["actions"]["brake_boundary_rate"] = 0.5
        with self.assertRaisesRegex(C0ProtocolError, "boundary decomposition"):
            validate_episode_record(
                broken_boundary,
                eval_id="eval",
                profile="smoke",
                checkpoint_sha256="c" * 64,
                expected_case=self.case,
            )

        conflict = copy.deepcopy(record)
        action_stats = conflict["agents"]["0"]["actions"]
        action_stats["throttle_mode_rate"] = 0.0
        action_stats["longitudinal_conflict_rate"] = 1.0
        with self.assertRaisesRegex(C0ProtocolError, "mutually exclusive"):
            validate_episode_record(
                conflict,
                eval_id="eval",
                profile="smoke",
                checkpoint_sha256="c" * 64,
                expected_case=self.case,
            )

    def test_repeatability_uses_frozen_discrete_signature(self):
        left = _episode_record(self.case, eval_id="left", checkpoint_sha="c" * 64)
        right = copy.deepcopy(left)
        right["eval_id"] = "right"
        result = compare_repeatability_records([left], [right], contract=self.contract)
        self.assertEqual(result["status"], "PASS")
        right["episode_length"] = 2
        result = compare_repeatability_records([left], [right], contract=self.contract)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["mismatches"][0]["case_id"], self.case["case_id"])

    def test_simulator_runner_uses_active_agents_and_deterministic_actor_mean(self):
        actors = {0: Actor(act_dim=2), 1: Actor(act_dim=2)}
        first = _run_case(
            env=_OneStepTimeoutEnv(),
            actors=actors,
            case=self.case,
            contract=self.contract,
            eval_id="eval",
            profile="smoke",
            checkpoint_sha256="c" * 64,
        )
        second = _run_case(
            env=_OneStepTimeoutEnv(),
            actors=actors,
            case=self.case,
            contract=self.contract,
            eval_id="eval",
            profile="smoke",
            checkpoint_sha256="c" * 64,
        )
        self.assertEqual(first["outcome"], "timeout")
        self.assertEqual(first["episode_length"], 1)
        self.assertEqual(first["agents"]["0"]["actions"], second["agents"]["0"]["actions"])
        self.assertEqual(first["agents"]["1"]["actions"], second["agents"]["1"]["actions"])

    def test_offline_validator_reconstructs_summary_and_detects_tampering(self):
        eval_id = "P0_BASE_C0_smoke_u000005_20260804_210000"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / eval_id
            output_dir.mkdir()
            checkpoint_path = Path(temp_dir) / "latest.pt"
            checkpoint_sha = save_checkpoint_file(
                _checkpoint_payload(self.contract, formal=False), checkpoint_path
            )
            checkpoint_record = load_and_validate_c0_checkpoint(
                checkpoint_path,
                contract=self.contract,
                allow_diagnostic_checkpoint=True,
            )
            record = _episode_record(self.case, eval_id=eval_id, checkpoint_sha=checkpoint_sha)
            summary = build_summary(
                [record], eval_id=eval_id, profile="smoke", checkpoint_sha256=checkpoint_sha
            )
            outputs = self.contract["outputs"]
            episodes_path = output_dir / outputs["episodes_jsonl"]
            summary_path = output_dir / outputs["summary_json"]
            summary_csv_path = output_dir / outputs["summary_csv"]
            append_jsonl(episodes_path, record)
            atomic_write_json(summary, summary_path)
            atomic_write_csv(summary_csv_path, summary_csv_rows(summary))
            resolved = {
                "resolved_config_version": C0_RESOLVED_CONFIG_VERSION,
                "created_at": "2026-08-04T00:00:00+00:00",
                "eval_id": eval_id,
                "profile": "smoke",
                "contract_id": self.contract["contract_id"],
                "contract_path": str(CONTRACT_PATH.resolve()),
                "contract_sha256": self.contract_record["sha256"],
                "contract_canonical_sha256": self.contract_record["canonical_sha256"],
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_update": 5,
                "checkpoint_formal_eligible": checkpoint_record["formal_eligible"],
                "checkpoint_formal_ineligibility_reasons": checkpoint_record[
                    "formal_ineligibility_reasons"
                ],
                "host": "127.0.0.1",
                "port": 2000,
                "device": "cpu",
                "command": "python -m evaluation.evaluate_base_policy_c0 <test>",
                "runtime": {
                    "carla_python_module": "test",
                    "carla_client_version": "0.9.16-test",
                    "carla_server_version": "0.9.16-test",
                },
                "case_count": 1,
                "cases": [{key: value for key, value in self.case.items() if key != "scenario_config"}],
                "git": {"commit": "test", "dirty": False, "dirty_paths": []},
            }
            resolved_path = output_dir / outputs["resolved_config"]
            atomic_write_json(resolved, resolved_path)
            manifest = {
                "manifest_version": C0_MANIFEST_VERSION,
                "eval_id": eval_id,
                "profile": "smoke",
                "contract_sha256": self.contract_record["sha256"],
                "resolved_config_sha256": sha256_file(resolved_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_update": 5,
                "formal_eligible": False,
                "status": "DIAGNOSTIC_PASS",
                "completed_cases": 1,
                "total_cases": 1,
                "created_at": "2026-08-04T00:00:00+00:00",
                "updated_at": "2026-08-04T00:00:01+00:00",
                "post_cleanup": {
                    "vehicle_actor_count": 0,
                    "sensor_actor_count": 0,
                    "synchronous_mode": False,
                    "fixed_delta_seconds": None,
                    "world_timing_restored": True,
                },
                "cleanup_error": "",
                "error": "",
                "episodes_sha256": sha256_file(episodes_path),
                "summary_sha256": sha256_file(summary_path),
                "summary_csv_sha256": sha256_file(summary_csv_path),
            }
            atomic_write_json(manifest, output_dir / outputs["manifest"])

            report = validate_c0_run(
                output_dir,
                expected_contract=CONTRACT_PATH,
                expected_checkpoint=checkpoint_path,
                expected_profile="smoke",
                expected_eval_id=eval_id,
                write_report=False,
            )
            self.assertEqual(report["status"], "PASS")

            tampered = copy.deepcopy(summary)
            tampered["overall"]["outcome_counts"]["timeout"] = 0
            atomic_write_json(tampered, summary_path)
            manifest["summary_sha256"] = sha256_file(summary_path)
            atomic_write_json(manifest, output_dir / outputs["manifest"])
            report = validate_c0_run(
                output_dir,
                expected_contract=CONTRACT_PATH,
                expected_checkpoint=checkpoint_path,
                write_report=False,
            )
            self.assertEqual(report["status"], "HOLD")
            self.assertIn("summary reconstructed", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
