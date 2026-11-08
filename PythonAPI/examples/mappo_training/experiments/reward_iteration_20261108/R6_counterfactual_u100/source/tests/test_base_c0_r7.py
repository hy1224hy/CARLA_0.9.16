import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from environment.offroad_protocol import r6_offroad_protocol_spec
from environment.road_legality_protocol import road_legality_protocol_spec
from evaluation.base_c0_protocol import (
    C0_MANIFEST_VERSION,
    C0_RESOLVED_CONFIG_VERSION,
    R7_OUTCOMES,
    R7_SUMMARY_CSV_FIELDS,
    TOWN05_R6_C0_PROTOCOL_VERSION,
    TOWN05_R7_C0_EPISODE_RECORD_VERSION,
    TOWN05_R7_C0_PROTOCOL_VERSION,
    append_jsonl,
    atomic_write_csv,
    build_summary,
    classify_outcome,
    expand_profile_cases,
    load_and_validate_c0_checkpoint,
    load_c0_contract,
    summary_csv_rows,
    validate_episode_record,
)
from evaluation.evaluate_base_policy_c0 import (
    FORMAL_EXECUTABLE_C0_PROTOCOL_VERSIONS,
    _run_case,
)
from evaluation.validate_base_c0_run import validate_c0_run
from models.actor_critic import Actor
from tests.test_checkpoint_resume import _checkpoint_payload
from training.action_protocol import (
    action_protocol_spec,
    policy_action_diagnostics,
)
from training.checkpoint_protocol import (
    R7_CHECKPOINT_SCHEMA_VERSION,
    R7_TRAINING_PROTOCOL_VERSION,
    atomic_write_json,
    save_checkpoint_file,
    sha256_file,
)
from training.critic_diagnostics import build_critic_probe_batch
from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    POLICY_ACTION_RATE_NAMES,
    R7_METRICS_PROTOCOL_VERSION,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
    env_action_diagnostics,
)
from training.train_no_rllib_mappo_robust_v2 import checkpoint_protocol_versions
from utils.validate_r7_road_legality_live import DIRECTION_SCENARIOS, ROAD_TICKS


PROJECT_DIR = Path(__file__).resolve().parents[1]
R7_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "evaluation" / "p0_base_c0_town05_r7_v5.json"
)
R6_CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "evaluation" / "p0_base_c0_town05_r6_v4.json"
)


def _stats(value=0.0):
    return {"count": 1, "min": value, "max": value, "mean": value, "last": value}


def _road_evidence(*, violation=False):
    return {
        "classification": "opposing_lane" if violation else "reference_lane",
        "reference_lane": [44, -1],
        "legal_lane_change": False,
        "lane_deviation_exempted": False,
        "violation_reason": "opposing_lane" if violation else "",
        "violation_raw": violation,
        "consecutive_violation_ticks": 5 if violation else 0,
        "violation_confirmed": violation,
        "legal_transition_grace_remaining": 0,
        "success_eligible": not violation,
        "update_reference_lane": not violation,
        "same_reference_lane": not violation,
        "adjacent_to_reference": False,
        "lane_change_permitted": False,
        "current_to_reference_cosine": -1.0 if violation else 1.0,
        "vehicle_to_current_cosine": 1.0,
        "in_junction": False,
    }


def _agent_record(*, road_terminated=False, truncated=False):
    components = {name: 0.0 for name in REWARD_COMPONENTS["v5"]}
    if road_terminated:
        components["road_rule"] = -50.0
    diagnostics = {name: _stats() for name in REWARD_DIAGNOSTICS["v5"]}
    if road_terminated:
        diagnostics["road_violation_raw"] = _stats(1.0)
        diagnostics["road_violation_consecutive_ticks"] = _stats(5.0)
        diagnostics["road_violation_terminated"] = _stats(1.0)
        diagnostics["road_success_eligible"] = _stats(0.0)
        diagnostics["road_current_to_reference_cosine"] = _stats(-1.0)
    else:
        diagnostics["road_same_reference_lane"] = _stats(1.0)
        diagnostics["road_success_eligible"] = _stats(1.0)
        diagnostics["road_current_to_reference_cosine"] = _stats(1.0)
        diagnostics["road_vehicle_to_current_cosine"] = _stats(1.0)
    env_rates = env_action_diagnostics([0.0, 0.0, 1.0], context="fixture")
    policy_rates = policy_action_diagnostics([0.0, 0.0])
    return {
        "episode_return": -50.0 if road_terminated else 0.0,
        "active_decision_steps": 1,
        "reached_target": False,
        "collision": False,
        "final_done": True,
        "final_terminated": not truncated,
        "final_truncated": truncated,
        "reward_component_sums": components,
        "reward_residual_max_abs": 0.0,
        "diagnostics": diagnostics,
        "actions": {
            "count": 1,
            "throttle_mean": 0.0,
            "steer_mean": 0.0,
            "brake_mean": 1.0,
            **{f"{name}_rate": env_rates[name] for name in ACTION_RATE_NAMES},
            "policy_signed_longitudinal_mean": 0.0,
            "policy_steer_mean": 0.0,
            **{
                f"{name}_rate": policy_rates[name]
                for name in POLICY_ACTION_RATE_NAMES
            },
        },
        "offroad_terminated": False,
        "first_offroad_evidence": None,
        "max_offroad_consecutive_ticks": 0,
        "max_footprint_outside_fraction": 0.0,
        "road_rule_terminated": road_terminated,
        "road_violation_confirmed": road_terminated,
        "road_legality": _road_evidence(violation=road_terminated),
        "lifecycle_transition": {
            "pre_active": True,
            "post_active": False,
            "done": True,
            "terminated": not truncated,
            "truncated": truncated,
        },
    }


def _record(case, *, eval_id="eval", checkpoint_sha="c" * 64):
    agents = {
        "0": _agent_record(road_terminated=True),
        "1": _agent_record(truncated=True),
    }
    return {
        "record_version": TOWN05_R7_C0_EPISODE_RECORD_VERSION,
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
        "started_at": "2026-08-24T00:00:00+00:00",
        "finished_at": "2026-08-24T00:00:01+00:00",
        "episode_length": 1,
        "episode_end_reason": "cooperative_road_violation",
        "outcome": "road_violation",
        "agents": agents,
        "team": {"mean_episode_return": -25.0, "total_active_decision_steps": 2},
    }


def _r7_checkpoint(contract):
    payload = _checkpoint_payload()
    versions = checkpoint_protocol_versions("v5", 20.0)
    payload.update({
        "checkpoint_schema_version": R7_CHECKPOINT_SCHEMA_VERSION,
        "training_protocol_version": R7_TRAINING_PROTOCOL_VERSION,
        "protocol_versions": versions,
        "checkpoint_kind": "periodic",
        "update": 5,
        "actors": {
            0: Actor(obs_dim=20, act_dim=2, hidden=128).state_dict(),
            1: Actor(obs_dim=20, act_dim=2, hidden=128).state_dict(),
        },
        "action_spec": action_protocol_spec(),
        "model_spec": {
            "obs_dim": 20,
            "action_dim": 2,
            "state_dim": 12,
            "num_agents": 2,
            "agent_ids": [0, 1],
            "hidden_dim": 128,
            "critic_type": "AgentConditionedCritic",
            "critic_output_semantics": "normalized_value",
        },
        "training_config": {
            **payload["training_config"],
            "gamma": 0.999,
            "reward_version": "v5",
        },
        "critic_probe_batch": build_critic_probe_batch(
            states=[[float(row + col) for col in range(12)] for row in range(4)],
            agent_ids=[0, 1, 0, 1],
            targets=[0.0, 1.0, -1.0, 2.0],
            maximum_rows=4,
        ),
        "total_road_violations": 0,
    })
    payload["run_metadata"].update({
        "experiment_contract_id": contract["checkpoint"]["required_training_contract_id"],
        "experiment_contract_sha256": contract["checkpoint"]["required_training_contract_sha256"],
        "runtime": {
            "environment_reward_gamma": 0.999,
            "offroad_protocol": r6_offroad_protocol_spec(),
            "road_legality_protocol": road_legality_protocol_spec(),
        },
    })
    payload["last_metrics"].update({
        "update": 5,
        "metrics_protocol_version": R7_METRICS_PROTOCOL_VERSION,
        "run_id": payload["run_metadata"]["run_id"],
    })
    return payload


class _TrafficManager:
    def set_random_device_seed(self, seed):
        self.seed = seed


class _OneStepRoadViolationEnv:
    def __init__(self):
        self.traffic_manager = _TrafficManager()
        self.reached_target = {0: False, 1: False}

    def apply_scenario_config(self, config):
        self.config = config

    def reset(self):
        return (
            {0: np.zeros(20, dtype=np.float32), 1: np.zeros(20, dtype=np.float32)},
            {"active_mask": {0: True, 1: True}, "episode_end_reason": "running"},
        )

    def step(self, actions):
        details = {0: {name: 0.0 for name in REWARD_COMPONENTS["v5"]},
                   1: {name: 0.0 for name in REWARD_COMPONENTS["v5"]}}
        details[0]["road_rule"] = -50.0
        diagnostics = {0: {name: 0.0 for name in REWARD_DIAGNOSTICS["v5"]},
                       1: {name: 0.0 for name in REWARD_DIAGNOSTICS["v5"]}}
        diagnostics[0].update({
            "road_violation_raw": 1.0,
            "road_violation_consecutive_ticks": 5.0,
            "road_violation_terminated": 1.0,
            "road_current_to_reference_cosine": -1.0,
        })
        diagnostics[1].update({
            "road_same_reference_lane": 1.0,
            "road_success_eligible": 1.0,
            "road_current_to_reference_cosine": 1.0,
            "road_vehicle_to_current_cosine": 1.0,
        })
        return (
            {0: np.zeros(20, dtype=np.float32), 1: np.zeros(20, dtype=np.float32)},
            {0: -50.0, 1: 0.0},
            {0: True, 1: True},
            {
                "reward_details": details,
                "reward_diagnostics": diagnostics,
                "terminated": {0: True, 1: False},
                "truncated": {0: False, 1: True},
                "active_mask": {0: False, 1: False},
                "episode_end_reason": "cooperative_road_violation",
                "offroad_terminated": {0: False, 1: False},
                "first_offroad_evidence": {0: None, 1: None},
                "max_offroad_consecutive_ticks": {0: 0, 1: 0},
                "max_footprint_outside_fraction": {0: 0.0, 1: 0.0},
                "road_violation_confirmed": {0: True, 1: False},
                "road_rule_terminated": {0: True, 1: False},
                "road_legality": {0: _road_evidence(violation=True), 1: _road_evidence()},
            },
        )


class R7C0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_record = load_c0_contract(
            R7_CONTRACT_PATH, project_dir=PROJECT_DIR
        )
        cls.contract = cls.contract_record["payload"]
        cls.case = expand_profile_cases(
            cls.contract, "smoke", project_dir=PROJECT_DIR
        )[0]

    def test_r7_contract_is_formal_and_does_not_mutate_r6_identity(self):
        self.assertIn(TOWN05_R7_C0_PROTOCOL_VERSION, FORMAL_EXECUTABLE_C0_PROTOCOL_VERSIONS)
        self.assertEqual(self.contract["protocol_version"], TOWN05_R7_C0_PROTOCOL_VERSION)
        self.assertEqual(tuple(name.removesuffix("_reasons") for name in self.contract["outcomes"] if name.endswith("_reasons")), R7_OUTCOMES)
        r6 = load_c0_contract(R6_CONTRACT_PATH, project_dir=PROJECT_DIR)["payload"]
        self.assertEqual(r6["protocol_version"], TOWN05_R6_C0_PROTOCOL_VERSION)
        self.assertIn("other_reasons", r6["outcomes"])
        self.assertNotIn("road_violation_reasons", r6["outcomes"])

    def test_five_outcomes_are_mutually_exclusive_and_explicit(self):
        outcomes = self.contract["outcomes"]
        expected = {
            "cooperative_success": "success",
            "cooperative_collision": "collision",
            "cooperative_offroad": "offroad",
            "cooperative_road_violation": "road_violation",
            "mission_timeout": "timeout",
        }
        for reason, outcome in expected.items():
            self.assertEqual(classify_outcome(reason, outcomes), outcome)

    def test_live_validator_registers_all_four_directions_and_debounce(self):
        self.assertEqual(
            set(DIRECTION_SCENARIOS),
            {"northbound", "southbound", "eastbound", "westbound"},
        )
        self.assertEqual(ROAD_TICKS, 5)

    def test_r7_episode_and_summary_preserve_road_evidence(self):
        record = _record(self.case)
        validate_episode_record(
            record,
            eval_id="eval",
            profile="smoke",
            checkpoint_sha256="c" * 64,
            expected_case=self.case,
        )
        summary = build_summary(
            [record], eval_id="eval", profile="smoke", checkpoint_sha256="c" * 64
        )
        self.assertEqual(summary["overall"]["outcome_counts"]["road_violation"], 1)
        self.assertEqual(summary["overall"]["outcome_counts"]["offroad"], 0)
        rows = summary_csv_rows(summary)
        self.assertEqual(tuple(rows[0]), R7_SUMMARY_CSV_FIELDS)
        self.assertEqual(rows[0]["road_violation_count"], "1")

        tampered = copy.deepcopy(record)
        tampered["agents"]["0"]["road_legality"]["success_eligible"] = True
        with self.assertRaisesRegex(Exception, "raw violation cannot be success-eligible"):
            validate_episode_record(
                tampered,
                eval_id="eval",
                profile="smoke",
                checkpoint_sha256="c" * 64,
                expected_case=self.case,
            )

    def test_r7_checkpoint_requires_protocols_and_critic_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ckpt_000005.pt"
            payload = _r7_checkpoint(self.contract)
            save_checkpoint_file(payload, path)
            result = load_and_validate_c0_checkpoint(path, contract=self.contract)
            self.assertTrue(result["formal_eligible"])

            broken = copy.deepcopy(payload)
            broken.pop("critic_probe_batch")
            broken_path = Path(temp_dir) / "broken.pt"
            save_checkpoint_file(broken, broken_path)
            with self.assertRaisesRegex(Exception, "critic_probe_batch"):
                load_and_validate_c0_checkpoint(broken_path, contract=self.contract)

    def test_runner_emits_r7_road_rule_record(self):
        record = _run_case(
            env=_OneStepRoadViolationEnv(),
            actors={0: Actor(obs_dim=20, act_dim=2), 1: Actor(obs_dim=20, act_dim=2)},
            case=self.case,
            contract=self.contract,
            eval_id="eval",
            profile="smoke",
            checkpoint_sha256="c" * 64,
        )
        self.assertEqual(record["outcome"], "road_violation")
        self.assertTrue(record["agents"]["0"]["road_rule_terminated"])
        self.assertEqual(
            record["agents"]["0"]["road_legality"]["classification"],
            "opposing_lane",
        )

    def test_offline_validator_reconstructs_r7_five_outcome_artifacts(self):
        eval_id = "P0_BASE_C0_smoke_u000005_20260824_120000"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / eval_id
            output_dir.mkdir()
            checkpoint_path = root / "ckpt_000005.pt"
            checkpoint_sha = save_checkpoint_file(
                _r7_checkpoint(self.contract), checkpoint_path
            )
            checkpoint_record = load_and_validate_c0_checkpoint(
                checkpoint_path, contract=self.contract
            )
            record = _record(
                self.case, eval_id=eval_id, checkpoint_sha=checkpoint_sha
            )
            summary = build_summary(
                [record],
                eval_id=eval_id,
                profile="smoke",
                checkpoint_sha256=checkpoint_sha,
            )
            outputs = self.contract["outputs"]
            episodes_path = output_dir / outputs["episodes_jsonl"]
            summary_path = output_dir / outputs["summary_json"]
            csv_path = output_dir / outputs["summary_csv"]
            append_jsonl(episodes_path, record)
            atomic_write_json(summary, summary_path)
            atomic_write_csv(csv_path, summary_csv_rows(summary))
            resolved = {
                "resolved_config_version": C0_RESOLVED_CONFIG_VERSION,
                "created_at": "2026-08-24T12:00:00+00:00",
                "eval_id": eval_id,
                "profile": "smoke",
                "contract_id": self.contract["contract_id"],
                "contract_path": str(R7_CONTRACT_PATH.resolve()),
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
                "command": "python -m evaluation.evaluate_base_policy_c0 <r7-test>",
                "runtime": {
                    "carla_python_module": "test",
                    "carla_client_version": "0.9.16-test",
                    "carla_server_version": "0.9.16-test",
                },
                "case_count": 1,
                "cases": [
                    {
                        key: value
                        for key, value in self.case.items()
                        if key != "scenario_config"
                    }
                ],
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
                "formal_eligible": True,
                "status": "PASS",
                "completed_cases": 1,
                "total_cases": 1,
                "created_at": "2026-08-24T12:00:00+00:00",
                "updated_at": "2026-08-24T12:00:01+00:00",
                "post_cleanup": {
                    "vehicle_actor_count": 0,
                    "sensor_actor_count": 0,
                    "world_timing_restored": True,
                },
                "cleanup_error": "",
                "error": "",
                "episodes_sha256": sha256_file(episodes_path),
                "summary_sha256": sha256_file(summary_path),
                "summary_csv_sha256": sha256_file(csv_path),
            }
            atomic_write_json(
                manifest, output_dir / outputs["manifest"]
            )
            report = validate_c0_run(
                output_dir,
                expected_contract=R7_CONTRACT_PATH,
                expected_checkpoint=checkpoint_path,
                expected_profile="smoke",
                expected_eval_id=eval_id,
                write_report=False,
            )
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
    append_jsonl,
    atomic_write_csv,
