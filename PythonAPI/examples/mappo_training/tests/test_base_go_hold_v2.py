import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.base_c0_protocol import read_jsonl
from evaluation.base_go_hold_protocol import (
    GO_HOLD_PROTOCOL_VERSION,
    GoHoldProtocolError,
    evaluate_behavioral_health,
    evaluate_milestone_decision,
    evaluate_training_health,
    load_go_hold_contract,
    summarize_c0_gate,
)
from evaluation.judge_base_go_hold import _load_previous_milestone_evidence


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_DIR / "configs" / "evaluation" / "p0_base_go_hold_v2.json"
)
FAMILIES = ("basic", "offset", "close", "rear")


def _metric_records(count: int, *, degenerate_updates=(), overrides=None):
    degenerate_updates = set(degenerate_updates)
    overrides = overrides or {}
    records = []
    for update in range(1, count + 1):
        degenerate = update in degenerate_updates
        scalars = {
            "critic/grad_clip_activation_rate": 1.0 if degenerate else 0.1,
            "critic/explained_variance": 0.0 if degenerate else 0.3,
            "critic/value_prediction_std": 0.01 if degenerate else 0.7,
            "critic/return_target_std": 1.0,
            "critic/value_error_abs_mean": 1.2 if degenerate else 0.4,
            "critic/normalized_return_target_std": 1.0,
            "critic/value_normalizer_std": 1.0,
            "critic/value_normalizer_update_count": float(update),
            "episode/count": 16.0,
            "reward_audit/missing_field_count": 0.0,
            "reward_audit/nonfinite_count": 0.0,
            "reward_audit/max_abs_step_residual": 0.0,
            "train/update_skipped": 0.0,
        }
        for agent_id in (0, 1):
            prefix = f"policy/agent_{agent_id}"
            scalars.update(
                {
                    f"{prefix}/approx_kl": 0.01,
                    f"{prefix}/clip_frac": 0.1,
                    f"{prefix}/grad_clip_activation_rate": 0.1,
                    f"{prefix}/log_std_min": -0.2,
                    f"{prefix}/log_std_max": 0.2,
                    f"{prefix}/preupdate_max_abs_ratio_error": 0.0,
                    f"action/agent_{agent_id}/longitudinal_conflict_rate": 0.0,
                }
            )
            for channel in ("throttle", "steer", "brake"):
                scalars[f"action/agent_{agent_id}/{channel}_saturation_rate"] = 0.1
        for tag, sequence in overrides.items():
            scalars[tag] = sequence[update - 1]
        records.append({"update": update, "scalars": scalars})
    return records


def _summary(
    *,
    success=0,
    collision=0,
    timeout=None,
    other=0,
    brake=0.4,
    speed=0.5,
    progress=5.0,
):
    case_count = 16
    if timeout is None:
        timeout = case_count - success - collision - other
    return {
        "case_count": case_count,
        "outcome_counts": {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "other": other,
        },
        "family_case_counts": {name: 4 for name in FAMILIES},
        "family_success_counts": {name: 0 for name in FAMILIES},
        "scenario_case_counts": {f"scenario_{index}": 1 for index in range(16)},
        "scenario_success_counts": {f"scenario_{index}": 0 for index in range(16)},
        "lane_invasion_episode_count": 0,
        "behavior": {
            "timeout_rate": timeout / case_count,
            "agents": {
                str(agent_id): {
                    "active_decision_steps": 16000,
                    "brake_mode_rate": brake,
                    "throttle_mode_rate": max(0.0, 1.0 - brake),
                    "coast_mode_rate": 0.0,
                    "mean_speed_mps": speed,
                    "mean_progress_m_per_episode": progress,
                }
                for agent_id in (0, 1)
            },
        },
    }


class GoHoldV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)
        cls.contract = cls.record["payload"]

    def test_contract_binds_r3_artifacts_and_freezes_false_hold_guard(self):
        contract = self.contract
        self.assertEqual(contract["protocol_version"], GO_HOLD_PROTOCOL_VERSION)
        self.assertEqual(
            contract["bindings"]["training_contract_id"],
            "P0_BASE_MAPPO_TOWN02_CRITIC_V4",
        )
        self.assertEqual(
            contract["bindings"]["c0_contract_id"],
            "P0_BASE_C0_TOWN02_CRITIC_V4",
        )
        behavior = contract["behavioral_health"]
        self.assertEqual(behavior["warning_from_update"], 5)
        self.assertEqual(behavior["hold_from_update"], 10)
        self.assertEqual(behavior["required_consecutive_milestones"], 2)
        self.assertFalse(
            contract["decision_policy"]["single_early_zero_success_can_hold"]
        )

    def test_previous_report_must_be_adjacent_and_bound_to_same_run(self):
        summary = _summary(brake=1.0, speed=0.01, progress=0.0)
        report = {
            "report_version": "p0_base_go_hold_decision_v2",
            "decision_id": "P0_BASE_GO_HOLD_u000005_20260811_000000",
            "mode": "milestone",
            "expected_update": 5,
            "contract": {
                "id": self.contract["contract_id"],
                "sha256": self.record["sha256"],
            },
            "training": {"run_id": "R3_TEST_RUN"},
            "c0": {"summary": summary},
            "result": {
                "decision": "CONTINUE_WITH_WARNING",
                "behavior_assessment": {"status": "WARNING"},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "go_hold_decision.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            evidence = _load_previous_milestone_evidence(
                path,
                contract_record=self.record,
                contract=self.contract,
                run_id="R3_TEST_RUN",
                current_update=10,
            )
            self.assertEqual(evidence["expected_update"], 5)
            self.assertEqual(evidence["c0_summary"], summary)

            wrong_run = copy.deepcopy(report)
            wrong_run["training"]["run_id"] = "OTHER_RUN"
            path.write_text(json.dumps(wrong_run), encoding="utf-8")
            with self.assertRaisesRegex(
                GoHoldProtocolError, "different training run"
            ):
                _load_previous_milestone_evidence(
                    path,
                    contract_record=self.record,
                    contract=self.contract,
                    run_id="R3_TEST_RUN",
                    current_update=10,
                )


class BehavioralHoldV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]

    def _decide(self, update, current, *, previous=None):
        health = evaluate_training_health(
            _metric_records(update), contract=self.contract
        )
        return evaluate_milestone_decision(
            update=update,
            training_health=health,
            c0_summary=current,
            repeatability_pass=True,
            contract=self.contract,
            previous_c0_summary=previous,
        )

    def test_zero_success_alone_never_triggers_early_hold(self):
        inexperienced_but_moving = _summary(
            success=0,
            timeout=16,
            brake=0.4,
            speed=0.5,
            progress=5.0,
        )
        result = self._decide(
            10, inexperienced_but_moving, previous=inexperienced_but_moving
        )
        self.assertNotEqual(result["decision"], "HOLD")
        self.assertFalse(result["behavior_assessment"]["collapsed_now"])

    def test_first_collapsed_milestone_warns_but_does_not_hold(self):
        collapsed = _summary(brake=1.0, speed=0.01, progress=0.0)
        result = self._decide(5, collapsed)
        self.assertEqual(result["decision"], "CONTINUE_WITH_WARNING")
        self.assertEqual(result["behavior_assessment"]["status"], "WARNING")

    def test_two_adjacent_collapsed_milestones_hold_with_reason_code(self):
        collapsed = _summary(brake=1.0, speed=0.01, progress=0.0)
        result = self._decide(10, collapsed, previous=collapsed)
        self.assertEqual(result["decision"], "HOLD")
        self.assertEqual(result["reason_codes"], ["BEHAVIORAL_COLLAPSE"])
        self.assertTrue(result["behavior_assessment"]["persistent_collapse"])

    def test_one_missing_signal_prevents_behavioral_hold(self):
        not_brake_dominant = _summary(brake=0.949, speed=0.01, progress=0.0)
        result = self._decide(
            10, not_brake_dominant, previous=not_brake_dominant
        )
        self.assertNotEqual(result["decision"], "HOLD")
        self.assertFalse(result["behavior_assessment"]["collapsed_now"])

    def test_missing_previous_evidence_is_an_integrity_hold_not_a_collapse_claim(self):
        collapsed = _summary(brake=1.0, speed=0.01, progress=0.0)
        result = self._decide(10, collapsed)
        self.assertEqual(result["decision"], "HOLD")
        self.assertEqual(result["reason_codes"], ["BEHAVIOR_HISTORY_MISSING"])
        self.assertFalse(result["behavior_assessment"]["persistent_collapse"])

    def test_historical_failed_run_is_caught_counterfactually_at_update_10(self):
        primary_5 = PROJECT_DIR / "evaluations" / "P0_BASE_C0_gate_u000005_20260809_210149"
        primary_10 = PROJECT_DIR / "evaluations" / "P0_BASE_C0_gate_u000010_20260809_220248"
        summary_5 = summarize_c0_gate(read_jsonl(primary_5 / "episodes.jsonl"))
        summary_10 = summarize_c0_gate(read_jsonl(primary_10 / "episodes.jsonl"))
        assessment = evaluate_behavioral_health(
            update=10,
            c0_summary=summary_10,
            previous_c0_summary=summary_5,
            contract=self.contract,
        )
        self.assertEqual(assessment["status"], "HOLD")
        self.assertEqual(assessment["reason_codes"], ["BEHAVIORAL_COLLAPSE"])
        self.assertTrue(assessment["persistent_collapse"])


class CriticHoldV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]

    def test_four_degenerate_updates_warn_but_do_not_hold(self):
        health = evaluate_training_health(
            _metric_records(10, degenerate_updates=range(7, 11)),
            contract=self.contract,
        )
        self.assertEqual(health["status"], "PASS")
        self.assertEqual(
            health["observed"]["critic/degeneracy"]["all_signal_streak"], 4
        )

    def test_five_joint_degenerate_updates_hold(self):
        health = evaluate_training_health(
            _metric_records(10, degenerate_updates=range(6, 11)),
            contract=self.contract,
        )
        self.assertEqual(health["status"], "HOLD")
        self.assertIn("CRITIC_DEGENERACY", health["reason_codes"])

    def test_one_absent_critic_signal_prevents_joint_hold(self):
        records = _metric_records(10, degenerate_updates=range(6, 11))
        for record in records[5:]:
            record["scalars"]["critic/value_prediction_std"] = 0.2
        health = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(health["status"], "PASS")

    def test_value_normalizer_update_count_mismatch_holds(self):
        records = _metric_records(5)
        records[-1]["scalars"]["critic/value_normalizer_update_count"] = 4.0
        health = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(health["status"], "HOLD")
        self.assertIn("VALUENORM_STATE_INVALID", health["reason_codes"])


if __name__ == "__main__":
    unittest.main()
