import copy
import unittest
from pathlib import Path

from evaluation.base_go_hold_protocol import (
    GoHoldProtocolError,
    evaluate_milestone_decision,
    evaluate_training_health,
    evaluate_validation_decision,
    load_go_hold_contract,
    summarize_c0_gate,
    validate_go_hold_contract,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_DIR / "configs" / "evaluation" / "p0_base_go_hold_v1.json"
FAMILIES = ("basic", "offset", "close", "rear")


def _metric_records(count=5, *, overrides=None):
    overrides = overrides or {}
    records = []
    for update in range(1, count + 1):
        scalars = {
            "critic/grad_clip_activation_rate": 0.1,
            "critic/explained_variance": 0.2,
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
    case_count=16,
    success=0,
    collision=0,
    timeout=None,
    other=0,
    family_success=None,
    scenario_success=None,
    lane=0,
):
    if timeout is None:
        timeout = case_count - success - collision - other
    if family_success is None:
        family_success = {name: 0 for name in FAMILIES}
    if scenario_success is None:
        scenario_success = {f"scenario_{index}": 0 for index in range(8)}
    return {
        "case_count": case_count,
        "outcome_counts": {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "other": other,
        },
        "family_case_counts": {name: case_count // 4 for name in FAMILIES},
        "family_success_counts": dict(family_success),
        "scenario_case_counts": {
            name: case_count // len(scenario_success) for name in scenario_success
        },
        "scenario_success_counts": dict(scenario_success),
        "lane_invasion_episode_count": lane,
    }


class GoHoldContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)
        cls.contract = cls.record["payload"]

    def test_repository_contract_freezes_bindings_and_release_policy(self):
        contract = self.contract
        self.assertEqual(contract["status"], "frozen")
        self.assertEqual(
            contract["bindings"]["training_contract_sha256"],
            "342a1fdb14743428cc2c73e1658de4eb3eaf6cbb6ad5988ff29b64529e051f6f",
        )
        self.assertEqual(
            contract["bindings"]["c0_contract_sha256"],
            "0314f53a15c0095a3f1c6d7e9022054bbfc20403efbde052881096ae7f3f2832",
        )
        self.assertFalse(contract["decision_policy"]["returns_are_selection_criteria"])
        self.assertFalse(contract["decision_policy"]["update_80_can_release_multiseed"])
        self.assertTrue(
            contract["decision_policy"]["only_update_100_pilot_go_releases_multiseed"]
        )

    def test_unknown_field_and_bound_hash_drift_are_rejected(self):
        unknown = copy.deepcopy(self.contract)
        unknown["silent_override"] = True
        with self.assertRaisesRegex(GoHoldProtocolError, "unknown"):
            validate_go_hold_contract(unknown, project_dir=PROJECT_DIR)
        drift = copy.deepcopy(self.contract)
        drift["bindings"]["c0_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(GoHoldProtocolError, "SHA256"):
            validate_go_hold_contract(drift, project_dir=PROJECT_DIR)


class TrainingHealthBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]

    def test_kl_thresholds_use_registered_strict_operators(self):
        records = _metric_records(
            2,
            overrides={"policy/agent_0/approx_kl": [0.05, 0.1]},
        )
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "PASS")

        records[-1]["scalars"]["policy/agent_0/approx_kl"] = 0.100001
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "HOLD")
        self.assertTrue(any("max" in item for item in result["hard_failures"]))

        records = _metric_records(
            2,
            overrides={"policy/agent_0/approx_kl": [0.050001, 0.050001]},
        )
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "HOLD")
        self.assertTrue(any("consecutive" in item for item in result["hard_failures"]))

    def test_clip_fraction_consecutive_hold_and_saturation_warning_boundaries(self):
        records = _metric_records(
            3,
            overrides={
                "policy/agent_1/clip_frac": [0.1, 0.800001, 0.800001],
                "action/agent_0/throttle_saturation_rate": [0.65, 0.65, 0.65],
            },
        )
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(any("true saturation" in item for item in result["warnings"]))

        records = _metric_records(
            3,
            overrides={
                "action/agent_0/throttle_saturation_rate": [0.650001] * 3,
            },
        )
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(any("true saturation" in item for item in result["warnings"]))

    def test_missing_or_nonfinite_health_metric_is_rejected(self):
        records = _metric_records(1)
        del records[0]["scalars"]["critic/explained_variance"]
        with self.assertRaisesRegex(GoHoldProtocolError, "missing"):
            evaluate_training_health(records, contract=self.contract)
        records = _metric_records(1)
        records[0]["scalars"]["critic/explained_variance"] = float("nan")
        with self.assertRaisesRegex(GoHoldProtocolError, "finite"):
            evaluate_training_health(records, contract=self.contract)

    def test_any_update_reward_or_ratio_pollution_is_hold(self):
        records = _metric_records(
            3,
            overrides={
                "reward_audit/max_abs_step_residual": [0.0, 0.0000011, 0.0],
                "policy/agent_0/preupdate_max_abs_ratio_error": [0.0, 0.000001, 0.0],
            },
        )
        result = evaluate_training_health(records, contract=self.contract)
        self.assertEqual(result["status"], "HOLD")
        self.assertTrue(any("reward residual" in item for item in result["hard_failures"]))
        self.assertTrue(any("ratio error" in item for item in result["hard_failures"]))


class MilestoneDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
    def decide(self, update, summary, *, repeat=True):
        return evaluate_milestone_decision(
            update=update,
            training_health=evaluate_training_health(
                _metric_records(update), contract=self.contract
            ),
            c0_summary=summary,
            repeatability_pass=repeat,
            contract=self.contract,
        )

    def test_early_milestones_are_health_gates_and_update_40_warns(self):
        self.assertEqual(self.decide(5, _summary())["decision"], "CONTINUE")
        result = self.decide(40, _summary())
        self.assertEqual(result["decision"], "CONTINUE_WITH_WARNING")
        self.assertTrue(any("zero autonomous success" in item for item in result["warnings"]))

    def test_update_60_exact_floor_passes_and_below_floor_holds(self):
        exact = _summary(success=1, collision=8, timeout=7)
        self.assertEqual(self.decide(60, exact)["decision"], "CONTINUE")
        below = _summary(success=0, collision=8, timeout=8)
        self.assertEqual(self.decide(60, below)["decision"], "HOLD")

    def test_update_80_candidate_is_provisional_only(self):
        exact = _summary(
            success=12,
            collision=2,
            timeout=2,
            family_success={name: 3 for name in FAMILIES},
        )
        result = self.decide(80, exact)
        self.assertEqual(result["decision"], "CANDIDATE")
        self.assertFalse(result["multiseed_released"])
        below = _summary(success=11, collision=2, timeout=3)
        self.assertEqual(
            self.decide(80, below)["decision"], "CONTINUE_WITH_WARNING"
        )

    def test_update_100_exact_boundary_releases_multiseed(self):
        exact = _summary(
            success=14,
            collision=0,
            timeout=2,
            family_success={"basic": 4, "offset": 4, "close": 3, "rear": 3},
            lane=0,
        )
        result = self.decide(100, exact)
        self.assertEqual(result["decision"], "PILOT_GO")
        self.assertTrue(result["multiseed_released"])
        self.assertEqual(
            self.decide(100, {**exact, "lane_invasion_episode_count": 1})["decision"],
            "PILOT_NO_GO",
        )
        collision = _summary(
            success=14,
            collision=1,
            timeout=1,
            family_success={"basic": 4, "offset": 4, "close": 3, "rear": 3},
        )
        self.assertEqual(self.decide(100, collision)["decision"], "PILOT_NO_GO")

    def test_other_outcome_and_repeatability_failure_are_structural_hold(self):
        other = _summary(success=14, timeout=1, other=1)
        self.assertEqual(self.decide(100, other)["decision"], "HOLD")
        self.assertEqual(self.decide(5, _summary(), repeat=False)["decision"], "HOLD")


class ValidationDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_go_hold_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)[
            "payload"
        ]
        cls.health = evaluate_training_health(
            _metric_records(100), contract=cls.contract
        )

    def test_validation_exact_boundary_accepts_and_below_rejects(self):
        scenarios = {f"scenario_{index}": value for index, value in enumerate((3, 3, 2, 2, 2, 2, 2, 2))}
        exact = _summary(
            case_count=24,
            success=18,
            collision=0,
            timeout=6,
            family_success={"basic": 5, "offset": 5, "close": 4, "rear": 4},
            scenario_success=scenarios,
            lane=0,
        )
        result = evaluate_validation_decision(
            training_health=self.health,
            c0_summary=exact,
            repeatability_pass=True,
            contract=self.contract,
        )
        self.assertEqual(result["decision"], "VALIDATION_ACCEPT")
        exact["scenario_success_counts"]["scenario_7"] = 0
        result = evaluate_validation_decision(
            training_health=self.health,
            c0_summary=exact,
            repeatability_pass=True,
            contract=self.contract,
        )
        self.assertEqual(result["decision"], "VALIDATION_REJECT")

    def test_c0_summary_counts_lane_invasion_once_per_episode(self):
        records = []
        for index, outcome in enumerate(("success", "timeout")):
            maximum = 1.0 if index == 0 else 0.0
            records.append(
                {
                    "outcome": outcome,
                    "scenario_family": "basic",
                    "scenario_name": f"scenario_{index}",
                    "agents": {
                        "0": {"diagnostics": {"lane_invasion_sensor_event": {"max": maximum}}},
                        "1": {"diagnostics": {"lane_invasion_sensor_event": {"max": maximum}}},
                    },
                }
            )
        summary = summarize_c0_gate(records)
        self.assertEqual(summary["lane_invasion_episode_count"], 1)
        self.assertEqual(summary["outcome_counts"]["success"], 1)


if __name__ == "__main__":
    unittest.main()
