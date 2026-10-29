"""Offline behavioral gate checks: real evidence, per-agent failures, no reward ranking."""
import copy
from pathlib import Path
import tempfile
import unittest

from evaluation.base_driving_qualification import (
    PROTOCOL, SCHEMA, load_contract, qualify_episode, qualify_evaluation, select_candidate,
    sha256_file, verify_evaluation_sources,
)


def evidence(first_motion=(1, 1), arrivals=(60, 70), terminal_missing=True):
    case = {"id": "case", "scenario": "north_stationary", "seed": 63001, "mode": "deterministic"}
    scenario = {"task_family": "free_flow", "direction": "northbound", "startup_mode": "stationary",
                "agent_roles": {"agent0": "left", "agent1": "right"}, "pair_id": "pair",
                "agents": {"agent0": {"initial_speed_mps": 0.0}, "agent1": {"initial_speed_mps": 0.0}},
                "physical_template_id": "template", "symmetry_partner": "mirror"}
    rows = [{"type": "reset", "case": case, "frame": 100,
             "kinematics": {"0": {"speed": 0.0}, "1": {"speed": 0.0}},
             "info": {"active_mask": {"0": True, "1": True}}}]
    for step in range(1, max(arrivals) + 1):
        diagnostics, active, terminated, post, rewards = {}, {}, {}, {}, {}
        for i, aid in enumerate(("0", "1")):
            reached = step >= arrivals[i]
            active[aid] = not reached
            terminated[aid] = reached
            diagnostics[aid] = {"speed_mps": 3.0 if step >= first_motion[i] else 0.0,
                                "lateral_distance_m": .1, "distance_to_goal_m": 0.0 if reached else 15.0,
                                "pace_front_blocker_exempt": 0.0, "success_event": float(step == arrivals[i]),
                                "collision_event": 0.0, "offroad_terminated": 0.0,
                                "road_violation_terminated": 0.0, "mission_timeout_event": 0.0}
            post[aid] = None if reached and terminal_missing else {"control": [.3, .01, 0.0]}
            rewards[aid] = 1.0 if step <= arrivals[i] else 0.0
        info = {"active_mask": active, "terminated": terminated,
                "truncated": {"0": False, "1": False}, "reward_diagnostics": diagnostics,
                "reward_details": {aid: {"progress": rewards[aid]} for aid in ("0", "1")},
                "episode_end_reason": "all_reached" if step == max(arrivals) else None}
        rows.append({"type": "step", "step": step, "frame": 100 + step, "info": info,
                     "post": post, "reward": rewards})
    episode = {"case": case, "steps": max(arrivals), "reason": "all_reached",
               "checkpoint_sha256": "abc", "actor_hashes_before": {"0": "a", "1": "b"},
               "actor_hashes_after": {"0": "a", "1": "b"},
               "agents": {str(i): {"steps": v, "return": float(v)} for i, v in enumerate(arrivals)}}
    context = {"protocol": PROTOCOL, "timestep_seconds": .1, "checkpoint_eligible": True}
    return episode, rows, scenario, context


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def qualify(self, values):
        episode, rows, scenario, context = values
        return qualify_episode(episode, rows, scenario, self.contract, context)

    def test_natural_arrival_gap_and_missing_terminal_readback_are_reported(self):
        result = self.qualify(evidence())
        self.assertEqual(result["evidence_status"], "VALID")
        self.assertTrue(result["qualified"])
        self.assertAlmostEqual(result["arrival_gap_seconds_descriptive_only"], 1.0)
        self.assertEqual(result["agents"]["0"]["metrics"]["missing_terminal_actual_control"], 1)
        self.assertGreater(len(result["warnings"]), 0)
        self.assertEqual(result["agents"]["0"]["metrics"]["actual_steer_rate_p95_per_second"], 0.0)

    def test_joint_arrival_cannot_hide_one_slow_start(self):
        result = self.qualify(evidence(first_motion=(1, 30)))
        self.assertTrue(result["both_reached"])
        self.assertTrue(result["agents"]["0"]["qualified"])
        self.assertFalse(result["agents"]["1"]["checks"]["stationary_unblocked_startup"])
        self.assertFalse(result["qualified"])

    def test_rolling_start_has_no_fictitious_fast_start_credit(self):
        values = evidence()
        values[2]["startup_mode"] = "rolling"
        for aid in ("0", "1"):
            values[1][0]["kinematics"][aid]["speed"] = 2.5
            values[2]["agents"]["agent" + aid]["initial_speed_mps"] = 2.5
        result = self.qualify(values)
        metrics = result["agents"]["0"]["metrics"]
        self.assertFalse(metrics["startup_applicable"])
        self.assertIsNone(metrics["unblocked_first_motion_seconds"])

    def test_missing_active_readback_or_blocker_diagnostic_is_not_zero(self):
        values = evidence()
        values[1][5]["post"]["1"] = None
        result = self.qualify(values)
        self.assertEqual(result["evidence_status"], "INVALID_OR_INCOMPLETE")
        self.assertFalse(result["qualified"])
        values = evidence()
        del values[1][5]["info"]["reward_diagnostics"]["1"]["pace_front_blocker_exempt"]
        self.assertEqual(self.qualify(values)["evidence_status"], "INVALID_OR_INCOMPLETE")

    def test_terminal_speed_and_lane_evidence_are_included(self):
        values = evidence()
        values[1][60]["info"]["reward_diagnostics"]["0"]["speed_mps"] = 9.0
        result = self.qualify(values)
        self.assertEqual(result["agents"]["0"]["metrics"]["peak_speed_mps"], 9.0)
        self.assertFalse(result["agents"]["0"]["checks"]["upper_task_speed"])

    def test_legacy_reference_never_qualifies(self):
        values = evidence()
        values[3]["checkpoint_eligible"] = False
        result = self.qualify(values)
        self.assertEqual(result["evidence_status"], "VALID")
        self.assertTrue(result["agents"]["0"]["qualified"])
        self.assertFalse(result["qualified"])
        values[3]["protocol"] = "reward_fork_behavior_v1"
        self.assertEqual(self.qualify(values)["evidence_status"], "INVALID_OR_INCOMPLETE")

    def test_stratum_and_registration_coverage_are_required(self):
        contract = copy.deepcopy(self.contract)
        cfg = contract["evaluation"]
        cfg.update(minimum_cases=2, required_directions=["northbound"], required_startup_modes=["stationary"])
        cases = []
        for seed in (63001, 63002):
            values = evidence()
            values[0]["case"]["seed"] = seed
            cases.append(self.qualify(values))
        report = qualify_evaluation(cases, contract)
        self.assertTrue(report["qualified"])
        cases[1]["agents"]["1"]["qualified"] = False
        cases[1]["agents"]["1"]["failed_checks"] = ["stationary_unblocked_startup"]
        self.assertFalse(qualify_evaluation(cases, contract)["qualified"])
        self.assertFalse(qualify_evaluation(cases, contract, [("abc", "missing", 63001, "deterministic")])["checks"]["complete_valid_evidence"])

    def test_selection_never_promotes_a_fast_but_unqualified_candidate(self):
        context = {"contract_sha256": "c", "scenario_suite_sha256": "s", "suite_role": "validation",
                   "checkpoint_sha256": "u100", "qualified": False, "evidence_status": "COMPLETE_VALID_EVIDENCE"}
        self.assertIsNone(select_candidate([context])["selected_checkpoint_sha256"])
        context["qualified"] = True
        self.assertIsNone(select_candidate([context])["selected_checkpoint_sha256"])
        context["suite_role"] = "heldout"
        with self.assertRaises(ValueError):
            select_candidate([context])
        with self.assertRaises(ValueError):
            select_candidate([{**context, "suite_role": "validation"}, {**context, "scenario_suite_sha256": "other"}])

    def test_selection_waits_for_same_checkpoint_parallel_retention(self):
        # These report-shaped unit fixtures are not written as evaluation data.
        validation = {"schema": SCHEMA, "protocol": PROTOCOL, "contract_sha256": "c",
                      "scenario_suite_sha256": "validation_suite", "suite_role": "validation",
                      "checkpoint_sha256": "checkpoint_u100", "qualified": True,
                      "evidence_status": "COMPLETE_VALID_EVIDENCE", "checks": {"new_protocol_checkpoint": True},
                      "training_scenario_pool_sha256": "a" * 64,
                      "cases": [{"agents": {"0": {"metrics": {"above_preferred_speed_fraction": 0.,
                            "reference_lateral_p95_m": .1, "arrival_seconds": 10.}}}}]}
        regression = {**copy.deepcopy(validation), "suite_role": "training_screen",
                      "scenario_suite_sha256": "parallel_subset", "parallel_regression": True,
                      "scenario_suite_definition": {"kind": "parallel_stage0_regression",
                          "pool_manifest_sha256": "a" * 64, "source_scenario_count": 48,
                          "scenario_names": [f"parallel_{i}" for i in range(16)]}}
        pending = select_candidate([validation])
        self.assertEqual(pending["decision"], "PENDING_PARALLEL_REGRESSION")
        self.assertIsNone(pending["selected_checkpoint_sha256"])
        self.assertTrue(validation["qualified"], "its validation qualification remains a separate fact")
        chosen = select_candidate([validation], [regression])
        self.assertEqual(chosen["selected_checkpoint_sha256"], "checkpoint_u100")
        self.assertTrue(chosen["parallel_regression_required"])
        for field, value in (("qualified", False), ("contract_sha256", "different"),
                             ("suite_role", "heldout"), ("parallel_regression", False)):
            broken = {**regression, field: value}
            with self.subTest(field=field):
                result = select_candidate([validation], [broken])
                self.assertEqual(result["decision"], "PARALLEL_REGRESSION_FAILED")
                self.assertIsNone(result["selected_checkpoint_sha256"])
        wrong_checkpoint = {**regression, "checkpoint_sha256": "different_checkpoint"}
        self.assertEqual(select_candidate([validation], [wrong_checkpoint])["decision"], "PENDING_PARALLEL_REGRESSION")

    def test_execution_and_analysis_source_identity_cannot_drift_or_escape_root(self):
        names = ["evaluation/base_driving_qualification.py", "evaluation/evaluate_base_driving_vnext.py",
                 "evaluation/diagnose_r8c1_behavior.py", "environment/base_driving_env.py"]
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_source_identity_") as temp:
            root = Path(temp)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# TEST_ONLY source identity fixture\n", encoding="utf-8")
            sources = {name: sha256_file(root / name) for name in names}
            verify_evaluation_sources(sources, root)
            with self.assertRaisesRegex(ValueError, "escapes ROOT"):
                verify_evaluation_sources({**sources, "../outside_source.py": "unknown"}, root)
            (root / names[0]).write_text("# changed analysis semantics\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source SHA drift"):
                verify_evaluation_sources(sources, root)


if __name__ == "__main__":
    unittest.main()
