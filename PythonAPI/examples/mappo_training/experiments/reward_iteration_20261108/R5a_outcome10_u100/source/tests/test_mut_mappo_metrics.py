"""Small offline checks for result validity, denominators and serial evaluation."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from evaluation.mut_mappo_metrics import paired_report, summarize_cases
from evaluation.evaluate_mut_mappo import evaluate, evaluate_records


def triplet(key, first=True, second=False, eligible=True, *, team="A", attempt=0):
    return [{"case_key": key, "team": team, "condition": condition, "paired_attempt": attempt,
             "red_success": success, "blue_eligible": eligible if condition == "C2" else True,
             "response": condition == "C2", "valid": True}
            for condition, success in (("C0", True), ("C1", first), ("C2", second))]


class ExternalMetricsTests(unittest.TestCase):
    def test_attack_contributions_and_q_keep_all_valid_denominators(self):
        rows = triplet("safe_worse") + triplet("unsafe_worse", eligible=False) + triplet("safe_better", False, True)
        report = paired_report(rows)
        self.assertAlmostEqual(report["AttackGain_raw"], 1 / 3)
        self.assertEqual(report["AttackGain_safe"], 0)
        self.assertAlmostEqual(report["AttackGain_unsafe"], 1 / 3)
        self.assertEqual(report["counts"]["safe_worsened"], 1)
        self.assertEqual(report["counts"]["safe_improved"], 1)
        c2 = report["conditions"]["C2"]
        self.assertEqual(c2["q_count"], 2)
        self.assertAlmostEqual(c2["q"], 2 / 3)
        self.assertAlmostEqual(c2["b"], 1 / 3)

    def test_invalid_baseline_excludes_whole_pair_but_censored_red_does_not(self):
        invalid = triplet("invalid")
        invalid[0].update(valid=False, invalid_reason="sensor_failure")
        real_failure = triplet("true_failure")
        real_failure[2]["red_results"] = {
            "0": {"success": False, "end_reason": "collision"},
            "1": {"success": False, "end_reason": "other_red_failure", "censored": True},
        }
        report = paired_report(invalid + real_failure)
        self.assertEqual(report["n_common_valid"], 1)
        self.assertEqual(report["n_excluded"], 1)
        self.assertEqual(report["AttackGain_raw"], 1)
        self.assertEqual(report["conditions"]["C2"]["red_censored_count"], 1)
        self.assertEqual(report["conditions"]["C2"]["red_end_reason_counts"], {"collision": 1})

    def test_retry_requires_latest_complete_group_and_retains_consumed_attempts(self):
        old = triplet("rerun")
        latest = triplet("rerun", second=True, attempt=1)
        self.assertEqual(paired_report(old + latest)["AttackGain_raw"], 0)
        self.assertEqual(summarize_cases(old + latest)["n_attempted_episodes"], 6)
        self.assertEqual(paired_report(old + latest[:2])["n_common_valid"], 0)
        latest[2].update(valid=False, invalid_reason="manual_stop")
        self.assertEqual(paired_report(old + latest)["n_common_valid"], 0)

    def test_teams_do_not_become_independent_replicates(self):
        report = paired_report(triplet("one", team="A") + triplet("one", second=True, team="B"))
        self.assertNotIn("AttackGain_raw", report)
        self.assertEqual(report["teams"]["A"]["AttackGain_raw"], 1)
        self.assertEqual(report["teams"]["B"]["AttackGain_raw"], 0)

    def test_duplicate_case_condition_cannot_improve_denominator(self):
        rows = triplet("duplicate")
        report = paired_report(rows + [rows[2]])
        self.assertEqual(report["n_common_valid"], 0)
        self.assertIn("C2:duplicate", report["excluded"][0]["reasons"])
        with self.assertRaises(ValueError):
            summarize_cases([{**rows[0], "valid": "false"}])
        with self.assertRaisesRegex(ValueError, "training-window"):
            summarize_cases([{**rows[0], "record_kind": "training_episode"}])

    def test_empty_and_safety_report_do_not_mix_red_and_blue_failures(self):
        self.assertEqual(paired_report([])["teams"], {})
        row = triplet("unsafe", eligible=False)[2]
        row["safety"] = {"2": {"collision": True, "task_violation": True}, "3": {"collision": True}}
        row["end_reasons"] = {"0": "offroad"}
        summary = summarize_cases([row])
        self.assertEqual(summary["blue_violation_episode_counts"], {"collision": 1, "task_violation": 1})
        self.assertEqual(summarize_cases([])["q"], None)


class EvaluationLoopTests(unittest.TestCase):
    def test_offline_preflight_and_live_retry_use_same_factory_serially(self):
        trainer = ModuleType("training.train_mut_mappo")
        session = ModuleType("utils.mut_mappo_session")
        calls = []
        live = [False]
        c2_calls = [0]
        class Owner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                calls.append("owner_enter")
            def __exit__(self, *_):
                self.assert_closed()
                calls.append("owner_exit")
            def assert_closed(self):
                if live[0]:
                    raise AssertionError("runtime leaked")
        class Runtime:
            def __init__(self, condition, audit_callback):
                if live[0]:
                    raise AssertionError("simultaneous runtimes")
                live[0] = True
                self.condition = condition
                self.audit_callback = audit_callback
                audit_callback({"kind": "setup", "ticks": 1})
                calls.append(condition)
            def close(self):
                self.audit_callback({"kind": "setup", "ticks": 1})
                live[0] = False
                calls.append("close")
        trainer.load_config = lambda path: json.loads(Path(path).read_text())
        trainer.validate_config = lambda config, config_path=None: config
        trainer.validate_checkpoint_for_config = lambda config, path: {"status": "MODEL_VALIDATED"}
        trainer.create_runtime = lambda config, checkpoint, condition, team, audit_callback: Runtime(condition, audit_callback)
        def run_case(runtime, case, *, seed, deterministic, trace_path):
            runtime.audit_callback({"kind": "decision", "ticks": 1})
            valid = True
            if runtime.condition == "C2":
                c2_calls[0] += 1
                valid = c2_calls[0] > 1
            return {"case_key": case["case_key"], "condition": runtime.condition,
                    "red_success": runtime.condition != "C2", "blue_eligible": True,
                    "response": runtime.condition == "C2", "valid": valid,
                    "invalid_reason": None if valid else "sensor_failure"}
        trainer.run_case = run_case
        session.CarlaOwner = Owner
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "cases.json"
            manifest.write_text(json.dumps({"schema_version": "test", "cases": [{"case_key": "k", "seed": 7}],
                                           "splits": {"selection": ["k"]}}))
            config = base / "config.json"
            config.write_text(json.dumps({"case_manifest": str(manifest), "evaluation": {"max_invalid_retries": 1}}))
            checkpoint = base / "checkpoint.pt"
            checkpoint.write_bytes(b"offline identity fixture")
            modules = {"training.train_mut_mappo": trainer, "utils.mut_mappo_session": session}
            import training as training_package
            trainer.validate_checkpoint_for_config = lambda config, path: {"completed_round": 1}
            with patch.dict(sys.modules, modules), patch.object(training_package, "train_mut_mappo", trainer, create=True):
                result = evaluate(config, checkpoint, team="A", validate_only=True)
                self.assertFalse(result["carla_connected"])
                self.assertEqual(calls, [])
                output = base / "evaluation"
                result = evaluate(config, checkpoint, team="A", output_dir=output)
            self.assertEqual(calls, ["owner_enter", "C0", "close", "C1", "close", "C2", "close",
                                     "C0", "close", "C1", "close", "C2", "close", "owner_exit"])
            self.assertEqual(result["paired"]["n_common_valid"], 1)
            self.assertEqual(result["paired"]["n_invalid_attempted_episodes"], 1)
            self.assertEqual(result["attempts"]["consumed_tick_counts"], {"total": 18, "decision": 6, "setup": 12})
            self.assertEqual(len((output / "cases.jsonl").read_text().splitlines()), 6)
            index = json.loads((output / "result_index.json").read_text())
            self.assertEqual(index["entries"][0]["status"], "COMPLETED")
            self.assertFalse(index["entries"][0]["teams_are_independent_seeds"])


if __name__ == "__main__":
    unittest.main()
