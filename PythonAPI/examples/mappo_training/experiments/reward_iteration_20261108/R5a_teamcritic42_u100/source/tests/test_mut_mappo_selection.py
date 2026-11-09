"""Result-critical selection checks without CARLA or environment clocks."""
import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

import torch

from evaluation.mut_mappo_metrics import paired_report, summarize_cases
from evaluation.mut_mappo_selection import EXPORT_SCHEMA, select_and_export
from evaluation.evaluate_mut_mappo import evaluate, preflight
from models.actor_critic import Actor
from training.checkpoint_protocol import load_checkpoint_file
from training.mut_mappo_checkpoint import SCHEMA_VERSION
from training.mut_mappo_learning import TeamLearner, actor_states_hash


def checkpoint(round_no=1, teams=("A", "B")):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        initial = {aid: Actor(20, 2, hidden=16).state_dict() for aid in (0, 1)}
    learners = {name: TeamLearner(name, 7 + index, initial, {"hidden_dim": 16, "method": "off"})
                for index, name in enumerate(teams)}
    for team in learners.values():
        team.update_count = round_no
        team.schedule_state["update"] = round_no
    return {"schema_version": SCHEMA_VERSION, "team_count": len(teams), "teams": {name: learner.state_dict() for name, learner in learners.items()},
            "teachers": {name: None for name in teams}, "gate": {}, "completed_round": round_no,
            "next_team": "A", "counters": {}, "last_metrics": {"completed_round": round_no},
            "identity": {"red_sha256": "a" * 64, "case_manifest_sha256": "b" * 64,
                         "model_spec": {"obs_dim": 20, "action_dim": 2}, "config_sha256": "c" * 64}}


def records(team, *, gain=4, responses=4, bad=0, task=False, attempt=0):
    rows = []
    for index in range(8):
        for condition in ("C0", "C1", "C2"):
            unsafe = condition == "C2" and index >= 8 - bad
            rows.append({"case_key": f"case_{index}", "team": team, "condition": condition,
                         "paired_attempt": attempt, "valid": True,
                         "red_success": condition != "C2" or index >= gain,
                         "blue_eligible": not unsafe,
                         "response": condition == "C2" and index < responses,
                         "safety": {"2": {"task_violation" if task else "collision": True}} if unsafe else {}})
    return rows


def candidate(payload=None, *, digest="d" * 64, rows=None, split="selection"):
    payload = checkpoint() if payload is None else payload
    rows = records("A") + records("B") if rows is None else rows
    teams = sorted({row["team"] for row in rows})
    report = {"evidence": "independent_evaluation", "metadata": {
        "status": "COMPLETED", "split": split, "teams": teams, "case_keys": [f"case_{i}" for i in range(8)],
        "conditions": ["C0", "C1", "C2"], "checkpoint_sha256": digest,
        "case_manifest_sha256": "b" * 64, "configuration_identity": {"red_sha256": "a" * 64}},
        "paired": paired_report(rows), "attempts": summarize_cases(rows)}
    return {"checkpoint_payload": payload, "checkpoint_sha256": digest, "report": report}


class SelectionTests(unittest.TestCase):
    config = {"evaluation": {"selection_max_bad_rate": 1 / 8}, "research_status": "candidate"}

    def select(self, candidates):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = select_and_export(candidates, self.config, output)
            exported = load_checkpoint_file(output / "frozen_blue_team.pt")[0] if result["export"] else None
            self.assertTrue((output / "selection.json").is_file())
            if exported:
                self.assertEqual(exported["schema_version"], EXPORT_SCHEMA)
                self.assertEqual(set(exported["actors"]), {2, 3})
                self.assertEqual(exported["model_spec"]["obs_dim"], 20)
                self.assertEqual(exported["model_spec"]["action_dim"], 2)
                self.assertEqual(actor_states_hash(exported["actors"]), exported["actor_hash"])
                self.assertEqual(exported["source_checkpoint_sha256"], result["selected"]["checkpoint_sha256"])
                self.assertTrue(all(not tensor.requires_grad for state in exported["actors"].values() for tensor in state.values()))
            return result, exported

    def test_safety_filter_precedes_attack_gain_and_export_retains_identity(self):
        result, exported = self.select([candidate(rows=records("A", gain=8, bad=2) + records("B", gain=3))])
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"]["team"], "B")
        self.assertEqual(exported["identity"]["red_sha256"], "a" * 64)
        self.assertEqual(exported["identity"]["case_manifest_sha256"], "b" * 64)
        self.assertEqual(exported["purpose"], "frozen_blue_inference_only_not_training_resume")

    def test_ranking_uses_gain_then_q_then_task_then_earlier_round_and_stable_team(self):
        result, _ = self.select([candidate(rows=records("A", gain=3, responses=7) + records("B", gain=4, responses=1))])
        self.assertEqual(result["selected"]["team"], "B")
        result, _ = self.select([candidate(rows=records("A", responses=3) + records("B", responses=5))])
        self.assertEqual(result["selected"]["team"], "B")
        result, _ = self.select([candidate(rows=records("A", bad=1, task=True) + records("B", bad=1))])
        self.assertEqual(result["selected"]["team"], "B")
        result, _ = self.select([candidate(checkpoint(3), digest="e" * 64), candidate(checkpoint(1))])
        self.assertEqual((result["selected"]["completed_round"], result["selected"]["team"]), (1, "A"))

    def test_final_common_denominator_excludes_retry_cost_from_selection(self):
        old = records("A", bad=8)
        old[0].update(valid=False, invalid_reason="reset_failed")
        rows = old + records("A", attempt=1) + records("B")
        item = candidate(rows=rows)
        self.assertGreater(item["report"]["attempts"]["teams"]["A"]["C2"]["b"], 0)
        result, _ = self.select([item])
        self.assertEqual(result["selected"]["team"], "A")
        self.assertEqual(result["selected"]["b"], 0)
        self.assertEqual(result["selected"]["n_common_valid"], 8)

    def test_incomplete_candidate_blocks_selection_and_unsafe_set_exports_nothing(self):
        rows = records("A") + records("B")
        rows[-1].update(valid=False, invalid_reason="sensor_failed")
        result, exported = self.select([candidate(rows=rows)])
        self.assertEqual(result["status"], "INCOMPLETE_EVIDENCE")
        self.assertIsNone(exported)
        result, exported = self.select([candidate(rows=records("A", bad=2) + records("B", bad=2))])
        self.assertEqual(result["status"], "NONE_QUALIFIED")
        self.assertIsNone(exported)

    def test_test_gate_initialization_and_wrong_identity_are_rejected(self):
        for split in ("test", "gate"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "selection-split"):
                self.select([candidate(split=split)])
        with self.assertRaisesRegex(ValueError, "round-zero"):
            self.select([candidate(checkpoint(0))])
        altered = candidate()
        altered["report"]["metadata"]["configuration_identity"]["red_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "red policy"):
            self.select([altered])
        with self.assertRaisesRegex(ValueError, "selection-split"):
            evaluate("unused.json", split="test", select_best=True, validate_only=True)

    def test_no_export_overwrite_and_no_cross_experiment_competition(self):
        one, other = candidate(), candidate(checkpoint(2), digest="e" * 64)
        other["checkpoint_payload"]["identity"]["config_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "frozen experiment identity"):
            self.select([one, other])
        with tempfile.TemporaryDirectory() as temporary:
            select_and_export([one], self.config, temporary)
            with self.assertRaises(FileExistsError):
                select_and_export([one], self.config, temporary)

    def test_single_team_selection_and_preflight_use_checkpoint_topology(self):
        result, _ = self.select([candidate(checkpoint(1, ("A",)), rows=records("A"))])
        self.assertEqual(result["selected"]["team"], "A")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "cases.json"
            manifest.write_text(json.dumps({"schema_version": "fixture", "cases": [{"case_key": "one"}],
                                           "splits": {"selection": ["one"]}}))
            config = {"case_manifest": str(manifest), "team_count": 1}
            source = base / "config.json"
            source.write_text(json.dumps(config))
            checkpoint_source = base / "source.pt"
            checkpoint_source.write_bytes(b"offline placeholder")
            with patch("training.train_mut_mappo.load_config", return_value=config), \
                    patch("training.train_mut_mappo.validate_config", return_value=config), \
                    patch("training.train_mut_mappo.validate_checkpoint_for_config", return_value={"teams": {"A": {}}}) as validator:
                resolved = preflight(source, checkpoint_source, split="selection", team="both", conditions=("C1", "C2"))
                self.assertEqual(resolved["metadata"]["teams"], ["A"])
                with self.assertRaisesRegex(ValueError, "absent"):
                    preflight(source, checkpoint_source, split="selection", team="B", conditions=("C1", "C2"))
                screen = preflight(source, None, split="selection", team="both", conditions=("C0", "C1"))
                self.assertEqual(screen["metadata"]["teams"], ["A"])
                validator.return_value = {"team": "B", "completed_round": 3,
                                          "model_spec": {"algorithm": "ippo"},
                                          "purpose": "frozen_blue_inference_only_not_training_resume"}
                exported = preflight(source, checkpoint_source, split="selection", team="both", conditions=("C1", "C2"))
                self.assertEqual(exported["metadata"]["teams"], ["B"])
                self.assertEqual(exported["metadata"]["checkpoint_algorithm"], "ippo")
                self.assertEqual(exported["metadata"]["checkpoint_completed_round"], 3)
                self.assertEqual(exported["metadata"]["checkpoint_purpose"], "frozen_blue_inference_only_not_training_resume")


if __name__ == "__main__":
    unittest.main()
