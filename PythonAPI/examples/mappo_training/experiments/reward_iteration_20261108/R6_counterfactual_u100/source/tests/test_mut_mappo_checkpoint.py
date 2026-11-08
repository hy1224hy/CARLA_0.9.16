import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.checkpoint_protocol import CheckpointProtocolError
from training.mut_mappo_checkpoint import CheckpointStore
from tests.test_mut_mappo_learning import learner, buffer_for


def complete_payload(completed=1):
    teams = {name: learner(name) for name in ("A", "B")}
    for _ in range(completed):
        for team in teams.values():
            team.update(buffer_for(team))
    return {"teams": {name: team.state_dict() for name, team in teams.items()},
            "teachers": {name: team.snapshot(1) for name, team in teams.items()},
            "gate": {}, "completed_round": completed, "next_team": "B",
            "counters": {"train_decision_ticks": 6 * completed},
            "last_metrics": {"completed_round": completed, "status": "offline_unit"}}


class MutCheckpointTests(unittest.TestCase):
    def test_roundtrip_and_metrics_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory, {"run_id": "offline", "config_hash": "frozen"})
            payload = complete_payload()
            store.save(payload)
            store.metrics_path.unlink()
            loaded = store.load()
            store.load()
            self.assertEqual(loaded["completed_round"], 1)
            self.assertEqual(len(store.metrics_path.read_text().splitlines()), 1)
            with store.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write('{"completed_round":2,"torn":')
            store.load()
            self.assertEqual(len(store.metrics_path.read_text().splitlines()), 1)
            restored = learner("A")
            restored.load_state_dict(loaded["teams"]["A"])
            self.assertEqual(restored.update_count, 1)
            with self.assertRaises(CheckpointProtocolError):
                CheckpointStore(directory, {"run_id": "changed"}).load()

    def test_partial_ab_is_rejected_and_actual_cost_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory, {"run_id": "offline"})
            payload = complete_payload()
            store.save(payload)
            store.append_audit({"kind": "partial_B", "actual_ticks": 19})
            payload["teams"]["B"]["update_count"] = 0
            payload["completed_round"] = 2
            with self.assertRaises(CheckpointProtocolError):
                store.save(payload)
            self.assertEqual(store.load()["completed_round"], 1)
            self.assertEqual(json.loads(store.audit_path.read_text())["actual_ticks"], 19)

    def test_staged_checkpoint_without_manifest_does_not_replace_safe_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory, {"run_id": "offline"})
            store.save(complete_payload(1))
            with patch("training.mut_mappo_checkpoint.atomic_write_json", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    store.save(complete_payload(2))
            self.assertEqual(store.load()["completed_round"], 1)
            self.assertEqual(len(list((Path(directory) / "boundaries").glob("*.pt"))), 2)

    def test_corrupt_checkpoint_fails_sha_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory, {"run_id": "offline"})
            manifest = store.save(complete_payload())
            with (Path(directory) / manifest["checkpoint"]).open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(CheckpointProtocolError):
                store.load()

    def test_first_resumed_cost_record_survives_a_torn_audit_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory, {"run_id": "offline"})
            store.append_audit({"kind": "world_ticks", "actual_counters": {"total": 4}})
            with store.audit_path.open("ab") as stream:
                stream.write(b'{"kind":"world_ticks","actual_counters":')
            store.append_audit({"kind": "world_ticks", "actual_counters": {"total": 5},
                                "note": "保留断写成本证据"})
            rows = []
            for line in store.audit_path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.assertEqual(rows[-1]["actual_counters"]["total"], 5)
            self.assertTrue(rows[-1]["previous_unterminated_tail"])


if __name__ == "__main__":
    unittest.main()
