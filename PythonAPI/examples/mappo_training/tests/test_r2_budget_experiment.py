import unittest
import tempfile
from pathlib import Path

from utils.r2_budget_experiment import milestone_boundaries, records


class MilestoneTests(unittest.TestCase):
    def test_first_complete_boundary_reaches_soft_cap_and_initial_is_retained(self):
        rows = [dict(completed_round=0), *[dict(completed_round=i, actual_counters={'train': n})
            for i, n in enumerate((23000, 24400, 48000, 48600, 72500), 1)]]
        nodes = milestone_boundaries(rows)
        self.assertEqual([r['completed_round'] for r in nodes], [0, 2, 3, 5])
        self.assertEqual([r['actual_train_ticks'] for r in nodes], [0, 24400, 48000, 72500])

    def test_incomplete_training_cannot_supply_a_final_node(self):
        with self.assertRaises(ValueError):
            milestone_boundaries([dict(completed_round=0), dict(completed_round=1, actual_counters={'train': 50000})])

    def test_file_monitor_ignores_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metrics.jsonl'
            path.write_text('{"round":1}\n{"round":',encoding='utf-8')
            self.assertEqual(records(path),[{'round':1}])


if __name__ == '__main__':
    unittest.main()
