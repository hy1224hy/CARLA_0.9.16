"""Recovery must preserve completed screens, the remaining budget and exact source."""
import unittest
from diagnostics.v6_long_recovery import recovery_stops, phase_digest


class LongRecoveryTests(unittest.TestCase):
    def test_recovery_uses_only_remaining_original_milestones(self):
        self.assertEqual(recovery_stops(131, [100, 200, 400, 600], [100]), [200, 400, 600])
        self.assertEqual(recovery_stops(200, [100, 200, 400, 600], [100, 200]), [400, 600])
        self.assertEqual(recovery_stops(555, [100, 200, 400, 600], [100, 200, 400]), [600])

    def test_missing_screens_or_expanded_budget_cannot_be_hidden_by_recovery(self):
        for source, stages, completed in ((131, [100, 200, 400, 600], []),
                                         (131, [100, 200, 400, 600], [100, 200]),
                                         (600, [100, 200, 400, 600], [100, 200, 400, 600]),
                                         (131, [100, 200, 400, 601], [100]),
                                         (True, [100, 200, 400, 600], [])):
            with self.assertRaises(ValueError):
                recovery_stops(source, stages, completed)

    def test_exact_initial_recovery_sha_and_no_skipped_freeze(self):
        record = {'source_update': 131, 'source_sha256': 'pinned-u131', 'stop_updates': [200, 400, 600]}
        self.assertEqual(phase_digest(record, 131, 200), 'pinned-u131')
        for start, stop in ((100, 200), (131, 400), (200, 600), (600, 601)):
            with self.assertRaises(ValueError):
                phase_digest(record, start, stop)


if __name__ == '__main__':
    unittest.main()
