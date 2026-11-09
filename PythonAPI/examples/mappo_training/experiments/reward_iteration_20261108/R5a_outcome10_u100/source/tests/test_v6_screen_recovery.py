"""A failed simulator connection must not cause completed training to be repeated."""
import unittest
from diagnostics.resume_v6_scratch_screen25 import require_finished_u25


class ScreenRecoveryTests(unittest.TestCase):
    def test_complete_u25_and_matching_exit_are_required(self):
        require_finished_u25({'update': 25}, {'launch_id': 'ok'}, {'launch_id': 'ok', 'returncode': 0})
        for update, exited in ((24, {'launch_id': 'ok', 'returncode': 0}),
                               (25, {}), (25, {'launch_id': 'old', 'returncode': 0}),
                               (25, {'launch_id': 'ok', 'returncode': 1})):
            with self.assertRaises(ValueError):
                require_finished_u25({'update': update}, {'launch_id': 'ok'}, exited)

    def test_missing_launch_identity_never_counts_as_success(self):
        with self.assertRaises(ValueError):
            require_finished_u25({'update': 25}, {}, {'returncode': 0})


if __name__ == '__main__':
    unittest.main()
