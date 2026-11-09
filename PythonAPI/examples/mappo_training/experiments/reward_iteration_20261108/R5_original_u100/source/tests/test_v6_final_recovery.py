"""Bounded final-stage recovery accepts only the pinned latest safe checkpoint."""
import unittest
from diagnostics.resume_v6_scratch_final import require_boundary


class FinalRecoveryTests(unittest.TestCase):
    def checkpoint(self, update=38):
        return {'update': update, 'run_metadata': {'run_id': 'same-run',
                'source_type': 'fresh', 'source_checkpoint': ''},
                'training_config': {'reward_version': 'v6', 'pace_profile': 'instant_pace_v6'}}

    def test_latest_pinned_boundary_can_finish_only_original_budget(self):
        for start in (25, 38, 49):
            require_boundary({'run_dir': 'same-run'}, self.checkpoint(start), start, 50)
        for expected, actual, stop in ((38, 35, 50), (38, 38, 51), (24, 24, 50), (50, 50, 50)):
            with self.assertRaises(ValueError):
                require_boundary({'run_dir': 'same-run'}, self.checkpoint(actual), expected, stop)

    def test_other_origin_or_run_cannot_be_resumed(self):
        for patch in ({'run_id': 'other'}, {'source_type': 'pretrained_weights'},
                      {'source_checkpoint': 'old.pt'}, {'reward_fork': {}}):
            checkpoint = self.checkpoint()
            checkpoint['run_metadata'].update(patch)
            with self.assertRaises(ValueError):
                require_boundary({'run_dir': 'same-run'}, checkpoint, 38, 50)

    def test_changed_reward_profile_is_rejected(self):
        checkpoint = self.checkpoint()
        checkpoint['training_config']['pace_profile'] = 'instant_pace_v5'
        with self.assertRaises(ValueError):
            require_boundary({'run_dir': 'same-run'}, checkpoint, 38, 50)


if __name__ == '__main__':
    unittest.main()
