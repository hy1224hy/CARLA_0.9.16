"""Reject budget overruns, skipped screens and incompatible full-state sources."""
import unittest
from diagnostics.v6_long_session import require_stages, require_boundary, expected_digest


class LongSessionTests(unittest.TestCase):
    def checkpoint(self, update=50):
        return {'update': update, 'total_episodes': update*16,
                'run_metadata': {'run_id': 'same-run', 'source_type': 'fresh', 'source_checkpoint': ''},
                'training_config': {'reward_version': 'v6', 'pace_profile': 'instant_pace_v6'},
                'lr_schedule': {'decay_horizon': 600}}

    def test_bounded_increasing_milestones(self):
        require_stages(50, [100, 200, 400, 600])
        for source, stops in ((0, [100, 600]), (50, []), (50, [100, 100, 600]),
                              (50, [200, 100, 600]), (50, [50, 600]), (50, [100, 601]),
                              (50, [True, 600])):
            with self.assertRaises(ValueError):
                require_stages(source, stops)

    def test_full_state_boundary_and_origin(self):
        require_boundary({'run_dir': 'same-run'}, self.checkpoint(), 50, 100)
        for update, expected, stop in ((49, 50, 100), (50, 50, 601), (100, 100, 100)):
            with self.assertRaises(ValueError):
                require_boundary({'run_dir': 'same-run'}, self.checkpoint(update), expected, stop)
        for patch in ({'run_id': 'other'}, {'source_type': 'pretrained_weights'},
                      {'source_checkpoint': 'old.pt'}, {'code_drift_override': True}, {'reward_fork': {}}):
            checkpoint = self.checkpoint()
            checkpoint['run_metadata'].update(patch)
            with self.assertRaises(ValueError):
                require_boundary({'run_dir': 'same-run'}, checkpoint, 50, 100)

    def test_schedule_and_episode_state_cannot_reset(self):
        for section, key, value in (('lr_schedule', 'decay_horizon', 100),
                                    ('training_config', 'pace_profile', 'instant_pace_v5')):
            checkpoint = self.checkpoint()
            checkpoint[section][key] = value
            with self.assertRaises(ValueError):
                require_boundary({'run_dir': 'same-run'}, checkpoint, 50, 100)
        checkpoint = self.checkpoint()
        checkpoint['total_episodes'] = 0
        with self.assertRaises(ValueError):
            require_boundary({'run_dir': 'same-run'}, checkpoint, 50, 100)

    def test_cannot_skip_an_intermediate_frozen_stage(self):
        plan = {'source_update': 50, 'source_sha256': 'pinned-u50', 'stop_updates': [100, 200, 400, 600]}
        self.assertEqual(expected_digest(plan, 50, 100), 'pinned-u50')
        for start, stop in ((50, 600), (100, 400), (200, 600), (600, 601)):
            with self.assertRaises(ValueError):
                expected_digest(plan, start, stop)


if __name__ == '__main__':
    unittest.main()
