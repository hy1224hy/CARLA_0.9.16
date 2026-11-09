"""Explicit interruption recovery keeps the original training identity and budget."""
import copy
import tempfile
from pathlib import Path
import unittest

from diagnostics.resume_v6_scratch import resume_command, require_boundary


class RecoveryTests(unittest.TestCase):
    def plan(self, run):
        return {'run_dir': str(run), 'training_args': {
            'reward_version': 'v6', 'pace_profile': 'instant_pace_v6',
            'seed': 42, 'lr_decay': True, 'lr_decay_horizon': 600,
            'gamma': .999, 'min_log_std': -2., 'cpu': False,
            'resume': '', 'pretrained_checkpoint': '', 'follow_spectator': True}}

    def checkpoint(self, run, update):
        return {'update': update, 'run_metadata': {'run_id': Path(run).name,
                'source_type': 'fresh', 'source_checkpoint': ''},
                'training_config': {'pace_profile': 'instant_pace_v6'}}

    def test_resume_at_u11_and_u25_only(self):
        plan = self.plan('same-run')
        for start, stop in ((11, 25), (25, 50)):
            require_boundary(plan, self.checkpoint('same-run', start), stop)
        for start, stop in ((0, 25), (12, 25), (11, 50), (25, 75)):
            with self.assertRaises(ValueError):
                require_boundary(plan, self.checkpoint('same-run', start), stop)

    def test_old_weights_or_other_run_rejected(self):
        plan = self.plan('same-run')
        for patch in ({'source_type': 'pretrained_weights'}, {'source_checkpoint': 'old.pt'},
                      {'run_id': 'other'}, {'reward_fork': {}}):
            checkpoint = self.checkpoint('same-run', 11)
            checkpoint['run_metadata'].update(patch)
            with self.assertRaises(ValueError):
                require_boundary(plan, checkpoint, 25)

    def test_command_preserves_scientific_settings_and_strict_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            before = copy.deepcopy(plan)
            command = resume_command(plan, self.checkpoint(directory, 11), 25)
            self.assertIn('training.train_no_rllib_mappo_robust_v2', command)
            for key, value in (('--resume', 'latest'), ('--updates', '25'),
                               ('--lr-decay-horizon', '600'), ('--gamma', '0.999'),
                               ('--min-log-std', '-2.0')):
                self.assertEqual(command[command.index(key)+1], value)
            self.assertIn('--lr-decay', command)
            self.assertNotIn('--pretrained-checkpoint', command)
            self.assertNotIn('--allow-resume-code-drift', command)
            self.assertEqual(plan, before)

    def test_injected_pretrained_or_drift_override_rejected(self):
        for field, value in (('pretrained_checkpoint', 'old.pt'), ('allow_resume_code_drift', True)):
            plan = self.plan('same-run')
            plan['training_args'][field] = value
            with self.assertRaises(ValueError):
                resume_command(plan, self.checkpoint('same-run', 11), 25)


if __name__ == '__main__':
    unittest.main()
