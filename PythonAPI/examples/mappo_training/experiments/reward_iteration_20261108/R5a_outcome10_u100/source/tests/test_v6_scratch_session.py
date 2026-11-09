"""Fresh-state, stage-budget, and committed-file checks; no simulator use."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics.v6_scratch_session import make_training_args, training_integrity, load_plan, archive_session
from utils.reward_fork_session import inspect, training_command


class ScratchSessionTests(unittest.TestCase):
    def plan(self, run):
        return {"run_dir": str(run), "stop_updates": [25, 50], "training_args": {
            "reward_version": "v6", "pace_profile": "instant_pace_v6",
            "lr_actor_final": 1.5e-5, "lr_critic_final": 1e-4,
            "critic_probe_batch_size": 512, "lr_decay_horizon": 600,
            "seed": 42, "rollout_episodes": 16,
        }}

    def test_first_stage_is_true_new_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_training_args(self.plan(directory), 25)
            self.assertEqual((args.resume, args.pretrained_checkpoint), ("", ""))
            self.assertEqual((args.updates, args.seed, args.lr_decay_horizon), (25, 42, 600))
            self.assertEqual(args.pace_profile, "instant_pace_v6")

    def test_fresh_stage_rejects_existing_checkpoint_and_injected_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            plan["training_args"]["pretrained_checkpoint"] = "old.pt"
            with self.assertRaisesRegex(ValueError, "fresh|source"):
                make_training_args(plan, 25)
            plan["training_args"].pop("pretrained_checkpoint")
            (Path(directory)/"latest.pt").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                make_training_args(plan, 25)

    def test_resume_requires_own_complete_fresh_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            checkpoint = {"update": 25, "run_metadata": {
                "source_type": "fresh", "source_checkpoint": "", "run_id": Path(directory).name}}
            args = make_training_args(plan, 50, checkpoint)
            self.assertEqual((args.resume, args.pretrained_checkpoint), ("latest", ""))
            for change in ({"source_type": "pretrained_weights"}, {"reward_fork": {}}, {"run_id": "other"}):
                bad = copy.deepcopy(checkpoint)
                bad["run_metadata"].update(change)
                with self.assertRaises(ValueError):
                    make_training_args(plan, 50, bad)
            checkpoint["update"] = 24
            with self.assertRaises(ValueError):
                make_training_args(plan, 50, checkpoint)

    def test_no_implicit_resume_or_budget_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            for stop in (0, 26, 75):
                with self.assertRaises(ValueError):
                    make_training_args(self.plan(directory), stop)
            with self.assertRaisesRegex(ValueError, "checkpoint|resume"):
                make_training_args(self.plan(directory), 50)

    def test_scratch_inspection_uses_zero_and_ignores_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/"session").mkdir()
            (run/"initialization.json").write_text(json.dumps({"version": "v6_scratch_v1", "source_update": 0}))
            (run/"checkpoint_manifest.json").write_text(json.dumps({"latest": {"update": 1}}))
            (run/"metrics.jsonl").write_text('\n'.join(json.dumps({"update": u, "scalars": {}}) for u in (1,2))+'\n')
            snapshot = inspect(run)
            self.assertEqual(snapshot["source_update"], 0)
            self.assertEqual(snapshot["new_updates"], 1)
            self.assertEqual(snapshot["last_committed_metrics"]["update"], 1)
            (run/"reward_fork.json").write_text('{"source_update":456}')
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                inspect(run)

    def test_supervisor_keeps_legacy_and_scratch_commands_separate(self):
        old = training_command(Path('example'), 481)
        new = training_command(Path('example'), 25, 'plan.json')
        self.assertIn('training.reward_fork_v2', old)
        self.assertIn('diagnostics.v6_scratch_session', new)
        self.assertIn('--plan', new)
        self.assertNotIn('training.reward_fork_v2', new)

    def test_integrity_checks_every_update_and_real_episode_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            scalars = {key: 0 for key in (
                'train/update_skipped', 'reward_audit/nonfinite_count', 'reward_audit/max_abs_step_residual',
                'policy/agent_0/preupdate_max_abs_ratio_error', 'policy/agent_1/preupdate_max_abs_ratio_error')}
            (run/'metrics.jsonl').write_text(json.dumps({'update': 1, 'scalars': scalars})+'\n')
            (run/'episodes.jsonl').write_text(''.join(json.dumps({'update': 1})+'\n' for _ in range(16)))
            self.assertEqual(training_integrity(run, 1)['episodes'], 16)
            (run/'episodes.jsonl').write_text('')
            with self.assertRaisesRegex(RuntimeError, 'episode'):
                training_integrity(run, 1)

    def test_frozen_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'logs').mkdir()
            (root/'reward.py').write_text('unchanged')
            import hashlib
            plan = {**self.plan(root/'logs/fresh'), 'version': 'v6_scratch_v1',
                    'source_sha256': {'reward.py': hashlib.sha256(b'unchanged').hexdigest()},
                    'scenario_sha256': {}}
            (root/'plan.json').write_text(json.dumps(plan))
            with patch('diagnostics.v6_scratch_session.ROOT', root):
                load_plan(root/'plan.json')
                (root/'reward.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'frozen source changed'):
                    load_plan(root/'plan.json')

    def test_only_matching_successful_session_can_be_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, control = root/'run/session', root/'control'
            session.mkdir(parents=True)
            control.mkdir()
            (session/'launch.json').write_text('{"launch_id":"current"}')
            (session/'exit.json').write_text('{"launch_id":"old", "returncode":0}')
            with self.assertRaisesRegex(RuntimeError, 'incomplete or failed'):
                archive_session(root/'run', control)
            self.assertTrue((session/'launch.json').exists())
            (session/'exit.json').write_text('{"launch_id":"current", "returncode":0}')
            archive_session(root/'run', control)
            self.assertFalse((session/'launch.json').exists())
            self.assertTrue((control/'completed_session_current/exit.json').exists())


if __name__ == '__main__':
    unittest.main()
