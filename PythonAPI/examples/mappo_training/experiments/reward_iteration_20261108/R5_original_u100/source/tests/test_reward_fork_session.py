"""Ownership and committed-file monitoring checks; no CARLA connection."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from utils.mut_mappo_session import competing_clients
from utils.reward_fork_session import inspect, rows


class RewardSessionTests(unittest.TestCase):
    def test_new_base_worker_is_seen_by_legacy_global_ownership_audit(self):
        processes = [dict(ProcessId=90002, ParentProcessId=8,
                          CommandLine='"venv/python.exe" -u -m utils.base_driving_session worker --plan other/plan.json --launch-id other-uuid')]
        with patch('utils.mut_mappo_session.os.name', 'nt'), \
             patch('utils.mut_mappo_session.subprocess.run', return_value=SimpleNamespace(
                 returncode=0, stdout=json.dumps(processes))):
            self.assertEqual(competing_clients(), [90002])

    def test_windows_launcher_ancestor_excluded_sibling_still_competes(self):
        own = os.getpid()
        command = '-u -m training.reward_fork_v2 --run-dir example'
        processes = [
            dict(ProcessId=own, ParentProcessId=90001, CommandLine='"native/python.exe" '+command),
            dict(ProcessId=90001, ParentProcessId=8, CommandLine='"venv/python.exe" '+command),
            dict(ProcessId=90002, ParentProcessId=8, CommandLine='"venv/python.exe" '+command),
        ]
        with patch('utils.mut_mappo_session.os.name', 'nt'), \
             patch('utils.mut_mappo_session.subprocess.run', return_value=SimpleNamespace(
                 returncode=0, stdout=json.dumps(processes))):
            self.assertEqual(competing_clients(), [90002])

    def test_files_only_snapshot_ignores_uncommitted_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'session').mkdir()
            (run/'checkpoint_manifest.json').write_text(json.dumps({'latest': {'update': 457}}))
            (run/'reward_fork.json').write_text(json.dumps({'source_update': 456}))
            (run/'metrics.jsonl').write_text('\n'.join(json.dumps({'update': update, 'scalars': {
                'train/update_skipped': 0, 'curriculum/agent_0/mean_speed_mps': speed}})
                for update, speed in ((457, 3), (458, 99)))+'\n')
            (run/'episodes.jsonl').write_text(json.dumps({'update': 457, 'scenario_name': 'example',
                                                        'success': True})+'\n')
            with patch('utils.mut_mappo_session.pid_alive', side_effect=AssertionError('process query')):
                snapshot = inspect(run)
            self.assertEqual(snapshot['safe_update'], 457)
            self.assertEqual(snapshot['last_committed_metrics']['update'], 457)
            self.assertEqual(snapshot['outcomes']['success'], 1)
            self.assertEqual(snapshot['inspection_mode'], 'FILES_ONLY')
            self.assertFalse(snapshot['metrics_commit_pending'])

    def test_only_partial_final_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'records.jsonl'
            path.write_text('{"update":457}\n{"update":')
            self.assertEqual(rows(path), [{'update': 457}])
            path.write_text('corrupt\n{"update":457}\n')
            with self.assertRaisesRegex(ValueError, 'invalid committed'):
                rows(path)


if __name__ == '__main__':
    unittest.main()
