"""Small supervisor checks using OS locks and mocked children; never CARLA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils.mut_mappo_session import RunSessionOwner, inspect_files, launch, supervise


def arguments(run):
    return argparse.Namespace(run_dir=str(run), config="configs/mut_mappo/local_candidate_v1.json",
                              resume=False, rounds=2, supervisor_log=None)


class SessionTests(unittest.TestCase):
    def test_monitor_options_reach_supervisor_and_training_child(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            args.monitor_mode, args.monitor_port = "web", 18081
            args.monitor_fps = 3.0
            with patch("utils.mut_mappo_session.subprocess.Popen") as child, patch("builtins.print"):
                child.return_value.pid = 456789
                child.return_value.poll.return_value = 0
                child.return_value.returncode = 0
                dispatched = launch(args)
                self.assertEqual(supervise(args), 0)
            commands = [dispatched["command"], child.call_args_list[1].args[0]]
            for command in commands:
                for flag, value in (("--monitor-mode", "web"), ("--monitor-port", "18081"),
                                    ("--monitor-fps", "3.0")):
                    self.assertEqual(command[command.index(flag)+1], value)

    def test_lock_rejects_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunSessionOwner(directory):
                with self.assertRaisesRegex(RuntimeError, "another supervisor"):
                    with RunSessionOwner(directory):
                        self.fail("a second supervisor acquired the same run")
            with RunSessionOwner(directory):
                pass

    def test_duplicate_supervise_does_not_spawn_or_replace_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with RunSessionOwner(run):
                path = run / "session/launch.json"
                path.write_text('{"trainer_pid":123,"command":["original"]}', encoding="utf-8")
                original = path.read_bytes()
                with patch("utils.mut_mappo_session.subprocess.Popen") as child:
                    with self.assertRaisesRegex(RuntimeError, "another supervisor"):
                        supervise(arguments(run))
                    child.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((run / "session/exit.json").exists())

    def test_live_recorded_trainer_or_supervisor_is_rejected(self):
        for role in ("trainer", "supervisor"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                session = run / "session"
                session.mkdir()
                record = {role + "_pid": 982341, "command": ["existing"]}
                path = session / "launch.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                with patch("utils.mut_mappo_session.pid_alive", return_value=True), \
                     patch("utils.mut_mappo_session.subprocess.Popen") as child:
                    with self.assertRaisesRegex(RuntimeError, "previous " + role):
                        supervise(arguments(run))
                    child.assert_not_called()
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)

    def test_supervisor_keeps_lock_until_child_exit_and_records_actual_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = arguments(run)
            case = self
            class FinishedChild:
                pid, returncode = 456789, 0
                def poll(self):
                    with case.assertRaisesRegex(RuntimeError, "another supervisor"):
                        supervise(args)
                    return self.returncode
            with patch("utils.mut_mappo_session.subprocess.Popen", return_value=FinishedChild()) as child, \
                 patch("builtins.print"):
                self.assertEqual(supervise(args), 0)
                child.assert_called_once()
            observed = inspect_files(run)
            self.assertEqual(observed["launch"]["trainer_pid"], 456789)
            self.assertEqual(observed["exit"]["trainer_pid"], 456789)
            self.assertTrue(observed["exit_is_current_launch"])
            self.assertIn("training.train_mut_mappo", observed["launch"]["command"])
            self.assertIn("supervisor_command", observed["launch"])
            with RunSessionOwner(run):
                pass

    def test_two_dispatches_keep_unique_logs_and_inspection_only_reads_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = arguments(run)
            args.resume = True
            child_pid = iter((654321, 654322))
            def fake_spawn(command, **kwargs):
                self.assertNotEqual(kwargs["stdout"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertIn("--supervisor-log", command)
                self.assertIn("--resume", command)
                kwargs["stdout"].write("duplicate startup or other failure is visible\n")
                return argparse.Namespace(pid=next(child_pid))
            with patch("utils.mut_mappo_session.subprocess.Popen", side_effect=fake_spawn):
                first, second = launch(args), launch(args)
            self.assertNotEqual(first["supervisor_log_path"], second["supervisor_log_path"])
            self.assertEqual(len(list((run / "session").glob("dispatch_*.json"))), 2)
            with patch("utils.mut_mappo_session.pid_alive", side_effect=AssertionError("process query")), \
                 patch("utils.mut_mappo_session.competing_clients", side_effect=AssertionError("process query")):
                observed = inspect_files(run)
            self.assertEqual(observed["inspection_mode"], "FILES_ONLY")
            self.assertEqual(observed["process_liveness"], "NOT_QUERIED_FILES_ONLY")
            self.assertEqual(len(observed["supervisor_logs"]), 2)
            self.assertTrue(all("failure is visible" in entry["tail"] for entry in observed["supervisor_logs"]))
            self.assertEqual(observed["latest_dispatch"]["status"], "DISPATCHED_VERIFY_LOGS")
            self.assertFalse(observed["automatic_restart"])

    def test_dispatch_failure_has_inspectable_log_even_without_launch_record(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.mut_mappo_session.subprocess.Popen", side_effect=OSError("offline mock failure")):
                with self.assertRaisesRegex(OSError, "offline mock failure"):
                    launch(arguments(directory))
            observed = inspect_files(directory)
            self.assertEqual(observed["launch"], {})
            self.assertIn("DISPATCH_FAILED", observed["supervisor_logs"][0]["tail"])
            self.assertIn("offline mock failure", observed["supervisor_logs"][0]["tail"])


if __name__ == "__main__":
    unittest.main()
