"""Offline safety boundaries of the explicit finite serial controller."""

import argparse
import builtins
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils import base_driving_session as session
from utils.base_driving_session import remaining_actions, validate_stops


class BaseDrivingSessionBoundariesTests(unittest.TestCase):
    def test_budget_requires_explicit_strictly_increasing_finite_stops(self):
        self.assertEqual(validate_stops([25, 50, 100, 200, 400, 600]),
                         [25, 50, 100, 200, 400, 600])
        for stops in ([], [0], [601], [25, 25], [50, 25], [True], [25.0]):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                validate_stops(stops)

    def test_fresh_run_alternates_training_and_screens_with_no_implicit_budget(self):
        self.assertEqual(remaining_actions([25, 50], 0, []), [
            ("train", 0, 25), ("screen", 25, 25),
            ("train", 25, 50), ("screen", 50, 50),
        ])

    def test_checkpoint_on_milestone_screens_first_without_retraining(self):
        self.assertEqual(remaining_actions([25, 50, 100], 50, [25]), [
            ("screen", 50, 50), ("train", 50, 100), ("screen", 100, 100),
        ])
        self.assertEqual(remaining_actions([25, 50], 25, []), [
            ("screen", 25, 25), ("train", 25, 50), ("screen", 50, 50),
        ])

    def test_partial_training_continues_from_safe_boundary_without_counting_tail(self):
        self.assertEqual(remaining_actions([100, 200, 400, 600], 327, [100, 200]), [
            ("train", 327, 400), ("screen", 400, 400),
            ("train", 400, 600), ("screen", 600, 600),
        ])

    def test_final_checkpoint_can_complete_only_after_its_pending_screen(self):
        self.assertEqual(remaining_actions([100, 200, 400, 600], 600, [100, 200, 400]),
                         [("screen", 600, 600)])
        self.assertEqual(remaining_actions([100, 200, 400, 600], 600, [100, 200, 400, 600]), [])

    def test_recovery_cannot_skip_an_earlier_unfinished_screen(self):
        for safe, screens in ((51, [25]), (100, []), (327, [100])):
            with self.subTest(safe=safe, screens=screens), self.assertRaisesRegex(ValueError, "cannot skip"):
                remaining_actions([25, 50, 100, 200, 400, 600], safe, screens)

    def test_safe_boundary_and_prior_screens_must_belong_to_authorized_budget(self):
        for safe, screens in ((-1, []), (601, []), (25.0, []), (True, []),
                              (25, [50]), (25, [20]), (0, [25])):
            with self.subTest(safe=safe, screens=screens), self.assertRaises(ValueError):
                remaining_actions([25, 50, 600], safe, screens)


class BaseDrivingSessionFileEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.log_root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix="base_session_test_", dir=session.ROOT / "logs")))
        self.diag_root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix="base_session_test_", dir=session.ROOT / "diagnostics")))
        self.source = self.diag_root / "source_identity.py"
        self.source.write_text("FROZEN_SOURCE = 1\n", encoding="utf-8")
        self.stack.enter_context(patch.object(session, "_source_paths", return_value=[self.source]))
        self.args = argparse.Namespace(
            run_dir=self.log_root / "fresh_run", control_dir=self.diag_root / "control",
            contract=session.ROOT / "configs/base_driving/base_driving_v6_v1.json",
            stop_updates=[25, 50], train_geometry=self.diag_root / "train_geometry.json",
            validation_geometry=self.diag_root / "validation_geometry.json", seed=42,
        )
        for path in (self.args.train_geometry, self.args.validation_geometry):
            path.write_text(json.dumps({"status": "OFFLINE_VALID_LIVE_PENDING"}), encoding="utf-8")

    def tearDown(self):
        self.stack.close()

    def _prepare(self):
        with redirect_stdout(io.StringIO()):
            return session.prepare(self.args)

    @property
    def plan(self):
        return self.args.control_dir / "plan.json"

    def test_prepare_and_inspect_are_file_only_and_do_not_claim_training(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "carla" or name.startswith("training.train_"):
                raise AssertionError(f"prepare/inspect imported simulator or trainer: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), \
             patch("subprocess.Popen", side_effect=AssertionError("process dispatch forbidden")), \
             patch("subprocess.run", side_effect=AssertionError("process query forbidden")), \
             patch("socket.create_connection", side_effect=AssertionError("network connection forbidden")):
            record = self._prepare()
            inspected = session.inspect(self.plan)
        self.assertEqual(record["total_budget_updates"], 50)
        self.assertEqual(record["total_budget_episodes"], 800)
        self.assertEqual(record["source_update"], 0)
        self.assertEqual(inspected["safe_update"], 0)
        self.assertEqual(inspected["current_control"]["status"], "PREPARED_NOT_TRAINED")
        initialization = session.read(self.args.run_dir / "initialization.json")
        self.assertEqual(initialization["status"], "PREPARED_NOT_TRAINED")
        self.assertEqual(initialization["plan_sha256"], session.sha256_file(self.plan))
        self.assertFalse((self.args.run_dir / "session/launch.json").exists())
        self.assertFalse((self.args.run_dir / "session/exit.json").exists())
        self.assertFalse((self.args.run_dir / "latest.pt").exists())

    def test_web_monitor_is_bound_to_plan_and_forwarded_to_training(self):
        self.args.monitor_mode = "web"
        self.args.monitor_port = 18081
        self.args.monitor_width = 800
        record = self._prepare()
        loaded = session.load_plan(self.plan)
        self.assertEqual(loaded["web_monitor"], record["web_monitor"])
        with patch("carla.Client", side_effect=AssertionError("offline test must not connect")):
            args = session.training_args(loaded, 0, 25)
        self.assertEqual(args.monitor_mode, "web")
        self.assertEqual(args.monitor_port, 18081)
        self.assertEqual(args.monitor_width, 800)
        self.assertEqual(args.monitor_view, "overhead")
        self.assertEqual(args.base_driving_contract, record["contract_path"])
        self.assertTrue(args._base_driving_clock_owned)
        self.assertEqual(args.pretrained_checkpoint, "")
        self.assertEqual(args.resume, "")

    def test_invalid_web_monitor_is_rejected_before_creating_run(self):
        self.args.monitor_mode = "web"
        self.args.monitor_fps = float("nan")
        with self.assertRaisesRegex(ValueError, "monitor"):
            self._prepare()
        self.assertFalse(self.args.run_dir.exists())

    def test_load_rejects_budget_plan_binding_and_source_hash_changes(self):
        record = self._prepare()
        original = self.plan.read_bytes()
        self.assertEqual(session.load_plan(self.plan), record)
        for name, value in (("total_budget_updates", 600),
                            ("total_budget_episodes", 9600),
                            ("automatic_restart", True),
                            ("stop_updates", [25, 600])):
            candidate = dict(record)
            candidate[name] = value
            self.plan.write_text(json.dumps(candidate), encoding="utf-8")
            with self.subTest(field=name), self.assertRaises(ValueError):
                session.load_plan(self.plan)
            self.plan.write_bytes(original)
        self.source.write_text("FROZEN_SOURCE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            session.load_plan(self.plan)

    def test_prepare_refuses_directory_reuse_and_empty_or_implicit_budget(self):
        self._prepare()
        with self.assertRaises(FileExistsError):
            self._prepare()
        for stops in (None, [], [601]):
            args = argparse.Namespace(**vars(self.args))
            args.stop_updates = stops
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                session.prepare(args)

    def test_archive_preserves_missing_exit_and_uncommitted_tail_without_fabrication(self):
        record = self._prepare()
        run = self.args.run_dir
        log = run / "session/training_unique.log"
        log.write_text("interrupted evidence\n", encoding="utf-8")
        (run / "session/launch.json").write_text(json.dumps({
            "launch_id": "unique", "log_path": str(log),
        }), encoding="utf-8")
        (run / "checkpoint_manifest.json").write_text(json.dumps({"latest": {"update": 11}}), encoding="utf-8")
        (run / "episodes.jsonl").write_text(
            json.dumps({"update": 11, "episode_index": 15}) + "\n" +
            json.dumps({"update": 12, "episode_index": 0}) + "\n", encoding="utf-8")
        (run / "metrics.jsonl").write_text(json.dumps({"update": 11}) + "\n", encoding="utf-8")
        session.archive_session(record, reason="interrupted")
        archives = list(self.args.control_dir.glob("interrupted_session_unique_*"))
        self.assertEqual(len(archives), 1)
        archived = archives[0]
        review = session.read(archived / "boundary_review.json")
        self.assertFalse(review["exit_present"])
        self.assertEqual(review["safe_update"], 11)
        self.assertEqual(review["uncommitted_episodes"], [{"update": 12, "episode_index": 0}])
        self.assertEqual(review["uncommitted_metrics"], [])
        self.assertFalse((run / "session/exit.json").exists())
        self.assertFalse((archived / "exit.json").exists())
        self.assertFalse((archived / "original_exit.json").exists())
        self.assertEqual(session.read(archived / "original_launch.json")["launch_id"], "unique")
        self.assertTrue((archived / "episodes.jsonl").exists())
        self.assertEqual((archived / "training.log").read_text(encoding="utf-8"), "interrupted evidence\n")
        self.assertTrue((run / "episodes.jsonl").exists())

    def test_normal_screen_requires_matching_real_worker_exit_identity(self):
        self._prepare()
        directory = self.args.run_dir / "session"
        launch = {"launch_id": "current", "stop_update": 25, "actual_worker_pid": 12345}
        (directory / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        self.assertFalse(session._matching_exit(self.args.run_dir, stop=25))
        valid = {"launch_id": "current", "trainer_pid": 12345, "returncode": 0}
        for changed in ({"launch_id": "old"}, {"trainer_pid": 54321}, {"returncode": 1}):
            exit_record = {**valid, **changed}
            (directory / "exit.json").write_text(json.dumps(exit_record), encoding="utf-8")
            with self.subTest(changed=changed):
                self.assertFalse(session._matching_exit(self.args.run_dir, stop=25))
        (directory / "exit.json").write_text(json.dumps(valid), encoding="utf-8")
        self.assertFalse(session._matching_exit(self.args.run_dir, stop=50))
        self.assertTrue(session._matching_exit(self.args.run_dir, stop=25))

    def test_explicit_screen_first_recovery_archives_partial_suite_without_faking_exit(self):
        record = self._prepare()
        run, control = self.args.run_dir, self.args.control_dir
        (run / "latest.pt").write_bytes(b"same complete safe checkpoint\n")
        digest = session.sha256_file(run / "latest.pt")
        (run / "latest.pt.sha256").write_text(digest + "\n", encoding="ascii")
        partial = control / "screen_u000025"
        partial.mkdir()
        (partial / "partial_evidence.json").write_text(json.dumps({"completed_cases": 3}), encoding="utf-8")
        calls = []

        def fake_evaluate(paths, output, manifest, qualification, **kwargs):
            calls.append((str(output), kwargs.get("parallel_regression", False)))
            output.mkdir()
            (output / "qualification.json").write_text(json.dumps({"offline_mock": True}), encoding="utf-8")
            return {"reports": [{"checkpoint_sha256": digest}]}

        with patch.object(session, "safe_boundary", return_value=(25, {}, digest)), \
             patch("evaluation.evaluate_base_driving_vnext.evaluate", side_effect=fake_evaluate), \
             patch("evaluation.base_driving_qualification.select_candidate", return_value={"decision": "NO_QUALIFIED_CANDIDATE"}), \
             patch("diagnostics.v6_scratch_session.training_integrity", return_value={"offline_mock": True}), \
             patch("subprocess.Popen", side_effect=AssertionError("live dispatch forbidden")):
            with self.assertRaisesRegex(ValueError, "matching actual worker exit"):
                session.frozen_screen(record, 25)
            self.assertTrue((partial / "partial_evidence.json").exists())
            summary = session.frozen_screen(record, 25, explicit_safe_boundary_recovery=True)
        archives = list(control.glob("interrupted_screen_u000025_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(session.read(archives[0] / "partial_evidence.json"), {"completed_cases": 3})
        self.assertFalse((partial / "partial_evidence.json").exists())
        self.assertEqual([value[1] for value in calls], [True, False])
        self.assertEqual(summary["checkpoint_sha256"], digest)
        self.assertTrue(summary["explicit_safe_boundary_recovery"])
        self.assertFalse(summary["normal_training_matching_exit_0"])
        self.assertFalse((run / "session/exit.json").exists())
        self.assertTrue((control / "u000025_summary.json").exists())

    def test_active_worker_lease_rejects_before_any_process_query_or_archive(self):
        record = self._prepare()
        launch = self.args.run_dir / "session/launch.json"
        launch.write_text(json.dumps({"launch_id": "active", "actual_worker_pid": 12345}), encoding="utf-8")
        original = launch.read_bytes()
        with session.RunSessionOwner(self.args.run_dir / "worker_guard"), \
             patch("subprocess.run", side_effect=AssertionError("lease probe must reject before process audit")), \
             patch.object(session, "archive_session") as archive:
            with self.assertRaises(RuntimeError):
                session._assert_worker_idle(record, self.plan)
            archive.assert_not_called()
        self.assertEqual(launch.read_bytes(), original)

    def test_uuid_process_identity_rejects_live_worker_but_not_reused_pid(self):
        if session.os.name != "nt":
            self.skipTest("Windows process identity audit branch")
        record = self._prepare()
        launch = self.args.run_dir / "session/launch.json"
        launch.write_text(json.dumps({"launch_id": "exact_uuid", "actual_worker_pid": 12345,
                                      "spawned_pid": 12344}), encoding="utf-8")
        original = launch.read_bytes()
        current_command = f'python.exe -u -m utils.base_driving_session worker --plan "{self.plan}" --launch-id exact_uuid'
        for command, should_reject in ((current_command, True),
                                       ("python.exe -m unrelated.module --job exact_uuid", False)):
            response = SimpleNamespace(returncode=0, stdout=json.dumps([
                {"ProcessId": 12345, "CommandLine": command},
            ]))
            with self.subTest(command=command), patch("subprocess.run", return_value=response) as process_audit:
                if should_reject:
                    with self.assertRaisesRegex(RuntimeError, "UUID-matched worker"):
                        session._assert_worker_idle(record, self.plan)
                else:
                    session._assert_worker_idle(record, self.plan)
                process_audit.assert_called_once()
            self.assertEqual(launch.read_bytes(), original)

    def test_controller_worker_precheck_failure_preserves_old_session_and_status(self):
        record = self._prepare()
        status = self.args.control_dir / "status.json"
        launch = self.args.run_dir / "session/launch.json"
        status.write_text(json.dumps({"status": "TRAINING", "old_evidence": True}), encoding="utf-8")
        launch.write_text(json.dumps({"launch_id": "still_active", "actual_worker_pid": 12345}), encoding="utf-8")
        original_status, original_launch = status.read_bytes(), launch.read_bytes()
        with patch.object(session, "_assert_worker_idle", side_effect=RuntimeError("previous UUID-matched worker still active")), \
             patch.object(session, "require_geometry") as geometry, \
             patch.object(session, "archive_session") as archive, \
             patch.object(session, "supervise") as supervise, \
             patch.object(session, "frozen_screen") as screen:
            with self.assertRaisesRegex(RuntimeError, "worker still active"):
                session.controller(self.plan, resume=True)
            geometry.assert_not_called()
            archive.assert_not_called()
            supervise.assert_not_called()
            screen.assert_not_called()
        self.assertEqual(status.read_bytes(), original_status)
        self.assertEqual(launch.read_bytes(), original_launch)
        failures = list(self.args.control_dir.glob("precheck_failed_*.json"))
        self.assertEqual(len(failures), 1)
        self.assertFalse(session.read(failures[0])["session_evidence_replaced"])


if __name__ == "__main__":
    unittest.main()
