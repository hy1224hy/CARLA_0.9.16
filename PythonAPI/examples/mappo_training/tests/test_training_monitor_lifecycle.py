"""Offline checks for monitor transitions owned by the actual training loop.

CARLA and the HTTP monitor are replaced with small fakes.  Policy inference,
PPO, metrics validation, checkpoint writes, and the training lifecycle stay real.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training import checkpoint_protocol
from training import train_no_rllib_mappo_robust_v2 as trainer
from training.metrics_protocol import reward_components, reward_diagnostics
from utils.training_metrics_recorder import TrainingMetricsRecorder


class TrainingMonitorResultTests(unittest.TestCase):
    def result(self, directory, *, manifest=None, started=0, outcomes=None, metrics=None):
        return trainer.training_monitor_result(
            args=SimpleNamespace(run_id="offline", updates=50),
            manifest=manifest,
            run_dir=directory,
            total_episodes=started,
            outcomes=outcomes or {"success": 0, "timeout": 0},
            last_metrics_payload=metrics,
        )

    def test_safe_update_comes_from_saved_checkpoint_record(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"retained checkpoint")
            result = self.result(
                directory,
                manifest={"latest": {"path": "latest.pt", "update": 7}},
                metrics={"update": 99, "scalars": {
                    "episode/team/success_rate": 0.5,
                    "timing/update_seconds": 2.0,
                    "train/update": 99,
                }},
            )
            self.assertEqual(result["safe_update"], 7)
            self.assertEqual(result["checkpoint"], str(checkpoint))
            self.assertEqual(result["planned_updates"], 50)
            self.assertEqual(result["last_metrics"], {
                "episode/team/success_rate": 0.5,
                "timing/update_seconds": 2.0,
            })

    def test_checkpoint_requires_both_manifest_record_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            for manifest in (None, {}, {"latest": {"path": "latest.pt", "update": 4}}):
                with self.subTest(manifest=manifest):
                    result = self.result(directory, manifest=manifest)
                    self.assertIsNone(result["safe_update"])
                    self.assertIsNone(result["checkpoint"])
            checkpoint.write_bytes(b"unregistered file")
            result = self.result(directory, manifest={"latest": None})
            self.assertIsNone(result["safe_update"])
            self.assertIsNone(result["checkpoint"])

    def test_persisted_manifest_takes_precedence_over_uncommitted_memory_record(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"successfully committed update 3")
            disk_manifest = {"latest": {
                "path": "latest.pt", "update": 3,
                "sha256": checkpoint_protocol.sha256_file(checkpoint),
            }}
            checkpoint_protocol.atomic_write_json(
                disk_manifest, Path(directory) / checkpoint_protocol.MANIFEST_NAME
            )
            result = self.result(directory, manifest={"latest": {
                "path": "latest.pt", "update": 4,
            }})
            self.assertEqual(result["safe_update"], 3)
            self.assertEqual(result["checkpoint"], str(checkpoint))

    def test_persisted_manifest_digest_must_match_retained_latest_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"old committed update 3")
            disk_manifest = {"latest": {
                "path": "latest.pt", "update": 3,
                "sha256": checkpoint_protocol.sha256_file(checkpoint),
            }}
            checkpoint_protocol.atomic_write_json(
                disk_manifest, Path(directory) / checkpoint_protocol.MANIFEST_NAME
            )
            # Updating the checkpoint before its manifest commit leaves the
            # durable record referring to bytes that no longer exist.
            checkpoint.write_bytes(b"new uncommitted update 4")
            result = self.result(directory, manifest={"latest": {
                "path": "latest.pt", "update": 4,
                "sha256": checkpoint_protocol.sha256_file(checkpoint),
            }})
            self.assertIsNone(result["safe_update"])
            self.assertIsNone(result["checkpoint"])

    def test_interrupted_episode_does_not_dilute_completed_success_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.result(directory, started=5, outcomes={
                "success": 2, "collision": 1, "timeout": 1,
                "offroad": 0, "road_violation": 0,
            })
            self.assertEqual(result["started_episodes"], 5)
            self.assertEqual(result["completed_episodes"], 4)
            self.assertEqual(result["success_rate"], 0.5)
            empty = self.result(directory, started=1)
            self.assertEqual(empty["completed_episodes"], 0)
            self.assertIsNone(empty["success_rate"])

    def test_web_training_defaults_to_smoothed_overhead_view(self):
        args = trainer.build_parser().parse_args(["--monitor-mode", "web"])
        self.assertEqual(args.monitor_view, "overhead")
        self.assertGreater(args.monitor_smoothing_seconds, 0)


class _FakeMonitor:
    url = "http://offline.invalid/"

    def __init__(self, events, **settings):
        self.events = events
        self.settings = settings
        self.finished = False
        self.final_phase = None
        self.result = None
        self.phases = []
        self.frames = []

    def start(self):
        self.events.append("monitor.start")

    def set_phase(self, phase, **fields):
        self.phases.append((phase, fields))
        self.events.append("phase." + phase)

    def begin_episode(self, **fields):
        self.events.append("monitor.begin_episode")

    def prepare_camera(self, env):
        self.events.append("monitor.prepare_camera")

    def submit(self, images, status):
        self.frames.append(status.copy())
        self.events.append("monitor.submit")
        return True

    def stop_camera(self):
        self.events.append("monitor.stop_camera")

    def finish(self, phase, *, result=None, **fields):
        self.finished = True
        self.final_phase = phase
        self.result = result
        self.events.append("monitor.finish")

    def close(self):
        self.events.append("monitor.close")


class _FakeEnvironment:
    """One valid truncated transition per episode, or a failure during step."""

    def __init__(self, events, *, step_error=None, close_error=None, **settings):
        self.events = events
        self.step_error = step_error
        self.close_error = close_error
        self.num_agents = settings["num_agents"]
        self.reward_gamma = settings["reward_gamma"]
        self.reward_version = settings["reward_version"]
        self.obs_dim = trainer.observation_dim_for_reward(self.reward_version)
        self._last_world_frame = 100
        self.client = SimpleNamespace(
            get_client_version=lambda: "offline",
            get_server_version=lambda: "offline",
        )

    def get_global_state(self):
        return np.zeros(self.obs_dim * self.num_agents, dtype=np.float32)

    def load_scenario_by_name(self, name, directory):
        return {"name": name}

    def set_scenario(self, scenario):
        self.events.append("env.set_scenario")

    def _observations(self):
        return {aid: np.zeros(self.obs_dim, dtype=np.float32) for aid in (0, 1)}

    def _info(self, done=False):
        return {
            "active_mask": {aid: not done for aid in (0, 1)},
            "terminated": {aid: False for aid in (0, 1)},
            "truncated": {aid: done for aid in (0, 1)},
            "episode_end_reason": "max_steps" if done else "running",
            "offroad_terminated": {aid: False for aid in (0, 1)},
            "first_offroad_evidence": {aid: None for aid in (0, 1)},
            "max_offroad_consecutive_ticks": {aid: 0 for aid in (0, 1)},
            "max_footprint_outside_fraction": {aid: 0.0 for aid in (0, 1)},
            "reward_details": {
                aid: {name: 0.0 for name in reward_components(self.reward_version)}
                for aid in (0, 1)
            },
            "reward_diagnostics": {
                aid: {name: 0.0 for name in reward_diagnostics(self.reward_version)}
                for aid in (0, 1)
            },
        }

    def reset(self):
        self.events.append("env.reset")
        return self._observations(), self._info()

    def step(self, actions):
        self.events.append("env.step")
        if self.step_error is not None:
            raise self.step_error
        self._last_world_frame += 1
        return (
            self._observations(), {aid: 0.0 for aid in (0, 1)},
            {aid: True for aid in (0, 1)}, self._info(done=True),
        )

    def get_camera_images(self):
        return {}

    def close(self):
        self.events.append("env.close")
        if self.close_error is not None:
            raise self.close_error


class TrainingMonitorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.events = []
        self.monitors = []
        source_project = Path(checkpoint_protocol.__file__).resolve().parents[1]
        self.args = trainer.build_parser().parse_args([
            "--cpu", "--updates", "1", "--rollout-episodes", "2",
            "--max-steps", "1", "--hidden-dim", "8", "--ppo-epochs", "1",
            "--minibatch-size", "2", "--min-transitions-per-update", "2",
            "--tb-histogram-interval", "0", "--monitor-mode", "web",
            "--run-id", "offline-monitor",
            "--save-dir", str(Path(self.directory.name) / "run"),
            "--scenario-dir", str(source_project / "data" / "training_scenarios_town05_v1"),
        ])
        self.args._explicit_cli_options = set()

    def _run(self, *, step_error=None, close_error=None, constructor_error=None,
             recorder_error=None, save_error=None, manifest_error=None):
        def monitor_factory(**settings):
            monitor = _FakeMonitor(self.events, **settings)
            self.monitors.append(monitor)
            return monitor

        def environment_factory(**settings):
            self.events.append("env.create")
            if constructor_error is not None:
                raise constructor_error
            return _FakeEnvironment(
                self.events, step_error=step_error, close_error=close_error, **settings
            )

        def recorder_factory(**settings):
            if recorder_error is not None:
                raise recorder_error
            recorder = TrainingMetricsRecorder(**settings)
            original_close = recorder.close

            def close():
                self.events.append("metrics.close")
                original_close()

            recorder.close = close
            return recorder

        original_save = trainer.save_checkpoint_file
        save_attempts = 0

        def save_checkpoint(payload, destination, **settings):
            nonlocal save_attempts
            save_attempts += 1
            if save_error is not None and save_attempts == 2:
                raise save_error
            return original_save(payload, destination, **settings)

        original_write_json = trainer.atomic_write_json
        manifest_attempts = 0

        def write_json(payload, destination):
            nonlocal manifest_attempts
            if Path(destination).name == checkpoint_protocol.MANIFEST_NAME:
                manifest_attempts += 1
                if manifest_error is not None and manifest_attempts > 1:
                    raise manifest_error
            return original_write_json(payload, destination)

        # Staging overlays contain only changed files.  Limit the fingerprint
        # to the trainer under test instead of depending on a complete checkout.
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(trainer, "training_source_paths", return_value=[Path(trainer.__file__)]))
            stack.enter_context(patch.object(trainer, "get_git_metadata", return_value={
                "commit": "offline", "dirty": False, "dirty_paths": [],
            }))
            stack.enter_context(patch.object(trainer, "CarlaMultiAgentEnv", side_effect=environment_factory))
            stack.enter_context(patch.object(trainer, "TrainingMetricsRecorder", side_effect=recorder_factory))
            stack.enter_context(patch.object(trainer, "save_checkpoint_file", side_effect=save_checkpoint))
            stack.enter_context(patch.object(trainer, "atomic_write_json", side_effect=write_json))
            stack.enter_context(patch("visualization.web_training_monitor.WebTrainingMonitor", side_effect=monitor_factory))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            trainer.train(self.args)

    def _assert_cleanup_before_result(self):
        positions = [self.events.index(name) for name in (
            "monitor.stop_camera", "env.close", "metrics.close",
            "monitor.finish", "monitor.close",
        )]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.events.count("monitor.finish"), 1)

    def test_actual_update_switches_episodes_optimizes_saves_and_retains_result(self):
        self._run()
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "completed")
        self.assertEqual(monitor.settings["view"], "overhead")
        self.assertEqual(monitor.result["safe_update"], 1)
        self.assertEqual(monitor.result["started_episodes"], 2)
        self.assertEqual(monitor.result["completed_episodes"], 2)
        self.assertEqual(monitor.result["outcomes"]["timeout"], 2)
        self.assertTrue(Path(monitor.result["checkpoint"]).is_file())
        self.assertTrue(Path(monitor.result["metrics_path"]).is_file())
        self.assertEqual([frame["world_frame"] for frame in monitor.frames], [101, 102])
        self.assertEqual([frame["episode"] for frame in monitor.frames], [1, 2])
        reset_positions = [i for i, value in enumerate(self.events) if value == "env.reset"]
        self.assertEqual(len(reset_positions), 2)
        for position in reset_positions:
            self.assertEqual(self.events[position - 1], "monitor.begin_episode")
            self.assertEqual(self.events[position + 1], "phase.rollout")
        phases = [phase for phase, fields in monitor.phases]
        self.assertIn("starting", phases)
        self.assertIn("optimizing", phases)
        self.assertGreater(self.events.index("phase.saving", self.events.index("phase.optimizing")), self.events.index("phase.optimizing"))
        self._assert_cleanup_before_result()

    def test_interrupt_during_episode_keeps_only_latest_safe_update(self):
        self._run(step_error=KeyboardInterrupt())
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "interrupted")
        self.assertEqual(monitor.result["safe_update"], 0)
        self.assertEqual(monitor.result["started_episodes"], 1)
        self.assertEqual(monitor.result["completed_episodes"], 0)
        self.assertIsNone(monitor.result["success_rate"])
        self.assertTrue(Path(monitor.result["checkpoint"]).is_file())
        self._assert_cleanup_before_result()

    def test_interrupt_before_first_v6_checkpoint_reports_no_safe_update(self):
        self.args.reward_version = "v6"
        self.args.lr_actor_final = 5e-5
        self.args.lr_critic_final = 5e-5
        self.args.critic_probe_batch_size = 32
        self.args.lidar_range_m = 20.0
        self.args.lr_decay_horizon = 10
        self._run(step_error=KeyboardInterrupt())
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "interrupted")
        self.assertIsNone(monitor.result["safe_update"])
        self.assertIsNone(monitor.result["checkpoint"])
        self.assertEqual(monitor.result["started_episodes"], 1)
        self.assertEqual(monitor.result["completed_episodes"], 0)
        self._assert_cleanup_before_result()

    def test_step_error_is_retained_and_original_error_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "offline simulator failure"):
            self._run(step_error=RuntimeError("offline simulator failure"))
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "error")
        self.assertIn("offline simulator failure", monitor.result["reason"])
        self.assertEqual(monitor.result["completed_episodes"], 0)
        self._assert_cleanup_before_result()

    def test_cleanup_error_changes_completed_training_to_error_result(self):
        with self.assertRaisesRegex(RuntimeError, "training cleanup failed"):
            self._run(close_error=RuntimeError("offline cleanup failure"))
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "error")
        self.assertIn("offline cleanup failure", monitor.result["reason"])
        self.assertEqual(monitor.result["safe_update"], 1)
        self._assert_cleanup_before_result()

    def test_failed_update_checkpoint_keeps_manifest_safe_update(self):
        with self.assertRaisesRegex(OSError, "offline checkpoint failure"):
            self._run(save_error=OSError("offline checkpoint failure"))
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "error")
        self.assertEqual(monitor.result["completed_episodes"], 2)
        # PPO has processed update 1, but its save failed.  The retained result
        # must continue to name the successfully committed update 0.
        self.assertEqual(monitor.result["safe_update"], 0)
        checkpoint, digest = checkpoint_protocol.load_checkpoint_file(
            monitor.result["checkpoint"], map_location="cpu"
        )
        self.assertEqual(checkpoint["update"], 0)
        self.assertIn("offline checkpoint failure", monitor.result["reason"])
        self._assert_cleanup_before_result()

    def test_manifest_commit_failure_does_not_publish_unregistered_update(self):
        with self.assertRaisesRegex(OSError, "offline manifest failure"):
            self._run(manifest_error=OSError("offline manifest failure"))
        monitor = self.monitors[0]
        run_dir = Path(self.args.save_dir)
        persisted = json.loads(
            (run_dir / checkpoint_protocol.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        checkpoint, digest = checkpoint_protocol.load_checkpoint_file(
            run_dir / checkpoint_protocol.LATEST_CHECKPOINT_NAME,
            map_location="cpu",
        )
        self.assertEqual(persisted["latest"]["update"], 0)
        self.assertEqual(checkpoint["update"], 1)
        self.assertNotEqual(persisted["latest"]["sha256"], digest)
        self.assertEqual(monitor.final_phase, "error")
        self.assertIsNone(monitor.result["safe_update"])
        self.assertIsNone(monitor.result["checkpoint"])
        self.assertIn("offline manifest failure", monitor.result["reason"])
        self._assert_cleanup_before_result()

    def test_environment_initialization_failure_keeps_error_result_available(self):
        with self.assertRaisesRegex(RuntimeError, "offline connection failure"):
            self._run(constructor_error=RuntimeError("offline connection failure"))
        monitor = self.monitors[0]
        self.assertLess(self.events.index("monitor.start"), self.events.index("env.create"))
        self.assertEqual(monitor.final_phase, "error")
        self.assertIn("offline connection failure", monitor.result["reason"])
        self.assertEqual(self.events.count("monitor.finish"), 1)
        self.assertEqual(self.events[-1], "monitor.close")
        self.assertFalse(hasattr(self.args, "_web_monitor_instance"))

    def test_metrics_initialization_failure_cleans_environment_before_result(self):
        with self.assertRaisesRegex(RuntimeError, "offline metrics failure"):
            self._run(recorder_error=RuntimeError("offline metrics failure"))
        monitor = self.monitors[0]
        self.assertEqual(monitor.final_phase, "error")
        self.assertIn("offline metrics failure", monitor.result["reason"])
        self.assertLess(self.events.index("env.close"), self.events.index("monitor.finish"))
        self.assertEqual(self.events.count("monitor.finish"), 1)
        self.assertEqual(self.events[-1], "monitor.close")


if __name__ == "__main__":
    unittest.main()
