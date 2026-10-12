import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib.request import urlopen

import numpy as np

from visualization.web_training_monitor import WebTrainingMonitor


class WebTrainingMonitorTests(unittest.TestCase):
    def wait_for(self, condition, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.01)
        self.fail("monitor condition timed out")

    def test_submit_encodes_latest_frame_and_serves_status(self):
        monitor = WebTrainingMonitor(
            host="127.0.0.1", port=0, fps=1000, width=320, jpeg_quality=60
        )
        monitor.start()
        self.addCleanup(monitor.close)

        bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        self.assertTrue(
            monitor.submit(
                {0: bgr, 1: bgr.copy()},
                {"run_id": "test", "update": 3, "step": 7},
            )
        )

        deadline = time.time() + 3
        payload = {}
        while time.time() < deadline:
            with urlopen(monitor.url + "status.json", timeout=1) as response:
                payload = json.load(response)
            if payload.get("sequence", 0) >= 1:
                break
            time.sleep(0.02)

        self.assertEqual(payload["state"], "running")
        self.assertEqual(payload["run_id"], "test")
        self.assertEqual(payload["agents"], [0, 1])
        self.assertEqual(payload["view"], "vehicle")
        with urlopen(monitor.url + "healthz", timeout=1) as response:
            self.assertEqual(response.read(), b"ok\n")
        with urlopen(monitor.url + "stream.mjpg", timeout=1) as response:
            self.assertIn(b"--frame", response.read(128))

    def test_full_queue_replaces_old_frame_without_waiting(self):
        monitor = WebTrainingMonitor(port=0, fps=1000)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        self.assertTrue(monitor.submit({0: image}, {"step": 1}))
        monitor._last_submit = 0.0
        started = time.perf_counter()
        self.assertTrue(monitor.submit({0: image.copy()}, {"step": 2}))
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertEqual(monitor._dropped, 1)

    def test_invalid_settings_fail_early(self):
        for kwargs in (
            {"port": -1},
            {"port": 65536},
            {"fps": 0},
            {"fps": float("nan")},
            {"fps": float("inf")},
            {"width": 100},
            {"jpeg_quality": 0},
            {"jpeg_quality": 96},
            {"view": "invalid"},
            {"view": "overhead", "overhead_height": -1},
            {"smoothing_seconds": -1},
            {"smoothing_seconds": float("nan")},
            {"service_dir": "service", "output_dir": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WebTrainingMonitor(**kwargs)

    def test_overhead_is_single_panel_and_preserves_agent_status(self):
        monitor = WebTrainingMonitor(port=0, width=320, view="overhead")
        self.addCleanup(monitor.close)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        # No fallback to vehicle view while waiting for the first overhead frame.
        self.assertFalse(monitor.submit({0: image, 1: image}))
        monitor._camera.latest = (image, 42)
        self.assertTrue(monitor.submit({0: image, 1: image}, {"step": 5}))
        frame = monitor._pending.get_nowait()
        self.assertEqual(list(frame.images), [-1])
        self.assertEqual(frame.status["view"], "overhead")
        self.assertEqual(frame.status["agents"], [0, 1])
        self.assertEqual(frame.status["camera_frame"], 42)
        canvas = monitor._compose(frame.images, frame.status)
        self.assertEqual(canvas.size, (320, 284))
        self.assertEqual(canvas.getpixel((200, 200)), (255, 0, 0))

    def test_camera_failure_is_isolated_and_retried_after_reset(self):
        monitor = WebTrainingMonitor(port=0, view="overhead")
        self.addCleanup(monitor.close)
        monitor._camera = Mock()
        monitor._camera.update.side_effect = RuntimeError("camera unavailable")
        monitor.prepare_camera(Mock())
        monitor.prepare_camera(Mock())
        monitor._camera.update.assert_called_once()
        self.assertEqual(monitor._store.status()["monitor_state"], "camera_error")
        monitor._camera.close.assert_called_once()
        monitor.reset_camera()
        monitor._camera.update.side_effect = None
        monitor.prepare_camera(Mock())
        self.assertEqual(monitor._camera.update.call_count, 2)

    def test_vehicle_view_does_not_access_carla(self):
        monitor = WebTrainingMonitor(port=0)
        env = Mock()
        monitor.prepare_camera(env)
        monitor.reset_camera()
        monitor.close()
        self.assertEqual(env.mock_calls, [])

    def test_close_is_idempotent_and_closed_monitor_cannot_restart(self):
        monitor = WebTrainingMonitor(port=0)
        monitor.close()
        monitor.close()
        with self.assertRaises(RuntimeError):
            monitor.start()

    def test_status_advances_without_image_and_while_image_is_throttled(self):
        monitor = WebTrainingMonitor(port=0, fps=0.1)
        self.addCleanup(monitor.close)
        self.assertFalse(monitor.submit({0: None}, {"step": 1}))
        self.assertEqual(monitor._store.status()["step"], 1)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        self.assertTrue(monitor.submit({0: image}, {"step": 2}))
        self.assertFalse(monitor.submit({0: image.copy()}, {"step": 3}))
        monitor.set_phase("optimizing", update=8)
        self.assertEqual(monitor._store.status()["step"], 3)
        self.assertEqual(monitor._store.status()["phase"], "optimizing")
        self.assertEqual(monitor._store.status()["update"], 8)

    def test_delayed_encode_cannot_restore_rollout_phase(self):
        monitor = WebTrainingMonitor(port=0, fps=1000, width=320)
        entered, release = threading.Event(), threading.Event()
        compose = monitor._compose
        def blocked(images, status):
            entered.set()
            release.wait(3)
            return compose(images, status)
        monitor._compose = blocked
        monitor.start()
        self.addCleanup(monitor.close)
        self.addCleanup(release.set)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        monitor.set_phase("rollout", update=1)
        monitor.submit({0: image}, {"episode": 2, "step": 7})
        self.assertTrue(entered.wait(2))
        monitor.set_phase("optimizing", update=1)
        release.set()
        self.wait_for(lambda: monitor._store.status()["sequence"] == 1)
        self.assertEqual(monitor._store.status()["phase"], "optimizing")
        self.assertEqual(monitor._store.status()["frame_step"], 7)

    def test_episode_reset_rejects_delayed_old_encode_and_marks_held_frame(self):
        monitor = WebTrainingMonitor(port=0, fps=1000, width=320)
        monitor.start()
        self.addCleanup(monitor.close)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        monitor.begin_episode(episode=0, scenario="old")
        monitor.set_phase("rollout")
        monitor.submit({0: image}, {"episode": 0, "scenario": "old", "step": 1})
        self.wait_for(lambda: monitor._store.status()["sequence"] == 1)
        entered, release = threading.Event(), threading.Event()
        compose = monitor._compose
        def blocked(images, status):
            entered.set()
            release.wait(3)
            return compose(images, status)
        monitor._compose = blocked
        self.addCleanup(release.set)
        monitor._last_submit = 0
        monitor.submit({0: image.copy()}, {"episode": 0, "scenario": "old", "step": 2})
        self.assertTrue(entered.wait(2))
        monitor.begin_episode(episode=1, scenario="new", step=0)
        self.assertFalse(monitor._store.status()["frame_current"])
        release.set()
        self.assertTrue(monitor._idle.wait(2))
        self.assertEqual(monitor._store.status()["sequence"], 1)
        self.assertEqual(monitor._store.status()["scenario"], "new")
        self.assertEqual(monitor._store.status()["frame_scenario"], "old")
        monitor.set_phase("rollout")
        monitor.submit({0: image.copy()}, {"episode": 1, "scenario": "new", "step": 1})
        self.wait_for(lambda: monitor._store.status()["sequence"] == 2)
        self.assertTrue(monitor._store.status()["frame_current"])

    def test_overhead_frame_dedup_does_not_drop_progress(self):
        monitor = WebTrainingMonitor(port=0, fps=1000, view="overhead")
        self.addCleanup(monitor.close)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        monitor._camera.latest = (image, 40)
        self.assertTrue(monitor.submit({0: None}, {"step": 1}))
        monitor._last_submit = 0
        self.assertFalse(monitor.submit({0: None}, {"step": 2}))
        self.assertEqual(monitor._store.status()["step"], 2)
        monitor._camera.latest = (image, 41)
        self.assertTrue(monitor.submit({0: None}, {"step": 3}))

    def test_finish_drains_image_freezes_result_and_writes_reopenable_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = WebTrainingMonitor(port=0, fps=1000, width=320, output_dir=directory)
            monitor.start()
            self.addCleanup(monitor.close)
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[:, :, 2] = 255
            monitor.submit({0: image}, {"run_id": "final-test", "update": 2})
            monitor.finish("completed", result={"completed_update": 2}, message="done")
            self.assertTrue(monitor.finished)
            monitor.finish("error", result={"completed_update": 0})
            monitor.set_phase("rollout")
            self.assertFalse(monitor.submit({0: image.copy()}, {"update": 3}))
            self.assertEqual(monitor._store.status()["phase"], "completed")
            root = Path(directory)
            self.assertEqual(json.loads((root / "result.json").read_text(encoding="utf-8"))["result"], {"completed_update": 2})
            self.assertTrue((root / "latest.jpg").read_bytes().startswith(b"\xff\xd8"))
            page = (root / "result.html").read_text(encoding="utf-8")
            self.assertIn("data:image/jpeg;base64,", page)
            self.assertIn("训练已完成", page)
            with urlopen(monitor.url + "latest.jpg", timeout=1) as response:
                self.assertTrue(response.read().startswith(b"\xff\xd8"))
            monitor.close()

    def test_heartbeat_advances_without_frames(self):
        monitor = WebTrainingMonitor(port=0)
        monitor.start()
        self.addCleanup(monitor.close)
        first = monitor._store.status()["heartbeat_at"]
        self.wait_for(lambda: monitor._store.status()["heartbeat_at"] > first + 0.5)

    def test_reset_discards_pending_work_without_delaying_finish(self):
        monitor = WebTrainingMonitor(port=0)
        self.addCleanup(monitor.close)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        monitor.submit({0: image}, {"episode": 0})
        self.assertFalse(monitor._idle.is_set())
        monitor.begin_episode(episode=1)
        self.assertTrue(monitor._idle.is_set())
        self.assertTrue(monitor._pending.empty())

    def test_status_hot_path_does_not_encode_jpeg(self):
        monitor = WebTrainingMonitor(port=0)
        self.addCleanup(monitor.close)
        monitor._store._jpeg = b"some already encoded bytes"
        with unittest.mock.patch("visualization.web_training_monitor.base64.b64encode", side_effect=AssertionError("hot path encoded JPEG")):
            monitor.set_phase("optimizing", update=2)
            self.assertEqual(monitor._store.status()["update"], 2)

    def test_mjpeg_stream_delivers_next_frame_without_reopening_page(self):
        monitor = WebTrainingMonitor(port=0, fps=10, width=320)
        monitor.start()
        self.addCleanup(monitor.close)
        first = np.zeros((12, 12, 3), dtype=np.uint8)
        first[:, :, 2] = 255
        monitor.submit({0: first}, {"step": 1})
        self.wait_for(lambda: monitor._store.status()["sequence"] == 1)
        def read_frame(response):
            length = None
            while True:
                line = response.readline()
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
                if line == b"\r\n" and length is not None:
                    break
            return response.read(length)
        with urlopen(monitor.url + "stream.mjpg", timeout=2) as response:
            first_jpeg = read_frame(response)
            second = np.zeros((12, 12, 3), dtype=np.uint8)
            second[:, :, 1] = 255
            monitor._last_submit = 0
            monitor.submit({0: second}, {"step": 2})
            second_jpeg = read_frame(response)
            self.assertNotEqual(first_jpeg, second_jpeg)
        self.assertEqual(monitor._store.status()["frame_step"], 2)

    def test_snapshot_write_error_is_reported_without_stopping_monitor(self):
        monitor = WebTrainingMonitor(port=0)
        monitor._persist = Mock(side_effect=OSError("disk unavailable"))
        monitor.start()
        self.addCleanup(monitor.close)
        self.wait_for(lambda: monitor._store.status().get("monitor_state") == "snapshot_error")
        monitor.set_phase("optimizing")
        self.assertEqual(monitor._store.status()["phase"], "optimizing")


if __name__ == "__main__":
    unittest.main()
