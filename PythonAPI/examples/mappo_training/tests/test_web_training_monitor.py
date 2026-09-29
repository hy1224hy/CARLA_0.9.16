import json
import time
import unittest
from unittest.mock import Mock
from urllib.request import urlopen

import numpy as np

from visualization.web_training_monitor import WebTrainingMonitor


class WebTrainingMonitorTests(unittest.TestCase):
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
        self.assertTrue(monitor.submit({0: image}, {"step": 2}))
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
        self.assertEqual(monitor._store.status()["state"], "camera_error")
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


if __name__ == "__main__":
    unittest.main()
