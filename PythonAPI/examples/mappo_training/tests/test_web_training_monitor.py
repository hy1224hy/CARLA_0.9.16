import json
import time
import unittest
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
            {"width": 100},
            {"jpeg_quality": 0},
            {"jpeg_quality": 96},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WebTrainingMonitor(**kwargs)

    def test_close_is_idempotent_and_closed_monitor_cannot_restart(self):
        monitor = WebTrainingMonitor(port=0)
        monitor.close()
        monitor.close()
        with self.assertRaises(RuntimeError):
            monitor.start()


if __name__ == "__main__":
    unittest.main()
