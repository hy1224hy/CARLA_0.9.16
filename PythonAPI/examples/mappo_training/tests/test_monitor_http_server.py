import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from visualization.monitor_http_server import (
    PROTOCOL, SnapshotDirectory, atomic_json, process_alive, read_json,
    request_json, snapshot_status, stop_service,
)
from visualization.web_training_monitor import WebTrainingMonitor


class MonitorHTTPServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.service_dir = self.root / "service"
        self.monitors = []
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.cleanup_service)

    def cleanup_service(self):
        for monitor in self.monitors:
            monitor.close()
        stop_service(self.service_dir)
        deadline = time.monotonic() + 3
        while (self.service_dir / "server.json").exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        for monitor in self.monitors:
            child = monitor._server_process
            if child is not None:
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=3)

    def new_monitor(self, run="run1", port=0):
        monitor = WebTrainingMonitor(
            port=port, fps=1000, width=320,
            output_dir=str(self.root / run), service_dir=str(self.service_dir),
        )
        self.monitors.append(monitor)
        monitor.set_phase("starting", run_id=run)
        monitor.start()
        return monitor

    def wait_for_status(self, monitor, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = request_json(monitor.url + "status.json")
            if predicate(status):
                return status
            time.sleep(0.03)
        self.fail("HTTP snapshot did not reach expected state")

    def test_service_survives_producer_close_and_new_run_reuses_same_server(self):
        monitor = self.new_monitor()
        info = request_json(monitor.url + "server-info")
        self.assertEqual(info["protocol"], PROTOCOL)
        self.assertNotEqual(info["server_pid"], os.getpid())
        self.assertNotIn("token", info)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        monitor.submit({0: image}, {"update": 2, "episode": 4, "step": 5, "scenario": "Town05"})
        monitor.finish("completed", result={"completed_update": 2})
        monitor.close()
        self.wait_for_status(monitor, lambda s: s["phase"] == "completed")
        with urlopen(monitor.url + "latest.jpg", timeout=1) as response:
            self.assertTrue(response.read().startswith(b"\xff\xd8"))
        with urlopen(monitor.url, timeout=1) as response:
            self.assertIn("训练监控", response.read().decode("utf-8"))
        next_monitor = self.new_monitor("run2")
        self.assertEqual(next_monitor.port, monitor.port)
        next_info = request_json(next_monitor.url + "server-info")
        self.assertEqual(next_info["server_pid"], info["server_pid"])
        next_monitor.set_phase("optimizing", update=9)
        status = self.wait_for_status(next_monitor, lambda s: s.get("phase") == "optimizing")
        self.assertEqual(status["run_id"], "run2")
        self.assertNotEqual(status["session_id"], monitor.session_id)

    def test_another_active_producer_cannot_take_over(self):
        active = self.new_monitor()
        self.wait_for_status(active, lambda s: s.get("run_id") == "run1")
        with self.assertRaisesRegex(RuntimeError, "still active"):
            self.new_monitor("run2")
        status = request_json(active.url + "status.json")
        self.assertEqual(status["session_id"], active.session_id)

    def test_simultaneous_start_uses_one_server_and_one_active_producer(self):
        attempts = []
        barrier = threading.Barrier(2)
        def start(run):
            monitor = WebTrainingMonitor(
                port=0, output_dir=str(self.root / run), service_dir=str(self.service_dir),
            )
            self.monitors.append(monitor)
            barrier.wait(2)
            try:
                monitor.start()
                attempts.append((monitor, None))
            except Exception as error:
                attempts.append((monitor, error))
        threads = [threading.Thread(target=start, args=(run,)) for run in ("run1", "run2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(8)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        successful = [monitor for monitor, error in attempts if error is None]
        errors = [error for monitor, error in attempts if error is not None]
        self.assertEqual(len(successful), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("still active", str(errors[0]))
        self.assertEqual(len([m for m in self.monitors if m._server_process is not None]), 1)

    def test_browser_sessions_count_is_independent_of_frame_updates(self):
        monitor = self.new_monitor()
        first = request_json(monitor.url + "status.json?client=browser-1")
        second = request_json(monitor.url + "status.json?client=browser-2")
        self.assertEqual(first["clients"], 1)
        self.assertEqual(second["clients"], 2)

    def test_unrelated_http_port_is_refused_and_left_running(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                body = b'{"unrelated": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                self.new_monitor(port=server.server_port)
            self.assertEqual(request_json(f"http://127.0.0.1:{server.server_port}/"), {"unrelated": True})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_other_monitor_directory_cannot_be_reused(self):
        monitor = self.new_monitor()
        wrong = WebTrainingMonitor(
            port=monitor.port, output_dir=str(self.root / "wrongrun"),
            service_dir=str(self.root / "wrongservice"),
        )
        self.addCleanup(wrong.close)
        with self.assertRaisesRegex(RuntimeError, "unrelated service or directory"):
            wrong.start()
        self.assertEqual(request_json(monitor.url + "server-info")["state_dir"], str(self.service_dir.resolve()))

    def test_shutdown_requires_local_descriptor_token(self):
        monitor = self.new_monitor()
        with self.assertRaises(HTTPError) as context:
            request_json(monitor.url + "shutdown", {"token": "incorrect"})
        self.assertEqual(context.exception.code, 403)
        self.assertEqual(request_json(monitor.url + "server-info")["protocol"], PROTOCOL)

    def test_final_result_remains_http_accessible_after_producer_process_exits(self):
        source = self.root / "producer.py"
        source.write_text(
            "import json,sys\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "from visualization.web_training_monitor import WebTrainingMonitor\n"
            "m=WebTrainingMonitor(port=0,width=320,output_dir=sys.argv[1],service_dir=sys.argv[2])\n"
            "m.start()\n"
            "m.submit({0:np.zeros((12,12,3),dtype=np.uint8)},{'run_id':'exited','update':3})\n"
            "m.finish('completed',result={'completed_update':3})\n"
            "m.close()\n"
            "print(json.dumps({'url':m.url}))\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        completed = subprocess.run(
            [sys.executable, "-B", str(source), str(self.root / "exited"), str(self.service_dir)],
            capture_output=True, text=True, timeout=15, env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        url = json.loads(completed.stdout.strip())["url"]
        status = request_json(url + "status.json")
        self.assertEqual(status["phase"], "completed")
        self.assertEqual(status["connection_state"], "finished")
        self.assertFalse(status["producer_alive"])
        self.assertEqual(status["result"]["completed_update"], 3)
        with urlopen(url + "latest.jpg", timeout=1) as response:
            self.assertTrue(response.read().startswith(b"\xff\xd8"))

    def test_crashed_producer_is_stale_and_dead_owner_can_be_reclaimed(self):
        self.service_dir.mkdir()
        directory = SnapshotDirectory(self.service_dir)
        atomic_json(self.service_dir / "active.json", {
            "producer_pid": 99999999, "producer_token": "dead",
            "snapshot_path": str(self.root / "missing.json"), "claimed_at": time.time() - 100,
        })
        old = snapshot_status(directory.snapshot())
        self.assertEqual(old["connection_state"], "stale")
        claimed = directory.claim({
            "producer_pid": os.getpid(), "producer_token": "replacement",
            "output_dir": str(self.root / "replacement"),
        })
        self.assertEqual(claimed["producer_token"], "replacement")

    def test_stale_heartbeat_does_not_change_optimizing_or_completed_phase(self):
        status = {"phase": "optimizing", "state": "running", "heartbeat_at": time.time() - 20,
                  "producer_pid": os.getpid()}
        result = snapshot_status({"status": status})
        self.assertEqual(result["connection_state"], "stale")
        self.assertEqual(result["phase"], "optimizing")
        status["phase"] = "completed"
        self.assertEqual(snapshot_status({"status": status})["connection_state"], "finished")

    def test_missing_or_partial_snapshot_does_not_crash_page_reader(self):
        self.service_dir.mkdir()
        directory = SnapshotDirectory(self.service_dir)
        target = self.root / "snapshot.json"
        atomic_json(self.service_dir / "active.json", {
            "producer_pid": os.getpid(), "producer_token": "test",
            "snapshot_path": str(target), "claimed_at": time.time(),
        })
        target.write_text('{"status":', encoding="utf-8")
        self.assertEqual(directory.snapshot()["status"]["phase"], "starting")
        atomic_json(target, {"producer_token": "old", "status": {"phase": "completed"}})
        self.assertEqual(directory.snapshot()["status"]["phase"], "starting")

    def test_standalone_http_module_does_not_import_training_dependencies(self):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import visualization.monitor_http_server; "
             "assert not ({'torch','carla','numpy','PIL'} & set(sys.modules))"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
