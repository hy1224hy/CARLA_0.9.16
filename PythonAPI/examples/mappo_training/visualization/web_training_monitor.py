"""Non-blocking publisher with a persistent, independent HTTP monitor."""
from __future__ import annotations

import base64
import html
import io
import json
import math
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image, ImageDraw

from visualization.monitor_http_server import (
    INDEX_HTML, PROTOCOL, TERMINAL_PHASES, MonitorHTTPServer, atomic_json,
    make_handler, read_json, request_json, service_start_lock, validated_server_info,
)


@dataclass(frozen=True)
class MonitorFrame:
    images: Mapping[int, np.ndarray]
    status: Mapping[str, Any]
    submitted_at: float
    generation: int = 0


class _LatestFrameStore:
    """Training status and image provenance are independent channels."""

    def __init__(self, session_id: str = "") -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._sequence = 0
        self._closed = False
        self._terminal = False
        self._status: Dict[str, Any] = {
            "state": "starting", "phase": "starting", "sequence": 0,
            "clients": 0, "scene_generation": 0, "producer_pid": os.getpid(),
            "producer_token": session_id, "session_id": session_id,
            "heartbeat_at": time.time(),
        }

    def update_status(self, status: Mapping[str, Any]) -> bool:
        with self._condition:
            if self._terminal:
                return False
            self._status.update(status)
            self._status["status_updated_at"] = time.time()
            self._condition.notify_all()
            return True

    def begin_episode(self, status: Mapping[str, Any]) -> int:
        with self._condition:
            generation = int(self._status["scene_generation"]) + 1
            if not self._terminal:
                self._status.update(status)
                self._status.update(scene_generation=generation, phase="resetting", state="running")
                self._status["status_updated_at"] = time.time()
                self._condition.notify_all()
            return generation

    def publish(self, jpeg: bytes, frame: MonitorFrame, dropped: int) -> bool:
        with self._condition:
            if self._terminal or frame.generation != self._status["scene_generation"]:
                return False
            self._sequence += 1
            self._jpeg = jpeg
            self._status.update(
                sequence=self._sequence, encoded_at=time.time(),
                frame_submitted_at=frame.submitted_at,
                frame_scene_generation=frame.generation, dropped_frames=dropped,
            )
            for key in ("run_id", "update", "episode", "step", "scenario"):
                self._status["frame_" + key] = frame.status.get(key)
            self._status["camera_frame"] = frame.status.get("camera_frame")
            self._condition.notify_all()
            return True

    def snapshot(self, *, heartbeat: bool = False) -> Dict[str, Any]:
        with self._condition:
            if heartbeat and not self._closed:
                self._status["heartbeat_at"] = time.time()
            status = dict(self._status)
            status["frame_current"] = (
                self._jpeg is not None and
                status.get("frame_scene_generation") == status.get("scene_generation")
            )
            return {
                "protocol": PROTOCOL, "producer_token": status["producer_token"],
                "status": status,
                "jpeg_base64": base64.b64encode(self._jpeg).decode("ascii") if self._jpeg else "",
            }

    def status(self) -> Dict[str, Any]:
        with self._condition:
            status = dict(self._status)
            status["frame_current"] = (
                self._jpeg is not None and
                status.get("frame_scene_generation") == status.get("scene_generation")
            )
            return status

    def set_error(self, error: Exception, state: str = "encode_error") -> None:
        with self._condition:
            # Recording failures are still reportable after the outcome freezes.
            self._status.update(monitor_state=state, monitor_error=f"{type(error).__name__}: {error}")
            self._condition.notify_all()

    def finish(self, phase: str, result: Mapping[str, Any], message: str) -> None:
        with self._condition:
            if self._terminal:
                return
            self._terminal = True
            self._status.update(
                state=phase, phase=phase, result=dict(result), message=message,
                finished_at=time.time(), heartbeat_at=time.time(),
            )
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class WebTrainingMonitor:
    """Publish without waiting for HTTP clients.

    With output_dir and service_dir, a stdlib HTTP subprocess survives close().
    Without service_dir, this instance owns an in-process HTTP server.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8080, fps: float = 5.0,
        width: int = 960, jpeg_quality: int = 65, record_dir: str = "",
        view: str = "vehicle", overhead_height: float = 35.0,
        smoothing_seconds: float = 0.5, output_dir: str = "", service_dir: str = "",
    ) -> None:
        if not 0 <= port < 65536:
            raise ValueError("monitor port must be in [0, 65535]")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("monitor fps must be positive")
        if width < 160:
            raise ValueError("monitor width must be at least 160 pixels")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("monitor JPEG quality must be in [1, 95]")
        if view not in {"vehicle", "overhead"}:
            raise ValueError("monitor view must be vehicle or overhead")
        if not math.isfinite(smoothing_seconds) or smoothing_seconds < 0:
            raise ValueError("monitor smoothing seconds must be finite and nonnegative")
        if service_dir and not output_dir:
            raise ValueError("a persistent monitor service requires output_dir")

        self.host, self.port = host, port
        self.fps, self.width, self.jpeg_quality = float(fps), int(width), int(jpeg_quality)
        self.record_dir = Path(record_dir).resolve() if record_dir else None
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.service_dir = Path(service_dir).resolve() if service_dir else None
        self.view = view
        self.session_id = secrets.token_hex(16)
        self._camera = None
        self._camera_failed = False
        if view == "overhead":
            from visualization.overhead_training_camera import OverheadTrainingCamera
            self._camera = OverheadTrainingCamera(
                width=width, fps=fps, height=overhead_height,
                smoothing_seconds=smoothing_seconds,
            )
        self._pending: queue.Queue[MonitorFrame] = queue.Queue(maxsize=1)
        self._store = _LatestFrameStore(self.session_id)
        self._store.update_status({"view": view, "monitor_fps": self.fps})
        self._stop = threading.Event()
        self._changed = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._work_lock = threading.Lock()
        self._inflight = False
        self._archive_done = threading.Event()
        self._archive_requested = False
        self._finishing = False
        self._encoder_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[MonitorHTTPServer] = None
        self._last_submit = 0.0
        self._last_frame_key = None
        self._record_index = 0
        self._dropped = 0
        self._last_persist = 0.0
        self._server_process = None

    @property
    def finished(self) -> bool:
        return self._store.status()["phase"] in TERMINAL_PHASES

    @property
    def url(self) -> str:
        client_host = "127.0.0.1" if self.host in {"0.0.0.0", ""} else self.host
        return f"http://{client_host}:{self.port}/"

    def _start_persistent_server(self) -> None:
        self.service_dir.mkdir(parents=True, exist_ok=True)
        with service_start_lock(self.service_dir):
            self._start_persistent_server_locked()

    def _start_persistent_server_locked(self) -> None:
        record = read_json(self.service_dir / "server.json")
        candidate_port = self.port or int(record.get("port", 0))
        info = None
        client_host = "127.0.0.1" if self.host in {"0.0.0.0", ""} else self.host
        if candidate_port:
            try:
                info = validated_server_info(client_host, candidate_port, self.service_dir)
            except HTTPError as error:
                raise RuntimeError(f"monitor port {candidate_port} belongs to an unrelated HTTP service") from error
            except (URLError, TimeoutError, ConnectionError):
                info = None
        if info is not None:
            if record.get("server_pid") != info.get("server_pid") or not record.get("token"):
                raise RuntimeError("monitor service descriptor does not match the listening server")
            self.port = int(info["port"])
        else:
            startup_token = secrets.token_hex(16)
            command = [
                sys.executable, "-B", "-m", "visualization.monitor_http_server",
                "--state-dir", str(self.service_dir), "--host", self.host,
                "--port", str(self.port), "--startup-token", startup_token,
            ]
            options = {"cwd": str(Path(__file__).resolve().parent.parent),
                       "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                       "stderr": subprocess.DEVNULL}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                options["start_new_session"] = True
            self._server_process = subprocess.Popen(command, **options)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                record = read_json(self.service_dir / "server.json")
                if record.get("startup_token") == startup_token:
                    self.port = int(record["port"])
                    try:
                        validated_server_info(client_host, self.port, self.service_dir)
                        break
                    except (URLError, TimeoutError, ConnectionError):
                        pass
                if self._server_process.poll() is not None:
                    raise RuntimeError(f"persistent monitor service could not bind port {self.port}")
                time.sleep(0.05)
            else:
                self._server_process.terminate()
                self._server_process.wait(timeout=3)
                raise RuntimeError("persistent monitor service did not become ready")
        try:
            request_json(self.url + "claim", {
                "token": record["token"], "producer_pid": os.getpid(),
                "producer_token": self.session_id, "output_dir": str(self.output_dir),
                "run_id": self._store.status().get("run_id", ""),
            })
        except HTTPError as error:
            if error.code == 409:
                raise RuntimeError("another training producer is still active on this monitor") from error
            raise RuntimeError("persistent monitor service refused producer registration") from error

    def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("a closed monitor cannot be restarted")
        if self._encoder_thread is not None:
            return
        for directory in (self.record_dir, self.output_dir):
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
        if self.service_dir:
            self._start_persistent_server()
        else:
            info = {"protocol": PROTOCOL, "state_dir": "", "server_pid": os.getpid()}
            self._server = MonitorHTTPServer(
                (self.host, self.port), make_handler(self._store.snapshot, info)
            )
            self.port = int(self._server.server_port)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, name="carla-monitor-http", daemon=True
            )
            self._server_thread.start()
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="carla-monitor-encoder", daemon=True
        )
        self._changed.set()
        self._encoder_thread.start()

    def set_phase(self, phase: str, **status) -> None:
        if phase in TERMINAL_PHASES:
            raise ValueError("terminal phases must be published with finish()")
        status.update(phase=phase, state="starting" if phase == "starting" else "running")
        if self._store.update_status(status):
            self._changed.set()

    def begin_episode(self, **status) -> None:
        if self._finishing or self._stop.is_set():
            return
        self._store.begin_episode(status)
        with self._work_lock:
            while True:
                try:
                    self._pending.get_nowait()
                    self._pending.task_done()
                except queue.Empty:
                    break
            if not self._inflight:
                self._idle.set()
        self._last_submit = 0.0
        self._last_frame_key = None
        self._camera_failed = False
        if self._camera is not None:
            try:
                self._camera.reset_view()
            except Exception as error:
                self._camera_failed = True
                self._store.set_error(error, "camera_error")
        self._changed.set()

    def submit(self, images: Mapping[int, Optional[np.ndarray]], status: Optional[Mapping[str, Any]] = None) -> bool:
        """Update progress immediately; throttle/drop only the image work."""
        if self._finishing or self._stop.is_set():
            return False
        metadata = dict(status or {})
        metadata.update(view=self.view, agents=sorted(int(aid) for aid in images))
        if self._store.status()["phase"] == "starting":
            metadata.update(phase="rollout", state="running")
        self._store.update_status(metadata)
        self._changed.set()
        now = time.monotonic()
        if now - self._last_submit < 1.0 / self.fps:
            return False
        generation = int(self._store.status()["scene_generation"])
        if self._camera is not None:
            overhead, camera_frame = self._camera.latest
            images = {-1: overhead}
            metadata["camera_frame"] = camera_frame
            frame_key = (generation, camera_frame) if camera_frame is not None else None
        else:
            frame_key = (generation, tuple((int(aid), id(image)) for aid, image in sorted(images.items()) if image is not None))
        available = {int(aid): image for aid, image in images.items() if image is not None}
        if not available or (frame_key is not None and frame_key == self._last_frame_key):
            return False
        frame = MonitorFrame(available, metadata, time.time(), generation)
        with self._work_lock:
            self._idle.clear()
            try:
                self._pending.put_nowait(frame)
            except queue.Full:
                self._pending.get_nowait()
                self._pending.task_done()
                self._dropped += 1
                self._pending.put_nowait(frame)
        self._last_submit = now
        self._last_frame_key = frame_key
        self._changed.set()
        return True

    def prepare_camera(self, env) -> None:
        if self._camera is None or self._camera_failed or self._finishing:
            return
        try:
            self._camera.update(env)
            self._store.update_status({"monitor_state": "ok", "monitor_error": ""})
        except Exception as error:
            self._camera_failed = True
            self._store.set_error(error, "camera_error")
            print(f"[monitor] Overhead camera unavailable; training continues: {error}")
            try:
                self._camera.close()
            except Exception:
                pass
        self._changed.set()

    def reset_camera(self) -> None:
        """Compatibility alias; retain the independent sensor across episodes."""
        self.begin_episode()

    def stop_camera(self) -> None:
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as error:
                self._store.set_error(error, "camera_error")
                self._changed.set()

    def finish(self, phase: str, result: Optional[Mapping[str, Any]] = None, message: str = "") -> None:
        if phase not in TERMINAL_PHASES:
            raise ValueError("final monitor phase must be completed, interrupted or error")
        if self._finishing:
            return
        self._finishing = True
        self.stop_camera()
        if self._encoder_thread is not None:
            self._idle.wait(timeout=3.0)
        self._store.finish(phase, result or {}, message)
        self._archive_requested = True
        self._changed.set()
        if self._encoder_thread is not None and self.output_dir:
            if not self._archive_done.wait(timeout=5.0):
                print("[monitor] Final snapshot could not be confirmed; inspect monitor output directory")

    def close(self) -> None:
        if self._stop.is_set():
            return
        if self.service_dir and not self._finishing:
            self.finish("interrupted", message="Monitor producer closed before a final training result")
        self.stop_camera()
        self._stop.set()
        self._changed.set()
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=3.0)
        self._store.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)

    def _persist(self, snapshot: Dict[str, Any]) -> None:
        if self.output_dir:
            atomic_json(self.output_dir / "snapshot.json", snapshot)
        if self._archive_requested and self.output_dir and not self._archive_done.is_set():
            status = snapshot["status"]
            atomic_json(self.output_dir / "result.json", status)
            jpeg = base64.b64decode(snapshot["jpeg_base64"])
            if jpeg:
                temporary = self.output_dir / "latest.jpg.tmp"
                temporary.write_bytes(jpeg)
                temporary.replace(self.output_dir / "latest.jpg")
            phase = {"completed": "训练已完成", "interrupted": "训练已中断", "error": "训练异常退出"}[status["phase"]]
            encoded = snapshot["jpeg_base64"]
            result_text = html.escape(json.dumps(status, ensure_ascii=False, indent=2))
            page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>CARLA 训练结果</title>"
                    "<style>body{background:#030712;color:#e5e7eb;font:14px system-ui;max-width:1200px;margin:24px auto}"
                    "img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
                    f"<h1>CARLA 训练结果：{phase}</h1>"
                    + (f"<img src='data:image/jpeg;base64,{encoded}' alt='最后的训练画面'>" if encoded else "<p>本次训练未收到摄像头画面。</p>")
                    + f"<pre>{result_text}</pre></html>")
            temporary = self.output_dir / "result.html.tmp"
            temporary.write_text(page, encoding="utf-8")
            temporary.replace(self.output_dir / "result.html")
            self._archive_done.set()

    def _encode_loop(self) -> None:
        while not self._stop.is_set():
            self._changed.wait(timeout=0.1)
            with self._work_lock:
                try:
                    frame = self._pending.get_nowait()
                    self._inflight = True
                except queue.Empty:
                    frame = None
            if frame is not None:
                try:
                    canvas = self._compose(frame.images, frame.status)
                    output = io.BytesIO()
                    canvas.save(output, format="JPEG", quality=self.jpeg_quality, optimize=False)
                    jpeg = output.getvalue()
                    if self._store.publish(jpeg, frame, self._dropped):
                        self._changed.set()
                        if self.record_dir:
                            self._record_index += 1
                            (self.record_dir / f"frame_{self._record_index:08d}.jpg").write_bytes(jpeg)
                except Exception as error:
                    self._store.set_error(error)
                    self._changed.set()
                finally:
                    with self._work_lock:
                        self._pending.task_done()
                        self._inflight = False
                        if self._pending.empty():
                            self._idle.set()
            now = time.monotonic()
            if self._changed.is_set() or now - self._last_persist >= 1.0:
                self._changed.clear()
                try:
                    self._persist(self._store.snapshot(heartbeat=True))
                except Exception as error:
                    self._store.set_error(error, "snapshot_error")
                self._last_persist = now
        try:
            self._persist(self._store.snapshot(heartbeat=True))
        except Exception as error:
            print(f"[monitor] Snapshot persistence failed: {type(error).__name__}: {error}")

    def _compose(self, images: Mapping[int, np.ndarray], status: Mapping[str, Any]) -> Image.Image:
        panels = []
        for aid in sorted(images):
            array = np.asarray(images[aid])
            if array.ndim != 3 or array.shape[2] < 3:
                continue
            rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
            panels.append((aid, Image.fromarray(rgb)))
        if not panels:
            raise ValueError("no valid camera frames")
        columns = min(2, len(panels))
        panel_width = self.width // columns
        source_width, source_height = panels[0][1].size
        panel_height = max(1, round(panel_width * source_height / source_width))
        rows = (len(panels) + columns - 1) // columns
        header_height = 44
        canvas = Image.new("RGB", (self.width, header_height + rows * panel_height), "#111827")
        draw = ImageDraw.Draw(canvas)
        summary = "  ".join(f"{key}={value}" for key, value in status.items()
                            if key in {"run_id", "update", "episode", "step", "scenario"})
        draw.text((12, 8), summary or "CARLA training monitor", fill="white")
        for index, (aid, panel) in enumerate(panels):
            x = (index % columns) * panel_width
            y = header_height + (index // columns) * panel_height
            canvas.paste(panel.resize((panel_width, panel_height), Image.Resampling.BILINEAR), (x, y))
            draw.rectangle((x + 6, y + 6, x + 92, y + 28), fill="#111827")
            draw.text((x + 12, y + 10), "Overhead" if aid == -1 else f"Agent {aid}", fill="white")
        return canvas


_INDEX_HTML = INDEX_HTML
