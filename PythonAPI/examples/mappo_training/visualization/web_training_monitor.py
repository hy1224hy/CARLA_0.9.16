"""Non-blocking browser monitor for headless CARLA training.

Image submission only copies references into a size-one queue. Composition,
JPEG encoding, HTTP clients, and recording run in background threads. The
optional overhead camera is positioned before the training tick and never
waits for an image, though the extra sensor does add rendering/RPC work.
"""

from __future__ import annotations

import io
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MonitorFrame:
    images: Mapping[int, np.ndarray]
    status: Mapping[str, Any]
    submitted_at: float


class _LatestFrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._sequence = 0
        self._closed = False
        self._status: Dict[str, Any] = {
            "state": "starting",
            "sequence": 0,
            "clients": 0,
        }

    def publish(self, jpeg: bytes, status: Mapping[str, Any]) -> None:
        with self._condition:
            self._sequence += 1
            clients = int(self._status.get("clients", 0))
            self._jpeg = jpeg
            self._status = dict(status)
            self._status.update(
                state="running",
                sequence=self._sequence,
                encoded_at=time.time(),
                clients=clients,
            )
            self._condition.notify_all()

    def wait_after(self, sequence: int, timeout: float = 2.0):
        with self._condition:
            if self._sequence <= sequence:
                self._condition.wait(timeout)
            return self._sequence, self._jpeg, self._closed

    def status(self) -> Dict[str, Any]:
        with self._condition:
            return dict(self._status)

    def change_clients(self, delta: int) -> None:
        with self._condition:
            self._status["clients"] = max(
                0, int(self._status.get("clients", 0)) + delta
            )

    def set_error(self, error: Exception, state: str = "encode_error") -> None:
        with self._condition:
            self._status.update(
                state=state,
                error=f"{type(error).__name__}: {error}",
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._status["state"] = "stopped"
            self._condition.notify_all()


class WebTrainingMonitor:
    """Publish a latest-frame MJPEG stream without blocking training."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        fps: float = 5.0,
        width: int = 960,
        jpeg_quality: int = 65,
        record_dir: str = "",
        view: str = "vehicle",
        overhead_height: float = 35.0,
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

        self.host = host
        self.port = port
        self.fps = float(fps)
        self.width = int(width)
        self.jpeg_quality = int(jpeg_quality)
        self.record_dir = Path(record_dir).resolve() if record_dir else None
        self.view = view
        self._camera = None
        self._camera_failed = False
        if view == "overhead":
            from visualization.overhead_training_camera import OverheadTrainingCamera

            self._camera = OverheadTrainingCamera(
                width=width, fps=fps, height=overhead_height
            )
        self._pending: queue.Queue[MonitorFrame] = queue.Queue(maxsize=1)
        self._store = _LatestFrameStore()
        self._stop = threading.Event()
        self._encoder_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._last_submit = 0.0
        self._record_index = 0
        self._dropped = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("a closed monitor cannot be restarted")
        if self._encoder_thread is not None:
            return
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer(
            (self.host, self.port), self._handler_class()
        )
        self.port = int(self._server.server_port)
        self._server.daemon_threads = True
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="carla-monitor-encoder", daemon=True
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="carla-monitor-http",
            daemon=True,
        )
        self._encoder_thread.start()
        self._server_thread.start()

    def submit(
        self,
        images: Mapping[int, Optional[np.ndarray]],
        status: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Submit immediately; return False when throttled or no image exists."""
        now = time.monotonic()
        if now - self._last_submit < 1.0 / self.fps:
            return False
        metadata = dict(status or {})
        metadata.update(view=self.view, agents=sorted(int(aid) for aid in images))
        if self._camera is not None:
            overhead, camera_frame = self._camera.latest
            images = {-1: overhead}
            metadata["camera_frame"] = camera_frame
        available = {
            int(agent_id): image
            for agent_id, image in images.items()
            if image is not None
        }
        if not available:
            return False
        frame = MonitorFrame(available, metadata, time.time())
        try:
            self._pending.put_nowait(frame)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            self._pending.put_nowait(frame)
        self._last_submit = now
        return True

    def prepare_camera(self, env) -> None:
        """Called before env.step; sensor failures must not interrupt training."""
        if self._camera is None or self._camera_failed:
            return
        try:
            self._camera.update(env)
        except Exception as error:
            self._camera_failed = True
            self._store.set_error(error, state="camera_error")
            print(f"[monitor] Overhead camera unavailable; training continues: {error}")
            try:
                self._camera.close()
            except Exception:
                pass

    def reset_camera(self) -> None:
        """Called before env.reset and shutdown; no extra world ticks."""
        self._camera_failed = False
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as error:
                self._camera_failed = True
                self._store.set_error(error, state="camera_error")
                print(f"[monitor] Overhead camera cleanup failed: {error}")

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.reset_camera()
        self._store.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=3.0)
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)

    def _encode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                canvas = self._compose(frame.images, frame.status)
                output = io.BytesIO()
                canvas.save(
                    output,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=False,
                )
                jpeg = output.getvalue()
                status = dict(frame.status)
                status.update(
                    submitted_at=frame.submitted_at,
                    dropped_frames=self._dropped,
                )
                self._store.publish(jpeg, status)
                if self.record_dir:
                    self._record_index += 1
                    destination = self.record_dir / f"frame_{self._record_index:08d}.jpg"
                    destination.write_bytes(jpeg)
            except Exception as error:
                self._store.set_error(error)

    def _compose(
        self, images: Mapping[int, np.ndarray], status: Mapping[str, Any]
    ) -> Image.Image:
        panels = []
        for agent_id in sorted(images):
            array = np.asarray(images[agent_id])
            if array.ndim != 3 or array.shape[2] < 3:
                continue
            # CARLA raw camera buffers are BGRA.  The environment drops alpha
            # but keeps BGR byte order; Pillow expects RGB.
            rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
            panel = Image.fromarray(rgb, mode="RGB")
            panels.append((agent_id, panel))
        if not panels:
            raise ValueError("no valid camera frames")

        columns = min(2, len(panels))
        panel_width = self.width // columns
        source_width, source_height = panels[0][1].size
        panel_height = max(1, round(panel_width * source_height / source_width))
        rows = (len(panels) + columns - 1) // columns
        header_height = 44
        canvas = Image.new(
            "RGB", (self.width, header_height + rows * panel_height), "#111827"
        )
        draw = ImageDraw.Draw(canvas)
        summary = "  ".join(
            f"{key}={value}"
            for key, value in status.items()
            if key in {"run_id", "update", "episode", "step", "scenario"}
        )
        draw.text((12, 8), summary or "CARLA training monitor", fill="white")
        for index, (agent_id, panel) in enumerate(panels):
            x = (index % columns) * panel_width
            y = header_height + (index // columns) * panel_height
            resized = panel.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
            canvas.paste(resized, (x, y))
            label = "Overhead" if agent_id == -1 else f"Agent {agent_id}"
            draw.rectangle((x + 6, y + 6, x + 92, y + 28), fill="#111827")
            draw.text((x + 12, y + 10), label, fill="white")
        return canvas

    def _handler_class(self):
        store = self._store

        class Handler(BaseHTTPRequestHandler):
            server_version = "CarlaTrainingMonitor/1.0"

            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/status.json":
                    body = json.dumps(store.status(), ensure_ascii=False).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/healthz":
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(b"ok\n")
                elif path == "/stream.mjpg":
                    self._stream()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def _stream(self):
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = 0
                store.change_clients(1)
                try:
                    while True:
                        sequence, jpeg, closed = store.wait_after(sequence)
                        if closed:
                            break
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    store.change_clients(-1)

        return Handler


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA Training Monitor</title>
<style>body{margin:0;background:#030712;color:#e5e7eb;font:14px system-ui}
main{max-width:1200px;margin:auto;padding:16px}img{width:100%;background:#111827}
pre{white-space:pre-wrap;background:#111827;padding:12px;border-radius:8px}</style>
</head><body><main><h2>CARLA Training Monitor</h2>
<img src="/stream.mjpg" alt="等待 CARLA 摄像头画面">
<pre id="status">正在连接状态接口…</pre></main>
<script>async function poll(){try{let r=await fetch('/status.json',{cache:'no-store'});
document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}
setTimeout(poll,1000)}poll()</script></body></html>"""
