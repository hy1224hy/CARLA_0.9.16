"""Persistent monitor HTTP service; deliberately uses only the standard library.

Run independently with ``python -m visualization.monitor_http_server``.  Training
publishes atomic snapshots, so this process never imports CARLA, NumPy or Torch.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hmac
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen


PROTOCOL = "carla-training-monitor-v2"
TERMINAL_PHASES = frozenset({"completed", "interrupted", "error"})
HEARTBEAT_TIMEOUT = 5.0


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    """Replace one complete document; concurrent readers never see partial JSON."""
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def process_alive(pid: int) -> bool:
    """Check a producer without sending a signal (os.kill is unsafe on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied still implies an owner.
        try:
            code = ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def service_start_lock(state_dir: Path, timeout: float = 8.0):
    """Serialize competing producers starting the same persistent service."""
    path = state_dir / "startup.lock"
    token = secrets.token_hex(16)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "token": token}, stream)
            break
        except FileExistsError:
            owner = read_json(path)
            try:
                old_enough = time.time() - path.stat().st_mtime > 5
            except FileNotFoundError:
                continue
            if (owner and not process_alive(int(owner.get("pid", 0)))) or (not owner and old_enough):
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError("another producer is initializing this monitor service")
            time.sleep(0.05)
    try:
        yield
    finally:
        if read_json(path).get("token") == token:
            path.unlink(missing_ok=True)


def snapshot_status(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Add browser freshness information without changing the training outcome."""
    status = dict(snapshot.get("status", {}))
    now = time.time()
    age = max(0.0, now - float(status.get("heartbeat_at", now)))
    phase = status.get("phase", "starting")
    alive = process_alive(int(status.get("producer_pid", 0)))
    status["heartbeat_age_seconds"] = round(age, 2)
    status["producer_alive"] = alive
    status["connection_state"] = (
        "finished" if phase in TERMINAL_PHASES else
        "live" if alive and age <= HEARTBEAT_TIMEOUT else "stale"
    )
    encoded_at = status.get("encoded_at")
    status["frame_age_seconds"] = (
        round(max(0.0, now - float(encoded_at)), 2) if encoded_at else None
    )
    status["frame_is_previous_episode"] = (
        status.get("frame_scene_generation") is not None and
        status.get("frame_scene_generation") != status.get("scene_generation")
    )
    return status


def request_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 2.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        error.close()
        raise
    with response:
        return json.load(response)


class SnapshotDirectory:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self._lock = threading.RLock()

    def snapshot(self) -> Dict[str, Any]:
        active = read_json(self.state_dir / "active.json")
        snapshot = read_json(Path(active["snapshot_path"])) if active.get("snapshot_path") else {}
        if snapshot.get("producer_token") != active.get("producer_token"):
            snapshot = {}
        if not snapshot:
            snapshot = {"status": {
                "state": "starting", "phase": "starting", "sequence": 0,
                "producer_pid": active.get("producer_pid", 0),
                "producer_token": active.get("producer_token", ""),
                "run_id": active.get("run_id", ""),
                "heartbeat_at": active.get("claimed_at", time.time()),
                "message": "等待训练发布监控快照",
            }}
        return snapshot

    def claim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            active = read_json(self.state_dir / "active.json")
            if active and active.get("producer_token") != payload.get("producer_token"):
                phase = self.snapshot().get("status", {}).get("phase")
                if process_alive(int(active.get("producer_pid", 0))) and phase not in TERMINAL_PHASES:
                    raise RuntimeError("another training producer is still active on this monitor")
            output_dir = Path(payload["output_dir"]).resolve()
            owner = int(payload["producer_pid"])
            if not process_alive(owner):
                raise ValueError("monitor producer must be a live process")
            active = {
                "protocol": PROTOCOL,
                "producer_pid": owner,
                "producer_token": str(payload["producer_token"]),
                "snapshot_path": str(output_dir / "snapshot.json"),
                "output_dir": str(output_dir),
                "run_id": str(payload.get("run_id", "")),
                "claimed_at": time.time(),
            }
            atomic_json(self.state_dir / "active.json", active)
            return active


def make_handler(
    snapshot_reader: Callable[[], Dict[str, Any]],
    server_info: Dict[str, Any],
    *, token: str = "", claim: Optional[Callable] = None,
    shutdown: Optional[Callable[[], None]] = None,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CarlaTrainingMonitor/2.0"

        def log_message(self, _format, *_args):
            return

        def _send(self, body: bytes, content_type: str, code=HTTPStatus.OK):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _json(self, payload, code=HTTPStatus.OK):
            self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json", code)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._send(b"ok\n", "text/plain")
            elif path == "/server-info":
                self._json(server_info)
            elif path == "/status.json":
                status = snapshot_status(snapshot_reader())
                client = parse_qs(urlsplit(self.path).query).get("client", [""])[0]
                status["clients"] = self.server.browser_clients(client)
                self._json(status)
            elif path == "/latest.jpg":
                jpeg = self._jpeg(snapshot_reader())
                if jpeg:
                    self._send(jpeg, "image/jpeg")
                else:
                    self._send(b"Waiting for camera frame", "text/plain", HTTPStatus.NOT_FOUND)
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not token or not hmac.compare_digest(str(payload.get("token", "")), token):
                    self._json({"error": "unauthorized"}, HTTPStatus.FORBIDDEN)
                    return
                if self.path == "/claim" and claim is not None:
                    self._json(claim(payload))
                elif self.path == "/shutdown" and shutdown is not None:
                    self._json({"stopping": True})
                    threading.Thread(target=shutdown, daemon=True).start()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except RuntimeError as error:
                self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (KeyError, ValueError, TypeError, OSError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        @staticmethod
        def _jpeg(snapshot):
            try:
                return base64.b64decode(snapshot.get("jpeg_base64", ""), validate=True)
            except (ValueError, TypeError):
                return b""

        def _stream(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_key = None
            try:
                while not self.server.stopping.is_set():
                    snapshot = snapshot_reader()
                    status = snapshot.get("status", {})
                    key = (snapshot.get("producer_token"), status.get("sequence"))
                    jpeg = self._jpeg(snapshot)
                    if jpeg and key != last_key:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        last_key = key
                    if status.get("phase") in TERMINAL_PHASES:
                        break
                    fps = float(status.get("monitor_fps", 5.0))
                    self.server.stopping.wait(max(0.01, min(0.2, 0.5 / max(0.1, fps))))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


class MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler):
        self.stopping = threading.Event()
        self._client_lock = threading.Lock()
        self._browser_clients = {}
        super().__init__(address, handler)

    def browser_clients(self, client: str = "") -> int:
        with self._client_lock:
            now = time.monotonic()
            self._browser_clients = {
                key: last_seen for key, last_seen in self._browser_clients.items()
                if now - last_seen < 10.0
            }
            if client and len(client) <= 100:
                self._browser_clients[client] = now
            return len(self._browser_clients)

    def shutdown(self):
        self.stopping.set()
        super().shutdown()


def validated_server_info(host: str, port: int, state_dir: Path):
    info = request_json(f"http://{host}:{port}/server-info", timeout=0.5)
    if info.get("protocol") != PROTOCOL or Path(info.get("state_dir", "")).resolve() != state_dir.resolve():
        raise RuntimeError(f"monitor port {port} belongs to an unrelated service or directory")
    return info


def stop_service(state_dir: Path) -> bool:
    record = read_json(state_dir / "server.json")
    if not record:
        return False
    host, port = record["host"], int(record["port"])
    if host in {"0.0.0.0", ""}:
        host = "127.0.0.1"
    try:
        validated_server_info(host, port, state_dir)
        request_json(f"http://{host}:{port}/shutdown", {"token": record["token"]})
        return True
    except URLError:
        return False


def serve(state_dir: Path, host: str, port: int, startup_token: str = "") -> None:
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    directory = SnapshotDirectory(state_dir)
    secret = secrets.token_urlsafe(32)
    info = {"protocol": PROTOCOL, "state_dir": str(state_dir), "server_pid": os.getpid()}
    server = MonitorHTTPServer((host, port), make_handler(directory.snapshot, info, token=secret, claim=directory.claim))
    server.RequestHandlerClass = make_handler(
        directory.snapshot, info, token=secret, claim=directory.claim, shutdown=server.shutdown
    )
    info.update(host=host, port=server.server_port)
    record = dict(info, token=secret, startup_token=startup_token)
    atomic_json(state_dir / "server.json", record)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.stopping.set()
        server.server_close()
        # Do not remove a descriptor written by a replacement server.
        if read_json(state_dir / "server.json").get("token") == secret:
            (state_dir / "server.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--startup-token", default="", help=argparse.SUPPRESS)
    parser.add_argument("--stop", action="store_true", help="Stop only the matching monitor service")
    args = parser.parse_args()
    state_dir = Path(args.state_dir).resolve()
    if args.stop:
        print("Monitor service stopped" if stop_service(state_dir) else "Monitor service is not running")
    else:
        serve(state_dir, args.host, args.port, args.startup_token)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA 俯视训练监控</title><style>
body{margin:0;background:#030712;color:#e5e7eb;font:14px system-ui}main{max-width:1200px;margin:auto;padding:20px}
h1{font-size:24px}.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.badge{padding:7px 12px;background:#1e3a8a;border-radius:7px}
.meta{margin:12px 0;line-height:1.7;color:#cbd5e1}img{width:100%;min-height:120px;background:#111827;border-radius:8px}
.note{padding:10px 12px;background:#172554;border-radius:6px;margin:12px 0}.warning{background:#78350f}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;padding:12px;border-radius:8px}details{margin-top:14px}
</style></head><body><main><h1>CARLA 训练监控</h1><div class="bar"><strong class="badge" id="phase">正在连接</strong><span id="run"></span></div>
<div class="meta" id="progress">等待训练状态</div><div class="note" id="note">等待第一张摄像头画面</div>
<img id="camera" alt="等待 CARLA 摄像头画面"><div class="meta" id="frame"></div>
<section id="result-section" hidden><h2>最终结果</h2><pre id="result"></pre></section>
<details><summary>完整监控状态</summary><pre id="status"></pre></details></main><script>
const labels={starting:'启动中',resetting:'切换场景',rollout:'采样行驶中',running:'采样行驶中',optimizing:'正在更新模型',evaluating:'独立评估中',validating:'教师门控评估中',saving:'正在保存 checkpoint',finalizing:'正在整理结果',completed:'训练已完成',interrupted:'训练已中断',error:'训练异常退出'};
let imageKey='';let failures=0;const clientId=Math.random().toString(36).slice(2)+Date.now().toString(36);
document.querySelector('#camera').onerror=()=>{imageKey=''};
async function poll(){try{let r=await fetch('/status.json?client='+encodeURIComponent(clientId),{cache:'no-store'});if(!r.ok)throw Error(r.status);let s=await r.json();failures=0;
document.querySelector('#phase').textContent=(s.phase==='completed'&&s.purpose==='evaluation')?'评估已结束':(labels[s.phase]||s.phase||'启动中');document.querySelector('#run').textContent=s.run_id||'';
let progress=s.monitor_kind==='mut_mappo'?'轮次 '+(s.round??'—')+' / 蓝团队 '+(s.team??'—')+' / 条件 '+(s.condition??'—')+' / 红方源 u'+(s.red_source_update??'—'):'Update '+(s.update??'—');
document.querySelector('#progress').textContent=progress+' / Episode '+(s.episode??'—')+' / Step '+(s.step??'—')+' / 场景 '+(s.scenario??'—');
let messages=[];if(s.message)messages.push(s.message);if(s.connection_state==='stale')messages.push('训练进程心跳失联，当前保留最后已知画面与状态');
if(s.frame_is_previous_episode)messages.push('当前仍为上一回合画面，等待新场景第一帧');else if(!s.sequence)messages.push('等待摄像头第一帧');
if(s.phase==='optimizing'||s.phase==='saving'||s.phase==='finalizing')messages.push('此阶段仿真暂停，保留最后一帧');
if(s.monitor_error)messages.push('监控错误：'+s.monitor_error);if(s.connection_state==='finished')messages.push('结果已保留；新训练开始后页面会自动更新');
let note=document.querySelector('#note');note.textContent=messages.join(' · ')||'俯视摄像头持续跟随训练车辆';note.className='note'+(s.connection_state==='stale'||s.monitor_error?' warning':'');
let session=s.session_id||s.producer_token||'default';let finished=['completed','interrupted','error'].includes(s.phase);let key=session+(finished?':final':':stream');if(s.sequence&&key!==imageKey){document.querySelector('#camera').src=(finished?'/latest.jpg?session=':'/stream.mjpg?session=')+encodeURIComponent(session);imageKey=key;}
let frameProgress=s.frame_monitor_kind==='mut_mappo'?'轮次 '+(s.frame_round??'—')+' / 蓝团队 '+(s.frame_team??'—')+' / 条件 '+(s.frame_condition??'—'):'Update '+(s.frame_update??'—');
document.querySelector('#frame').textContent=s.sequence?'画面所属：'+frameProgress+' / Episode '+(s.frame_episode??'—')+' / Step '+(s.frame_step??'—')+' / '+(s.frame_scenario??'—')+' · 画面距今 '+(s.frame_age_seconds??0)+' 秒':'';
document.querySelector('#result-section').hidden=!s.result;document.querySelector('#result').textContent=JSON.stringify(s.result||{},null,2);document.querySelector('#status').textContent=JSON.stringify(s,null,2);
}catch(e){failures++;imageKey='';let note=document.querySelector('#note');note.className='note warning';note.textContent='监控服务连接中断，保留最后画面；正在自动重连（'+failures+'）';}setTimeout(poll,1000)}poll();
</script></body></html>"""


if __name__ == "__main__":
    main()
