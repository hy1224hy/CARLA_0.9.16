"""Detached single-owner supervisor and read-only training status entrypoint."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224"
RUN = ROOT / "logs" / RUN_ID
OPS = RUN / "continuation_u600"
CONTRACT = "configs/base_mappo/p0_base_town05_r8c1_pace_v1.json"
EXTENSION = "configs/base_mappo/p0_r8c1_continue_u600_v1.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5  # Access denied is not evidence of exit.
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def command(stop=600, interrupt=0):
    result = [sys.executable, "-u", "-m", "training.train_no_rllib_mappo_robust_v2",
              "--experiment-contract", CONTRACT, "--continuation-contract", EXTENSION,
              "--updates", str(stop), "--run-id", RUN_ID, "--save-dir", str(RUN),
              "--resume", "latest", "--host", "127.0.0.1", "--port", "2000",
              "--follow-spectator", "--stop-request-file", str(OPS / "STOP_REQUESTED")]
    if interrupt:
        result += ["--p0-smoke-interrupt-after-episodes", str(interrupt)]
    return result


def follow(done, session):
    import carla
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(3)
    while not done.wait(0.25):
        try:
            world = client.get_world()
            vehicles = list(world.get_actors().filter("vehicle.*"))
            if not vehicles:
                continue
            locations = [v.get_location() for v in vehicles]
            x = sum(p.x for p in locations) / len(locations)
            y = sum(p.y for p in locations) / len(locations)
            span = max(math.hypot(p.x-x, p.y-y) for p in locations)
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=x, y=y, z=max(p.z for p in locations)+max(20, span*1.8+12)),
                carla.Rotation(pitch=-90, yaw=0)))
            session["camera"] = {"status": "FOLLOWING", "vehicles": len(vehicles), "at": now()}
        except Exception as error:
            session["camera"] = {"status": "WAITING", "error": str(error), "at": now()}


def run(args):
    import msvcrt
    OPS.mkdir(parents=True, exist_ok=True)
    # The OS releases this lock even if the supervisor is abruptly terminated.
    with (OPS / "session.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        previous_path = OPS / "session.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous.get("status") != "EXITED" and previous.get("trainer_pid") and pid_alive(previous["trainer_pid"]):
                raise RuntimeError("previous trainer still alive; refusing duplicate resume")
        if (OPS / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED exists; archive it before explicit resume")
        import carla
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10)
        world = client.get_world()
        if len(world.get_actors().filter("vehicle.*")) or len(world.get_actors().filter("sensor.*")):
            raise RuntimeError("CARLA has existing vehicles/sensors; refusing a concurrent training")
        if world.get_settings().synchronous_mode:
            raise RuntimeError("CARLA is already synchronous; verify previous owner exited")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = OPS / f"segment_{stamp}.log"
        cmd = command(args.stop_update, args.interrupt_after_episodes)
        session = {"status": "STARTING", "started_at": now(), "supervisor_pid": os.getpid(),
                   "run_id": RUN_ID, "command": cmd, "log_path": str(log_path),
                   "stop_update": args.stop_update, "interrupt_after_episodes": args.interrupt_after_episodes}
        done = threading.Event()
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
            session.update(status="RUNNING", trainer_pid=child.pid)
            thread = threading.Thread(target=follow, args=(done, session), daemon=True)
            thread.start()
            try:
                while child.poll() is None:
                    session["checked_at"] = now()
                    write_json(OPS / "session.json", session)
                    time.sleep(2)
            finally:
                done.set()
                thread.join(timeout=4)
            session.update(status="EXITED", exit_code=child.returncode, ended_at=now())
            write_json(OPS / "session.json", session)
            write_json(OPS / f"segment_{stamp}.json", session)
            return child.returncode


def status():
    manifest = json.loads((RUN / "checkpoint_manifest.json").read_text())
    boundary = int(manifest["latest"]["update"])
    rows = []
    for line in (RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["update"] <= boundary:
            rows.append(row)
    window = rows[-20:]
    s = rows[-1]["scalars"]
    relevant = {k:v for k,v in s.items() if k.startswith("curriculum/") or
                any(t in k for t in ("explained_variance", "preupdate_max_abs_ratio_error", "max_abs_step_residual",
                    "update_skipped", "approx_kl", "post_tanh_entropy", "entropy_coefficient", "learning_rate"))}
    issues = []
    for row in window:
        for k,v in row["scalars"].items():
            if isinstance(v, (int,float)) and not math.isfinite(v):
                issues.append(f"u{row['update']} nonfinite {k}")
            if ("preupdate_max_abs_ratio_error" in k or "max_abs_step_residual" in k) and abs(v)>1e-6:
                issues.append(f"u{row['update']} {k}={v}")
            if k == "train/update_skipped" and v:
                issues.append(f"u{row['update']} skipped")
    snapshot = {"at":now(), "safe_update":boundary, "metric_update": rows[-1]["update"],
                "last_20_update_range":[window[0]["update"],window[-1]["update"]],
                "latest_metrics":relevant, "numeric_issues":issues,
                "session":json.loads((OPS/"session.json").read_text()) if (OPS/"session.json").exists() else None}
    if snapshot["session"]:
        snapshot["trainer_alive"] = pid_alive(snapshot["session"].get("trainer_pid", 0))
    write_json(OPS / "monitoring" / "latest.json", snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "status", "stop"])
    parser.add_argument("--stop-update", type=int, default=600)
    parser.add_argument("--interrupt-after-episodes", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "run":
        raise SystemExit(run(args))
    if args.mode == "status":
        status()
    else:
        OPS.mkdir(parents=True, exist_ok=True)
        (OPS / "STOP_REQUESTED").write_text(now(), encoding="utf-8")
        print("Stop requested at next complete update; do not kill CARLA.")
