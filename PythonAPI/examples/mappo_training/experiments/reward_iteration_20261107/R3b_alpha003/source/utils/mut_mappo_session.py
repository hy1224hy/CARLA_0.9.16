"""Cross-branch CARLA ownership, detached launch, and file-only progress inspection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid):
    if os.name == "nt":
        from utils.r8c1_training_session import pid_alive as windows_pid_alive
        return windows_pid_alive(pid)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def competing_clients():
    """Check all Python CARLA trainers/evaluators, including other worktrees."""
    if os.name == "nt":
        script = "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\\.exe$'} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError("cannot audit all machine Python processes; inspect ownership before using CARLA")
        raw = json.loads(result.stdout or "[]")
        processes = raw if isinstance(raw, list) else [raw]
        pairs = [(int(p["ProcessId"]), p.get("CommandLine") or "") for p in processes]
        # Windows venv launchers can remain alive above the actual interpreter.
        # Exclude only same-argument Python ancestors, never another sibling run.
        by_pid = {int(p["ProcessId"]): p for p in processes}
        current = by_pid.get(os.getpid(), {})
        own_tail = re.sub(r'^\s*(?:"[^"]+"|\S+)\s*', '', current.get("CommandLine") or '')
        own_launchers = set()
        parent = int(current.get("ParentProcessId") or 0)
        while own_tail and parent in by_pid and parent not in own_launchers:
            record = by_pid[parent]
            tail = re.sub(r'^\s*(?:"[^"]+"|\S+)\s*', '', record.get("CommandLine") or '')
            if tail != own_tail:
                break
            own_launchers.add(parent)
            parent = int(record.get("ParentProcessId") or 0)
        pairs = [(pid, command) for pid, command in pairs if pid not in own_launchers]
    else:
        pairs = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                tokens = path.read_bytes().decode(errors="replace").split("\0")
            except (OSError, PermissionError):
                continue
            if tokens and "python" in Path(tokens[0]).name:
                pairs.append((int(path.parent.name), " ".join(tokens)))
    markers = ("training.train_", "training.reward_fork", "evaluation.evaluate_",
               "evaluation.diagnose_", "utils.r8c1_training_session run",
               "utils.base_driving_session worker",
               "diagnostics.v6_scratch_session train")
    return [pid for pid, command in pairs if pid != os.getpid() and any(m in command for m in markers)]


class CarlaOwner:
    """An OS-released lock in a machine-wide directory, never a world operation."""
    def __init__(self, host="127.0.0.1", port=2000, run_dir=None, purpose="training"):
        if host not in {"localhost", "127.0.0.1"} or port != 2000:
            raise ValueError("Mut-MAPPO uses the shared local 2000 endpoint")
        self.path = Path(tempfile.gettempdir()) / "codex_carla_127_0_0_1_2000.lock"
        self.run_dir, self.purpose, self.handle = run_dir, purpose, None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if not self.handle.read(1):
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            competitors = competing_clients()
            if competitors:
                raise RuntimeError(f"another CARLA trainer/evaluator may own port 2000: PIDs {competitors}")
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class RunSessionOwner:
    """Hold an OS-released per-run lock for the supervisor's whole lifetime.

    A launch request does not own CARLA. This lock prevents two supervisors
    from spawning trainers or replacing the same session evidence.
    """
    def __init__(self, run_dir):
        self.path = Path(run_dir).resolve() / "session/supervisor.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            self.handle.seek(0)
            if not self.handle.read(1):
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            self.handle.close()
            self.handle = None
            if isinstance(error, OSError):
                raise RuntimeError("another supervisor owns this run session; refusing duplicate launch") from error
            raise
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _unique_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:12]


def _log_tail(path, *, max_bytes=8192, max_lines=25):
    """Read only a bounded logfile tail; never query the recorded process."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - max_bytes))
        raw = stream.read(max_bytes).decode("utf-8", errors="replace")
    return "\n".join(raw.splitlines()[-max_lines:])


def last_json_record(path):
    """Read a bounded tail; a interrupted last write is not a committed row."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 262144))
        raw = stream.read().decode("utf-8", errors="replace")
    for line in reversed(raw.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def inspect_files(run_dir):
    """Only filesystem reads: no CARLA imports, connection, or process mutation."""
    run = Path(run_dir).resolve()
    manifest = {}
    path = run / "checkpoint_manifest.json"
    if path.exists():
        for attempt in range(3):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8-sig"))
                break
            except (OSError, json.JSONDecodeError):
                if attempt == 2:
                    raise
                time.sleep(0.05)
    metrics_path = run / "metrics.jsonl"
    last = last_json_record(metrics_path)
    actual_cost = last_json_record(run / "actual_cost_audit.jsonl")
    status_path = run / "session/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else {}
    session = run / "session"
    def session_record(name):
        path = session / name
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    launch, exited = session_record("launch.json"), session_record("exit.json")
    # Dispatch requests are immutable. Even a rejected second supervisor has
    # its own log, while launch/exit remain the exclusive owner's evidence.
    dispatches = sorted(session.glob("dispatch_*.json"), key=lambda p: p.name, reverse=True)
    dispatch = session_record(dispatches[0].name) if dispatches else {}
    logs = sorted(session.glob("supervisor_*.log"), key=lambda p: p.name, reverse=True)[:5]
    trainer_log = Path(launch["log_path"]) if launch.get("log_path") else None
    return {"checked_at": now(), "run_dir": str(run), "manifest": manifest,
            "last_committed_metrics": last, "session": status,
            "launch": launch, "exit": exited, "latest_dispatch": dispatch,
            "exit_is_current_launch": bool(launch and exited and
                launch.get("launch_id") is not None and launch["launch_id"] == exited.get("launch_id")),
            "supervisor_logs": [{"path": str(path), "tail": _log_tail(path)} for path in logs],
            "trainer_log": {"path": str(trainer_log), "tail": _log_tail(trainer_log)} if trainer_log else None,
            "process_liveness": "NOT_QUERIED_FILES_ONLY",
            "dispatch_meaning": "DISPATCHED is a request, not evidence that training is RUNNING",
            "last_actual_cost_audit": actual_cost,
            "metrics_age_seconds": time.time() - metrics_path.stat().st_mtime if metrics_path.exists() else None,
            "inspection_mode": "FILES_ONLY", "automatic_restart": False}


def supervise(args):
    with RunSessionOwner(args.run_dir):
        return _supervise_owned(args)


def _supervise_owned(args):
    from training.checkpoint_protocol import atomic_write_json
    run = Path(args.run_dir).resolve()
    session = run / "session"
    session.mkdir(parents=True, exist_ok=True)
    previous = session / "launch.json"
    if previous.exists():
        record = json.loads(previous.read_text(encoding="utf-8-sig"))
        for role in ("trainer", "supervisor"):
            pid = record.get(role + "_pid")
            if pid and int(pid) != os.getpid() and pid_alive(pid):
                raise RuntimeError(f"previous {role} still alive (PID {pid}); refusing duplicate launch")
    command = [sys.executable, "-u", "-m", "training.train_mut_mappo", "--config", args.config,
               "--run-dir", str(run), "--allow-candidate"]
    if args.resume:
        command.append("--resume")
    if args.rounds is not None:
        command += ["--stop-round", str(args.rounds)]
    if getattr(args, "stop_train_ticks", None) is not None:
        command += ["--stop-train-ticks", str(args.stop_train_ticks)]
    stamp = _unique_stamp()
    log = session / f"training_{stamp}.log"
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
        print(json.dumps({"trainer_pid": process.pid, "supervisor_pid": os.getpid(),
                          "trainer_command": command, "log_path": str(log),
                          "status": "TRAINER_DISPATCHED_VERIFY_STATUS"}), flush=True)
        atomic_write_json({"trainer_pid": process.pid, "supervisor_pid": os.getpid(),
                           "command": command, "supervisor_command": [sys.executable, *sys.argv],
                           "launch_id": stamp, "started_at": now(), "log_path": str(log),
                           "supervisor_log_path": getattr(args, "supervisor_log", None),
                           "status": "TRAINER_DISPATCHED_VERIFY_STATUS"}, previous)
        while process.poll() is None:
            try:
                snapshot = inspect_files(run)
                atomic_write_json(snapshot, session / "monitor_latest.json")
                with (session / "monitor_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            except (OSError, ValueError) as error:
                atomic_write_json({"error": str(error), "at": now()}, session / "monitor_error.json")
            deadline = time.monotonic() + 1200
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(min(5, max(0.01, deadline-time.monotonic())))
        atomic_write_json({"trainer_pid": process.pid, "returncode": process.returncode,
                           "supervisor_pid": os.getpid(), "launch_id": stamp,
                           "status": "EXITED", "ended_at": now(), "automatic_restart": False}, session / "exit.json")
    return process.returncode


def launch(args):
    """Dispatch a supervisor with retained startup/error output, without CARLA."""
    from training.checkpoint_protocol import atomic_write_json
    run = Path(args.run_dir).resolve()
    session = run / "session"
    session.mkdir(parents=True, exist_ok=True)
    stamp = _unique_stamp()
    log = session / f"supervisor_{stamp}.log"
    command = [sys.executable, "-u", "-m", "utils.mut_mappo_session", "supervise",
               "--config", str((ROOT / args.config).resolve()), "--run-dir", str(run),
               "--supervisor-log", str(log)]
    if args.resume:
        command.append("--resume")
    if args.rounds is not None:
        command += ["--rounds", str(args.rounds)]
    kwargs = {"cwd": str(ROOT), "stdin": subprocess.DEVNULL,
              "stderr": subprocess.STDOUT,
              "env": dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with log.open("x", encoding="utf-8") as output:
        try:
            process = subprocess.Popen(command, stdout=output, **kwargs)
        except BaseException as error:
            output.write(f"DISPATCH_FAILED {type(error).__name__}: {error}\n")
            output.flush()
            raise
    record = {"supervisor_pid": process.pid, "status": "DISPATCHED_VERIFY_LOGS",
              "command": command, "requested_at": now(), "supervisor_log_path": str(log),
              "monitor_interval_seconds": 1200, "automatic_restart": False}
    atomic_write_json(record, session / f"dispatch_{stamp}.json")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "launch", "supervise"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/mut_mappo/local_candidate_v1.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--stop-train-ticks", type=int)
    parser.add_argument("--supervisor-log", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "inspect":
        print(json.dumps(inspect_files(args.run_dir), ensure_ascii=False, indent=2))
    elif args.mode == "supervise":
        raise SystemExit(supervise(args))
    else:
        print(json.dumps(launch(args)))


if __name__ == "__main__":
    main()
