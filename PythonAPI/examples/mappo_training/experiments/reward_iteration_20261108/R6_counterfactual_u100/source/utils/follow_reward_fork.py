"""Spectator-only helper: never ticks, resets, loads, or spawns in CARLA."""
import argparse
import os
from pathlib import Path
import threading
from utils.r8c1_training_session import follow, pid_alive, write_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    done = threading.Event()
    status = {"pid": os.getpid(), "trainer_pid": args.trainer_pid, "started_at": now()}
    thread = threading.Thread(target=follow, args=(done, status), daemon=True)
    thread.start()
    try:
        while pid_alive(args.trainer_pid):
            write_json(args.run_dir / "session/camera.json", dict(status))
            done.wait(2)
    finally:
        done.set()
        thread.join(timeout=4)
        status["status"] = "EXITED"
        status["ended_at"] = now()
        write_json(args.run_dir / "session/camera.json", status)


if __name__ == "__main__":
    main()
