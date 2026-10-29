"""Runbook for the user-authorized V5 u461 -> u481 -> u506 adaptation.

This main-task controller reuses existing supervisors and evaluators serially.
It never imports CARLA, retries failed work, changes rewards, or starts CARLA.
The recurring monitor only reads its status files.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
RUN = ROOT / "logs/R8C1_InstantPaceV5_u456_to461_s42_20261013_v1"
DOC = ROOT / "docs/基础红策略_V5持续适应实验_20261015.md"
STOPS = (481, 506)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def command(module, *args, capture=False):
    cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
    print(json.dumps({"at": stamp(), "command": cmd}, ensure_ascii=False), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                            stdout=subprocess.PIPE if capture else None, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{module} exited {result.returncode}; no automatic retry")
    return json.loads(result.stdout) if capture else None


def snapshot(expected):
    manifest = read(RUN / "checkpoint_manifest.json")
    if manifest["latest"]["update"] != expected:
        raise RuntimeError(f"expected complete u{expected} boundary")
    digest = hashlib.sha256((RUN / "latest.pt").read_bytes()).hexdigest()
    if digest != manifest["latest"]["sha256"]:
        raise RuntimeError("latest checkpoint/manifest checksum mismatch")
    destination = CONTROL / f"u{expected:06d}.pt"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(RUN / "latest.pt", destination)
    destination.with_suffix(".pt.sha256").write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise RuntimeError("immutable checkpoint copy mismatch")
    write(CONTROL / f"u{expected:06d}_checkpoint_manifest.json", manifest)
    return destination, digest


def archive_completed_session():
    # A completed old PID may belong to an unrelated new process on Windows.
    # Retain both records together; never bypass a current/incomplete session.
    launch_path, exit_path = RUN / "session/launch.json", RUN / "session/exit.json"
    if not launch_path.exists():
        return
    launched = read(launch_path)
    exited = read(exit_path) if exit_path.exists() else {}
    launch_id = launched.get("launch_id")
    if not launch_id or exited.get("launch_id") != launch_id or exited.get("returncode") != 0:
        raise RuntimeError("previous session is incomplete or failed; explicit review required")
    destination = CONTROL / ("completed_session_" + launch_id)
    destination.mkdir(exist_ok=False)
    for name in ("camera.json", "monitor_latest.json"):
        source = RUN / "session" / name
        if source.exists():
            shutil.copy2(source, destination / name)
    launch_path.replace(destination / "launch.json")
    exit_path.replace(destination / "exit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = read(RUN / "checkpoint_manifest.json")
    if manifest["latest"]["update"] != 461:
        raise RuntimeError("this one-shot runbook requires the reviewed u461 starting boundary")
    for stop in STOPS:
        if (CONTROL / f"screen_u{stop:06d}").exists():
            raise FileExistsError("an output already exists; explicit recovery review required")
    if (RUN / "session/STOP_REQUESTED").exists():
        raise RuntimeError("STOP_REQUESTED exists")
    plan = {"authorized_on": "2026-10-15 Asia/Shanghai", "source_update": 456, "resume_update": 461,
            "stop_updates": list(STOPS), "new_updates_from_source": [25, 50], "run_dir": str(RUN),
            "profile": "instant_pace_v5", "reward_and_scenes": "unchanged", "evaluation_seeds": [62001],
            "automatic_restart": False, "monitor": "files only; never advances CARLA",
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "engineering adaptation; no automatic normal-driving or research qualification"}
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if (CONTROL / "plan.json").exists():
        raise FileExistsError("one-shot controller was already started; no automatic rerun")
    write(CONTROL / "plan.json", plan)
    state = {"started_at": stamp(), "pid": os.getpid(), "status": "STARTING", "completed_stages": [], **plan}
    write(CONTROL / "status.json", state)
    try:
        snapshot(461)
        for stop in STOPS:
            if (RUN / "session/STOP_REQUESTED").exists():
                raise RuntimeError("STOP_REQUESTED prevents the next planned stage")
            state.update(status="TRAINING", target_update=stop, changed_at=stamp())
            write(CONTROL / "status.json", state)
            command("training.reward_fork_v2", "--run-dir", RUN, "--stop-update", stop, "--validate-only")
            archive_completed_session()
            command("utils.reward_fork_session", "supervise", "--run-dir", RUN, "--stop-update", stop)
            evidence = command("utils.reward_fork_session", "inspect", "--run-dir", RUN, capture=True)
            write(CONTROL / f"u{stop:06d}_training_inspect.json", evidence)
            if (evidence["safe_update"] != stop or evidence["integrity_issues"]
                    or evidence["metrics_commit_pending"] or not evidence["exit_is_current_launch"]
                    or evidence["exit"].get("returncode") != 0):
                raise RuntimeError("training did not finish at a healthy complete boundary")
            checkpoint, digest = snapshot(stop)
            evaluation = CONTROL / f"screen_u{stop:06d}"
            state.update(status="FROZEN_EVALUATION", checkpoint=str(checkpoint), checkpoint_sha256=digest,
                         evaluation_dir=str(evaluation), changed_at=stamp())
            write(CONTROL / "status.json", state)
            command("diagnostics.run_reward_screen", "--checkpoints", checkpoint, "--output", evaluation, "--seeds", 62001)
            command("diagnostics.analyze_reward_screen", "--evaluation-dir", evaluation,
                    "--output", evaluation / "behavior_analysis.json")
            analysis = read(evaluation / "behavior_analysis.json")
            if analysis["analysis_evidence_status"] != "COMPLETE_VALID_EVIDENCE" or analysis["errors"]:
                raise RuntimeError("frozen evaluation evidence is incomplete")
            summary = {"update": stop, "new_updates_from_u456": stop - 456,
                       "checkpoint": str(checkpoint), "sha256": digest, "evaluation_dir": str(evaluation),
                       "aggregate": analysis["aggregate"], "completed_at": stamp(),
                       "qualification": "NOT_ASSIGNED_ENGINEERING_DESCRIPTION_ONLY"}
            write(CONTROL / f"u{stop:06d}_summary.json", summary)
            state["completed_stages"].append(summary)
            write(CONTROL / "status.json", state)
            with DOC.open("a", encoding="utf-8") as stream:
                stream.write(f"\n## u{stop} 已保存并完成冻结评估\n\n"
                             f"记录时间（UTC）：{stamp()}；累计新updates {stop-456}。"
                             f"有效案例{analysis['aggregate']['valid_case_count']}，"
                             f"两车共同到达{analysis['aggregate']['both_reached_count_valid_cases']}/12。"
                             f"完整指标见 `{evaluation / 'behavior_analysis.json'}`。"
                             "这是已知场景单seed工程筛选，尚未授予正常驾驶/泛化资格。\n")
            if analysis["aggregate"]["both_reached_count_valid_cases"] == 0:
                raise RuntimeError("all screening cases failed; stop before further training")
        state.update(status="COMPLETED", ended_at=stamp(), qualification="REQUIRES_MAIN_TASK_REVIEW")
        write(CONTROL / "status.json", state)
    except BaseException as error:
        state.update(status="FAILED_NO_RESTART", error=f"{type(error).__name__}: {error}", ended_at=stamp())
        write(CONTROL / "status.json", state)
        raise


if __name__ == "__main__":
    main()
