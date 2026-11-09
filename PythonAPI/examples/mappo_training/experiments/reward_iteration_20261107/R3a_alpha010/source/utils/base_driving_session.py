"""Explicit finite base-driving controller; prepare/inspect never use CARLA.

This adapter reuses the existing MAPPO trainer and complete-state checkpoints.
The controller is serial: training, parallel regression, then validation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from training.checkpoint_protocol import atomic_copy_checkpoint, atomic_write_json, sha256_file
from training.base_driving_contract import load_contract
from training.base_driving_scenarios_v1 import load_base_driving_manifest
from utils.preflight_base_driving_scenarios_v1 import require_geometry_preflight
from utils.mut_mappo_session import RunSessionOwner, inspect_files
from utils.reward_fork_session import rows

ROOT = Path(__file__).resolve().parents[1]
VERSION = "base_driving_finite_serial_v1"
VALIDATION_DIR = ROOT / "data/training_scenarios_town05_base_vnext_v1_validation"


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_stops(stops):
    if not stops or any(type(value) is not int or not 1 <= value <= 600 for value in stops) or stops != sorted(set(stops)):
        raise ValueError("stop_updates must be explicit strictly increasing integers within [1, 600]")
    return list(stops)


def remaining_actions(stops, safe_update, completed_screens):
    """A milestone reached before a crash is screened first, never retrained."""
    validate_stops(stops)
    if type(safe_update) is not int or not 0 <= safe_update <= stops[-1]:
        raise ValueError("safe boundary outside the finite authorized budget")
    if (completed_screens != sorted(set(completed_screens))
            or any(type(update) is not int or update not in stops or update > safe_update for update in completed_screens)):
        raise ValueError("completed screen lies outside the safe authorized milestones")
    missing_earlier = [stop for stop in stops if stop < safe_update and stop not in completed_screens]
    if missing_earlier:
        raise ValueError(f"cannot skip incomplete earlier frozen milestones: {missing_earlier}")
    actions, current = [], safe_update
    for stop in stops:
        if stop in completed_screens:
            continue
        if stop > current:
            actions.append(("train", current, stop))
            current = stop
        actions.append(("screen", stop, stop))
    return actions


def _within(path, parent):
    resolved = Path(path).resolve()
    resolved.relative_to(Path(parent).resolve())
    if resolved == Path(parent).resolve():
        raise ValueError("a dedicated child directory is required")
    return resolved


def _source_paths(metadata):
    from training.base_driving_contract import source_paths
    archive = read(ROOT / "reproducibility/v6_before_base_driving_20261020.json")
    paths = {ROOT / name for name in archive["registered_sources"]}
    paths.update(source_paths(ROOT, metadata))
    paths.update({Path(__file__), ROOT / "utils/build_base_driving_scenarios_v1.py",
                  ROOT / "utils/preflight_base_driving_scenarios_v1.py",
                  ROOT / "evaluation/evaluate_base_driving_vnext.py",
                  ROOT / "reproducibility/v6_before_base_driving_20261020.json",
                  VALIDATION_DIR / "protocol/manifest.json"})
    return sorted(paths)


def prepare(args):
    """Write only a new immutable finite plan; this never dispatches a process."""
    if any(getattr(args, name, None) is None for name in ("run_dir", "control_dir", "stop_updates", "train_geometry", "validation_geometry")):
        raise ValueError("prepare requires run/control directories, explicit stops and both geometry report paths")
    stops = validate_stops(args.stop_updates)
    run, control = _within(args.run_dir, ROOT / "logs"), _within(args.control_dir, ROOT / "diagnostics")
    if run.exists() or control.exists():
        raise FileExistsError("prepare requires new run and control directories; inspect/recover the previous plan explicitly")
    metadata = load_contract(args.contract, ROOT)
    load_base_driving_manifest(VALIDATION_DIR)
    geometry_paths = {"train": str(Path(args.train_geometry).resolve()), "validation": str(Path(args.validation_geometry).resolve())}
    # Offline pending reports may be pinned by path at preparation. Actual live
    # reports replace them later and are checked immediately before dispatch.
    geometry_prepared = {key: {"path": path, "sha256": sha256_file(path), "status": read(path).get("status")} for key, path in geometry_paths.items()}
    record = {
        "version": VERSION, "prepared_at": now(), "run_dir": str(run), "control_dir": str(control),
        "run_id": run.name, "seed": int(args.seed), "stop_updates": stops,
        "total_budget_updates": stops[-1], "total_budget_episodes": stops[-1] * 16,
        "source_type": "fresh", "source_update": 0,
        "contract_path": metadata["path"], "contract_record": metadata["record"],
        "train_scenario_dir": metadata["scenario_dir"], "validation_scenario_dir": str(VALIDATION_DIR.resolve()),
        "validation_manifest_sha256": sha256_file(VALIDATION_DIR / "protocol/manifest.json"),
        "geometry_reports": geometry_paths, "geometry_at_prepare": geometry_prepared,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in _source_paths(metadata)},
        "evaluation_order": ["parallel_training_regression_16x2", "validation_32x2", "selection"],
        "monitor_interval_seconds": 1200, "automatic_restart": False,
        "meaning": "PREPARED is not training; explicit run and passed geometry gates are required",
    }
    run.mkdir(parents=True, exist_ok=False)
    (run / "session").mkdir()
    control.mkdir(parents=True, exist_ok=False)
    plan = control / "plan.json"
    atomic_write_json(record, plan)
    atomic_write_json({"version": "base_driving_fresh_v1", "source_type": "fresh", "source_update": 0,
                       "plan_path": str(plan), "plan_sha256": sha256_file(plan), "created_at": now(),
                       "status": "PREPARED_NOT_TRAINED"}, run / "initialization.json")
    print(json.dumps({"status": "PREPARED_NOT_TRAINED", "plan": str(plan), "stop_updates": stops}, indent=2))
    return record


def load_plan(path):
    path = Path(path).resolve()
    record = read(path)
    validate_stops(record["stop_updates"])
    if (record.get("version") != VERSION or record.get("source_type") != "fresh" or record.get("source_update") != 0
            or record.get("total_budget_updates") != record["stop_updates"][-1]
            or record.get("total_budget_episodes") != record["stop_updates"][-1] * 16
            or record.get("automatic_restart") is not False):
        raise ValueError("base-driving finite plan identity or budget changed")
    run, control = _within(record["run_dir"], ROOT / "logs"), _within(record["control_dir"], ROOT / "diagnostics")
    if path != control / "plan.json" or record["run_id"] != run.name:
        raise ValueError("plan must remain in its declared control directory")
    initialization = read(run / "initialization.json")
    if (initialization.get("version") != "base_driving_fresh_v1" or initialization.get("source_update") != 0
            or initialization.get("source_type") != "fresh" or initialization.get("plan_sha256") != sha256_file(path)):
        raise ValueError("run initialization/plan binding changed")
    metadata = load_contract(record["contract_path"], ROOT)
    if metadata["record"] != record["contract_record"] or metadata["scenario_dir"] != record["train_scenario_dir"]:
        raise ValueError("base-driving contract or scenario pool changed")
    if record["validation_scenario_dir"] != str(VALIDATION_DIR.resolve()) or record["validation_manifest_sha256"] != sha256_file(VALIDATION_DIR / "protocol/manifest.json"):
        raise ValueError("validation pool identity changed")
    load_base_driving_manifest(VALIDATION_DIR)
    expected_sources = {source.relative_to(ROOT).as_posix() for source in _source_paths(metadata)}
    if set(record["source_sha256"]) != expected_sources:
        raise ValueError("registered source set changed")
    for name, digest in record["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"registered source changed after preparation: {name}")
    return record


def require_geometry(record):
    return {
        split: require_geometry_preflight(record[f"{split}_scenario_dir"], record["geometry_reports"][split])
        for split in ("train", "validation")
    }


def safe_boundary(record, *, expected=None):
    """Read and validate complete safe state without importing the trainer/CARLA."""
    from training.checkpoint_protocol import load_checkpoint_file, validate_checkpoint_header, validate_checkpoint_payload
    from diagnostics.v6_scratch_session import training_integrity
    run = Path(record["run_dir"])
    if not (run / "checkpoint_manifest.json").exists():
        if expected not in (None, 0) or (run / "latest.pt").exists():
            raise ValueError("requested safe boundary has no manifest")
        return 0, None, ""
    latest = read(run / "checkpoint_manifest.json").get("latest")
    if not latest:
        raise ValueError("run manifest does not register a safe checkpoint")
    checkpoint, digest = load_checkpoint_file(run / "latest.pt", map_location="cpu")
    validate_checkpoint_header(checkpoint)
    update = checkpoint["update"]
    metadata = checkpoint["run_metadata"]
    if (checkpoint.get("checkpoint_schema_version") != 8 or update != latest.get("update")
            or digest != latest.get("sha256") or not 1 <= update <= record["stop_updates"][-1]
            or checkpoint.get("total_episodes") != update * 16
            or checkpoint.get("base_driving_contract") != record["contract_record"]
            or metadata.get("run_id") != record["run_id"] or metadata.get("seed") != record["seed"]
            or metadata.get("source_type") != "fresh" or metadata.get("source_checkpoint")
            or metadata.get("code_drift_override") or "reward_fork" in metadata):
        raise ValueError("checkpoint is not the same complete fresh base-driving run/safe manifest boundary")
    if expected is not None and update != expected:
        raise ValueError(f"safe update {update} differs from required boundary {expected}")
    validate_checkpoint_payload(checkpoint, expected_protocol_versions=checkpoint["protocol_versions"],
        expected_action_spec=checkpoint["action_spec"], expected_model_spec=checkpoint["model_spec"],
        expected_training_config=checkpoint["training_config"], expected_critic_spec=checkpoint["critic_spec"])
    from training.base_driving_runtime import load_adapter
    metadata = load_contract(record["contract_path"], ROOT)
    adapter = load_adapter(record["train_scenario_dir"], config=metadata["payload"]["curriculum_config"])
    adapter.load_state_dict(checkpoint.get("base_driving_state"), expected_completed_update=update)
    adapter.validate_checkpoint_histories(checkpoint.get("curriculum_state"))
    if sum(adapter.scenario_total_sample_counts.values()) != update * 16:
        raise ValueError("complete-state scenario exposure does not match safe episode count")
    training_integrity(run, update)
    return update, checkpoint, digest


def inspect(path):
    """Pure file monitoring, restricted to the manifest's committed boundary."""
    record = load_plan(path)
    run, control = Path(record["run_dir"]), Path(record["control_dir"])
    evidence = inspect_files(run)
    safe = int(evidence["manifest"].get("latest", {}).get("update", 0))
    metrics = [row for row in rows(run / "metrics.jsonl") if 0 < row["update"] <= safe]
    evidence.update(protocol=VERSION, source_update=0, safe_update=safe,
                    committed_metric_updates=len(metrics), last_committed_metrics=metrics[-1] if metrics else None,
                    current_control=read(control / "status.json") if (control / "status.json").exists() else {"status": "PREPARED_NOT_TRAINED"},
                    camera=read(run / "session/camera.json") if (run / "session/camera.json").exists() else {},
                    interpretation="Training is not frozen driving qualification; PID/dispatch is not training success.")
    if safe and len(metrics) == safe:
        from diagnostics.v6_scratch_session import training_integrity
        evidence["integrity"] = training_integrity(run, safe)
    else:
        evidence["metrics_commit_pending"] = bool(safe)
    return evidence


def save_snapshot(path):
    snapshot = inspect(path)
    session = Path(snapshot["run_dir"]) / "session"
    atomic_write_json(snapshot, session / "monitor_latest.json")
    with (session / "monitor_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
    return snapshot


def training_args(record, start, stop):
    # The trainer imports CARLA classes; only the explicit live worker calls here.
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from training.base_driving_contract import configure
    from training.checkpoint_protocol import files_fingerprint, validate_checkpoint_payload
    if stop not in record["stop_updates"] or not 0 <= start < stop:
        raise ValueError("worker stage is outside the explicit finite plan")
    if any(start < value < stop for value in record["stop_updates"]):
        raise ValueError("worker cannot skip a frozen milestone")
    _, checkpoint, _ = safe_boundary(record, expected=start)
    args = trainer.build_parser().parse_args([])
    args.base_driving_contract = record["contract_path"]
    args.base_driving_geometry_report = record["geometry_reports"]["train"]
    args.save_dir, args.run_id, args.seed, args.updates = record["run_dir"], record["run_id"], record["seed"], stop
    args.resume, args.pretrained_checkpoint = ("latest" if checkpoint is not None else ""), ""
    args.follow_spectator, args.allow_resume_code_drift = True, False
    args.stop_request_file = str(Path(record["run_dir"]) / "session/STOP_REQUESTED")
    args._explicit_cli_options = {"--seed", "--run-id"}
    metadata = configure(args, ROOT)
    args._base_driving_clock_owned = True
    if checkpoint is not None:
        trainer.apply_resume_configuration(args, checkpoint)
        from training.base_driving_contract import source_paths
        paths = trainer.training_source_paths(ROOT) + source_paths(ROOT, metadata)
        if checkpoint["run_metadata"]["code_fingerprint"] != files_fingerprint(paths):
            raise ValueError("complete-state resume source fingerprint changed")
        validate_checkpoint_payload(checkpoint, expected_protocol_versions=checkpoint["protocol_versions"],
            expected_action_spec=checkpoint["action_spec"], expected_model_spec=checkpoint["model_spec"],
            expected_training_config=trainer.checkpoint_training_config(args), expected_critic_spec=checkpoint["critic_spec"])
    return args


def archive_session(record, *, reason):
    """Preserve raw interruption/tail evidence; never invent a successful exit."""
    run, control = Path(record["run_dir"]), Path(record["control_dir"])
    session = run / "session"
    if not (session / "launch.json").exists():
        return
    launch = read(session / "launch.json")
    destination = control / f"{reason}_session_{launch['launch_id']}_{uuid.uuid4().hex[:8]}"
    destination.mkdir()
    for name in ("episodes.jsonl", "metrics.jsonl", "checkpoint_manifest.json"):
        if (run / name).exists():
            shutil.copy2(run / name, destination / name)
    for name in ("launch.json", "exit.json", "camera.json", "monitor_latest.json"):
        if (session / name).exists():
            shutil.copy2(session / name, destination / name)
    if Path(launch["log_path"]).exists():
        shutil.copy2(launch["log_path"], destination / "training.log")
    safe = int(read(run / "checkpoint_manifest.json").get("latest", {}).get("update", 0)) if (run / "checkpoint_manifest.json").exists() else 0
    atomic_write_json({"at": now(), "safe_update": safe,
                       "uncommitted_episodes": [row for row in rows(run / "episodes.jsonl") if row["update"] > safe],
                       "uncommitted_metrics": [row for row in rows(run / "metrics.jsonl") if row["update"] > safe],
                       "exit_present": (session / "exit.json").exists(),
                       "meaning": "raw evidence; absence of matching exit is not success"}, destination / "boundary_review.json")
    for name in ("launch.json", "exit.json"):
        if (session / name).exists():
            (session / name).replace(destination / f"original_{name}")
    return str(destination)


def worker(path, start, stop, launch_id):
    record = load_plan(path)
    # This independent lease survives parent death and protects the evidence
    # during startup, model loading, simulator use and cleanup.
    with RunSessionOwner(Path(record["run_dir"]) / "worker_guard"):
        _worker_owned(path, record, start, stop, launch_id)


def _worker_owned(path, record, start, stop, launch_id):
    session = Path(record["run_dir"]) / "session"
    # Parent records the actual launcher before this child publishes its own PID.
    deadline = time.monotonic() + 15
    while True:
        launch = read(session / "launch.json")
        if launch.get("launch_id") != launch_id:
            raise ValueError("worker launch identity mismatch")
        if launch.get("spawned_pid") is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("parent did not commit launcher identity")
        time.sleep(0.05)
    launch.update(actual_worker_pid=os.getpid(), trainer_pid=os.getpid(), worker_started_at=now(), status="WORKER_STARTED_NOT_YET_TRAINED")
    atomic_write_json(launch, session / "launch.json")
    require_geometry(record)
    args = training_args(record, start, stop)
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    with CarlaOwner(run_dir=record["run_dir"], purpose="finite base-driving full-state training"):
        _preflight_idle_world(host="127.0.0.1", port=2000, expected_version="0.9.16")
        trainer.train(args)
    safe_boundary(record, expected=stop)


def _assert_worker_idle(record, path):
    """Reject active workers before replacing evidence, without trusting PID reuse."""
    run = Path(record["run_dir"])
    with RunSessionOwner(run / "worker_guard"):
        pass
    launch_path = run / "session/launch.json"
    if not launch_path.exists():
        return
    launch = read(launch_path)
    stamp = launch.get("launch_id")
    pids = sorted({pid for pid in (launch.get("actual_worker_pid"), launch.get("spawned_pid")) if type(pid) is int and pid > 0})
    if not stamp or not pids:
        return
    commands = []
    if os.name == "nt":
        selected = ",".join(str(pid) for pid in pids)
        script = ("$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | "
                  f"Where-Object {{$_.ProcessId -in @({selected})}} | "
                  "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError("cannot audit previous launch process identity; no session evidence was replaced")
        records = json.loads(result.stdout or "[]")
        if isinstance(records, dict):
            records = [records]
        commands = [item.get("CommandLine") or "" for item in records]
    else:
        for pid in pids:
            cmdline = Path(f"/proc/{pid}/cmdline")
            try:
                commands.append(cmdline.read_bytes().decode("utf-8", errors="replace").replace("\0", " "))
            except FileNotFoundError:
                continue
    required = ("utils.base_driving_session", "worker", str(stamp).casefold(), str(Path(path).resolve()).replace("\\", "/").casefold())
    for command in commands:
        normalized = command.replace("\\", "/").casefold()
        if all(token in normalized for token in required):
            raise RuntimeError("previous UUID-matched worker is still active; refusing to replace its session")


def supervise(path, record, start, stop):
    session, stamp = Path(record["run_dir"]) / "session", uuid.uuid4().hex
    if (session / "launch.json").exists() or (session / "exit.json").exists():
        raise ValueError("prior session must be archived before dispatch")
    log = session / f"training_{stamp}.log"
    command = [sys.executable, "-u", "-m", "utils.base_driving_session", "worker", "--plan", str(path),
               "--from-update", str(start), "--stop-update", str(stop), "--launch-id", stamp]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    launch = {"launch_id": stamp, "spawned_pid": None, "actual_worker_pid": None, "trainer_pid": None,
              "supervisor_pid": os.getpid(), "command": command, "log_path": str(log), "started_at": now(),
              "resume_update": start, "stop_update": stop, "status": "DISPATCHED_NOT_TRAINING_SUCCESS", "automatic_restart": False}
    atomic_write_json(launch, session / "launch.json")
    camera = None
    with log.open("x", encoding="utf-8") as output, (session / f"camera_{stamp}.log").open("x", encoding="utf-8") as camera_output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        launch["spawned_pid"] = child.pid
        atomic_write_json(launch, session / "launch.json")
        deadline = 0
        while child.poll() is None:
            actual = read(session / "launch.json").get("actual_worker_pid")
            if actual and camera is None:
                camera = subprocess.Popen([sys.executable, "-u", "-m", "utils.follow_reward_fork", "--trainer-pid", str(actual), "--run-dir", record["run_dir"]], cwd=ROOT,
                    stdout=camera_output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, creationflags=flags)
            if time.monotonic() >= deadline:
                try:
                    save_snapshot(path)
                except Exception as error:
                    atomic_write_json({"at": now(), "error": f"{type(error).__name__}: {error}"}, session / "monitor_error.json")
                deadline = time.monotonic() + 1200
            time.sleep(2)
        if camera is not None:
            try:
                camera.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        current = read(session / "launch.json")
        atomic_write_json({"launch_id": stamp, "spawned_pid": child.pid,
                           "trainer_pid": current.get("actual_worker_pid"), "supervisor_pid": os.getpid(),
                           "returncode": child.returncode, "ended_at": now(), "automatic_restart": False,
                           "camera_pid": camera.pid if camera else None, "camera_returncode": camera.poll() if camera else None}, session / "exit.json")
    if child.returncode != 0:
        raise RuntimeError(f"training exited {child.returncode}; explicit recovery required")
    if not current.get("actual_worker_pid"):
        raise RuntimeError("successful launcher exit has no actual worker identity")
    safe_boundary(record, expected=stop)
    save_snapshot(path)


def _matching_exit(run, *, stop):
    session = Path(run) / "session"
    if not (session / "launch.json").exists() or not (session / "exit.json").exists():
        return False
    launch, exited = read(session / "launch.json"), read(session / "exit.json")
    return bool(launch.get("launch_id") and launch["launch_id"] == exited.get("launch_id")
                and launch.get("stop_update") == stop and exited.get("returncode") == 0
                and launch.get("actual_worker_pid") and launch["actual_worker_pid"] == exited.get("trainer_pid"))


def _completed_screens(record):
    """Only persisted, hash-bound, complete stage summaries count as done."""
    from evaluation.base_driving_qualification import analyze_evaluation, select_candidate
    control = Path(record["control_dir"])
    completed = []
    for stop in record["stop_updates"]:
        path = control / f"u{stop:06d}_summary.json"
        if not path.exists():
            continue
        summary = read(path)
        if (summary.get("status") != "COMPLETED" or summary.get("update") != stop
                or summary.get("checkpoint_sha256") != sha256_file(summary["checkpoint"])):
            raise ValueError(f"completed frozen stage identity changed: {path}")
        suite_reports = {}
        for name in ("parallel", "validation"):
            directory = Path(summary[f"{name}_directory"])
            if summary[f"{name}_qualification_sha256"] != sha256_file(directory / "qualification.json"):
                raise ValueError(f"completed {name} qualification file changed")
            report = analyze_evaluation(directory, ROOT / load_contract(record["contract_path"], ROOT)["payload"]["qualification_contract"])
            if {item["checkpoint_sha256"] for item in report["reports"]} != {summary["checkpoint_sha256"]}:
                raise ValueError("frozen screen does not evaluate its recorded checkpoint")
            suite_reports[name] = report["reports"]
        if summary.get("selection") != select_candidate(suite_reports["validation"], regression_reports=suite_reports["parallel"]):
            raise ValueError("completed selection disagrees with verified regression/validation evidence")
        completed.append(stop)
    return completed


def frozen_screen(record, stop, *, explicit_safe_boundary_recovery=False):
    """Freeze a complete checkpoint and evaluate serially without an outer CARLA lease."""
    from diagnostics.v6_scratch_session import training_integrity
    from evaluation.evaluate_base_driving_vnext import evaluate
    from evaluation.base_driving_qualification import select_candidate
    run, control = Path(record["run_dir"]), Path(record["control_dir"])
    _, _, digest = safe_boundary(record, expected=stop)
    matched = _matching_exit(run, stop=stop)
    if not matched and not explicit_safe_boundary_recovery:
        raise ValueError("normal stage evaluation requires a matching actual worker exit 0")
    recovery_archive = archive_session(record, reason="explicit_safe_boundary_recovery") if explicit_safe_boundary_recovery else None
    frozen = control / f"u{stop:06d}.pt"
    if frozen.exists():
        if sha256_file(frozen) != digest:
            raise ValueError("immutable frozen checkpoint changed")
    else:
        atomic_copy_checkpoint(run / "latest.pt", frozen)
    stage = control / f"screen_u{stop:06d}"
    if stage.exists():
        # Explicit controller resume starts a clean complete suite; cases are
        # not resumed individually and old failure evidence remains intact.
        stage.replace(control / f"interrupted_screen_u{stop:06d}_{uuid.uuid4().hex}")
    stage.mkdir()
    qualification = ROOT / load_contract(record["contract_path"], ROOT)["payload"]["qualification_contract"]
    parallel = evaluate([frozen], stage / "parallel", Path(record["train_scenario_dir"]) / "protocol/manifest.json",
                        qualification, geometry_report=record["geometry_reports"]["train"], parallel_regression=True)
    validation = evaluate([frozen], stage / "validation", Path(record["validation_scenario_dir"]) / "protocol/manifest.json",
                          qualification, geometry_report=record["geometry_reports"]["validation"])
    selection = select_candidate(validation["reports"], regression_reports=parallel["reports"])
    atomic_write_json(selection, stage / "selection.json")
    if sha256_file(frozen) != digest:
        raise ValueError("frozen checkpoint changed during evaluation")
    summary = {"status": "COMPLETED", "update": stop, "ended_at": now(), "checkpoint": str(frozen),
               "checkpoint_sha256": digest, "parallel_directory": str(stage / "parallel"),
               "validation_directory": str(stage / "validation"),
               "parallel_qualification_sha256": sha256_file(stage / "parallel/qualification.json"),
               "validation_qualification_sha256": sha256_file(stage / "validation/qualification.json"),
               "selection": selection, "training_integrity": training_integrity(run, stop),
               "normal_training_matching_exit_0": matched,
               "explicit_safe_boundary_recovery": bool(explicit_safe_boundary_recovery and not matched),
               "prior_session_archive": recovery_archive,
               "qualification_scope": "engineering validation and training regression; not formal generalization"}
    atomic_write_json(summary, control / f"u{stop:06d}_summary.json")
    return summary


def controller(path, *, resume=False):
    """Explicit one-shot controller. Exceptions stop it; there is no retry loop."""
    path = Path(path).resolve()
    record = load_plan(path)
    run, control = Path(record["run_dir"]), Path(record["control_dir"])
    with ExitStack() as owners:
        for directory in (run, control):
            owners.enter_context(RunSessionOwner(directory))
        state_path = control / "status.json"
        if state_path.exists() and not resume:
            raise FileExistsError("controller already dispatched; inspect and explicitly use --resume-controller")
        if resume and not state_path.exists():
            raise ValueError("--resume-controller requires a prior dispatched controller")
        previous = read(state_path) if state_path.exists() else None
        if previous and previous.get("status") == "COMPLETED":
            raise ValueError("finite plan already completed; its budget cannot be extended")
        try:
            # Ownership probes precede any session archive or status replacement.
            _assert_worker_idle(record, path)
            geometry = require_geometry(record)
            if resume:
                from utils.mut_mappo_session import CarlaOwner
                with CarlaOwner(run_dir=run, purpose="explicit controller recovery ownership probe"):
                    pass
            safe, _, _ = safe_boundary(record)
            completed = _completed_screens(record)
            actions = remaining_actions(record["stop_updates"], safe, completed)
            if not resume and safe:
                raise ValueError("fresh controller cannot implicitly resume an existing checkpoint")
            if safe == 0 and any((run / name).exists() for name in ("metrics.jsonl", "episodes.jsonl", "metrics_schema.json")):
                raise ValueError("failed before first safe checkpoint; preserve evidence and prepare an explicitly new fresh run")
        except BaseException as error:
            atomic_write_json({"status": "PRECHECK_FAILED", "at": now(), "resume_controller": bool(resume),
                               "error": f"{type(error).__name__}: {error}", "session_evidence_replaced": False},
                              control / f"precheck_failed_{uuid.uuid4().hex}.json")
            raise
        recovery_id = uuid.uuid4().hex
        if previous:
            atomic_write_json(previous, control / f"interrupted_controller_{recovery_id}.json")
        state = {"version": VERSION, "status": "STARTING", "pid": os.getpid(), "started_at": now(),
                 "plan_sha256": sha256_file(path), "completed_stages": completed,
                 "geometry_report_sha256": {key: sha256_file(record["geometry_reports"][key]) for key in geometry},
                 "resume_controller": bool(resume), "initial_safe_update": safe, "automatic_restart": False}
        atomic_write_json(state, state_path)
        try:
            for action, start, stop in actions:
                load_plan(path)
                if (run / "session/STOP_REQUESTED").exists():
                    raise ValueError("STOP_REQUESTED exists; no implicit override")
                state.update(status="TRAINING" if action == "train" else "FROZEN_EVALUATION", target_update=stop, changed_at=now())
                atomic_write_json(state, state_path)
                if action == "train":
                    safe_boundary(record, expected=start)
                    archive_session(record, reason="interrupted" if resume and start == safe else "completed")
                    supervise(path, record, start, stop)
                else:
                    summary = frozen_screen(record, stop, explicit_safe_boundary_recovery=bool(resume and stop == safe))
                    state["completed_stages"].append(stop)
                    state["last_stage_selection"] = summary["selection"]
                    atomic_write_json(state, state_path)
            state.update(status="COMPLETED", ended_at=now())
            atomic_write_json(state, state_path)
            save_snapshot(path)
        except BaseException as error:
            state.update(status="FAILED_NO_RESTART", error=f"{type(error).__name__}: {error}", ended_at=now())
            atomic_write_json(state, state_path)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "inspect", "run", "worker"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/base_driving/base_driving_v6_v1.json")
    parser.add_argument("--stop-updates", type=int, nargs="+")
    parser.add_argument("--train-geometry", type=Path)
    parser.add_argument("--validation-geometry", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--from-update", type=int)
    parser.add_argument("--stop-update", type=int)
    parser.add_argument("--launch-id")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "inspect":
        print(json.dumps(inspect(args.plan), ensure_ascii=False, indent=2))
    elif args.mode == "worker":
        worker(args.plan, args.from_update, args.stop_update, args.launch_id)
    else:
        controller(args.plan, resume=args.resume_controller)


if __name__ == "__main__":
    main()
