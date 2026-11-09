"""Detached base-reward experiment with tracking camera and file-only monitoring."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from utils.mut_mappo_session import RunSessionOwner, inspect_files, pid_alive
from training.checkpoint_protocol import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def rows(path):
    if not path.exists():
        return []
    result = []
    raw = path.read_text(encoding='utf-8')
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index == len(lines)-1 and not raw.endswith('\n'):
                continue  # A writer's incomplete final line is not committed evidence.
            raise ValueError(f'{path}: invalid committed JSONL line {index+1}') from error
    return result


def inspect(run_dir):
    """Read only files; no CARLA import/connection or process inspection."""
    run = Path(run_dir).resolve()
    evidence = inspect_files(run)
    manifest = evidence['manifest']
    boundary = int(manifest.get('latest', {}).get('update', 0))
    fork_path, fresh_path = run / 'reward_fork.json', run / 'initialization.json'
    if fork_path.exists() and fresh_path.exists():
        raise ValueError('ambiguous fork and fresh initialization evidence')
    if fork_path.exists():
        source = int(json.loads(fork_path.read_text(encoding='utf-8'))['source_update'])
    else:
        fresh = json.loads(fresh_path.read_text(encoding='utf-8'))
        if fresh.get('version') != 'v6_scratch_v1' or fresh.get('source_update') != 0:
            raise ValueError('unregistered fresh initialization evidence')
        source = 0
    metrics = [r for r in rows(run / 'metrics.jsonl') if source < r['update'] <= boundary]
    window = metrics[-20:]
    first = window[0]['update'] if window else source + 1
    episodes = [r for r in rows(run / 'episodes.jsonl') if first <= r['update'] <= boundary]
    issues = []
    if [r['update'] for r in metrics] != list(range(source+1, source+len(metrics)+1)):
        issues.append('committed metrics contain a duplicate or missing update')
    for row in window:
        for key, value in row['scalars'].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                issues.append(f"u{row['update']} nonfinite {key}")
            if any(k in key for k in ('preupdate_max_abs_ratio_error', 'max_abs_step_residual')) and abs(value) > 1e-6:
                issues.append(f"u{row['update']} {key}={value}")
            if key == 'train/update_skipped' and value:
                issues.append(f"u{row['update']} skipped PPO update")
    last = window[-1]['scalars'] if window else {}
    selected = {k: v for k, v in last.items() if k.startswith('curriculum/agent_')
                or any(t in k for t in ('overspeed', 'pace_deficit', 'approx_kl', 'post_tanh_entropy',
                                       'explained_variance', 'learning_rate', 'preupdate_max_abs_ratio_error',
                                       'max_abs_step_residual', 'update_skipped'))}
    evidence.update(source_update=source, safe_update=boundary, new_updates=boundary-source,
                    committed_metric_updates=len(metrics), window=[first, boundary],
                    window_episodes=len(episodes), outcomes={
                        key: sum(bool(row.get(key, False)) for row in episodes)
                        for key in ('success', 'collision', 'offroad', 'road_violation', 'timeout')},
                    coverage=dict(Counter(row['scenario_name'] for row in episodes)),
                    latest_behavior_and_optimizer=selected, integrity_issues=issues,
                    metrics_commit_pending=bool(boundary > source and (not metrics or metrics[-1]['update'] != boundary)),
                    interpretation='Training diagnostics, not frozen-policy independent evaluation.')
    # Never report an uncommitted tail row as the safe metric record.
    evidence['last_committed_metrics'] = window[-1] if window else None
    camera = run / 'session/camera.json'
    evidence['camera'] = json.loads(camera.read_text(encoding='utf-8')) if camera.exists() else {}
    return evidence


def save_snapshot(run):
    snapshot = inspect(run)
    session = Path(run) / 'session'
    atomic_write_json(snapshot, session / 'monitor_latest.json')
    with (session / 'monitor_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + '\n')
    return snapshot


def training_command(run, stop_update, scratch_plan=None):
    if scratch_plan:
        return [sys.executable, '-u', '-m', 'diagnostics.v6_scratch_session', 'train',
                '--plan', str(scratch_plan), '--stop-update', str(stop_update)]
    return [sys.executable, '-u', '-m', 'training.reward_fork_v2', '--run-dir', str(run),
            '--stop-update', str(stop_update)]


def supervise(args):
    run = Path(args.run_dir).resolve()
    session = run / 'session'
    session.mkdir(exist_ok=True)
    with RunSessionOwner(run):
        previous = session / 'launch.json'
        if previous.exists():
            record = json.loads(previous.read_text(encoding='utf-8'))
            for key in ('trainer_pid', 'supervisor_pid'):
                if record.get(key) and record[key] != os.getpid() and pid_alive(record[key]):
                    raise RuntimeError(f"previous {key} still alive; refusing duplicate launch")
        if (session / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED exists; inspect and archive it before explicit resume')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        log = session / f'training_{stamp}.log'
        command = training_command(run, args.stop_update, getattr(args, 'scratch_plan', None))
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        with log.open('x', encoding='utf-8') as stream:
            trainer = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=environment)
            atomic_write_json({'launch_id': stamp, 'trainer_pid': trainer.pid, 'supervisor_pid': os.getpid(),
                               'command': command, 'log_path': str(log), 'started_at': now(),
                               'stop_update': args.stop_update, 'status': 'DISPATCHED_VERIFY_CHECKPOINT',
                               'monitor_interval_seconds': 1200, 'automatic_restart': False}, previous)
            camera_log = session / f'camera_{stamp}.log'
            with camera_log.open('x', encoding='utf-8') as camera_stream:
                camera = subprocess.Popen([sys.executable, '-u', '-m', 'utils.follow_reward_fork',
                                           '--trainer-pid', str(trainer.pid), '--run-dir', str(run)],
                                          cwd=ROOT, stdout=camera_stream, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=environment)
                while trainer.poll() is None:
                    try:
                        save_snapshot(run)
                    except (OSError, ValueError, KeyError) as error:
                        atomic_write_json({'at': now(), 'error': str(error)}, session / 'monitor_error.json')
                    deadline = time.monotonic() + 1200
                    while trainer.poll() is None and time.monotonic() < deadline:
                        time.sleep(min(2, max(.01, deadline-time.monotonic())))
                try:
                    camera.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    # Camera is a read-only helper; report its PID, never kill/restart implicitly.
                    pass
            atomic_write_json({'launch_id': stamp, 'trainer_pid': trainer.pid, 'supervisor_pid': os.getpid(),
                               'returncode': trainer.returncode, 'camera_pid': camera.pid,
                               'camera_returncode': camera.poll(), 'ended_at': now(),
                               'status': 'EXITED', 'automatic_restart': False}, session / 'exit.json')
            save_snapshot(run)
            return trainer.returncode


def launch(args):
    run = Path(args.run_dir).resolve()
    session = run / 'session'
    session.mkdir(exist_ok=True)
    stamp = uuid.uuid4().hex
    log = session / f'supervisor_{stamp}.log'
    command = [sys.executable, '-u', '-m', 'utils.reward_fork_session', 'supervise',
               '--run-dir', str(run), '--stop-update', str(args.stop_update)]
    if getattr(args, 'scratch_plan', None):
        command += ['--scratch-plan', str(args.scratch_plan)]
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=os.name != 'nt',
                                 env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
    record = {'status': 'SUPERVISOR_DISPATCHED', 'supervisor_pid': child.pid, 'command': command,
              'log_path': str(log), 'at': now(), 'stop_update': args.stop_update}
    atomic_write_json(record, session / f'dispatch_{stamp}.json')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'supervise', 'inspect'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stop-update', type=int, default=476)
    parser.add_argument('--scratch-plan', help='Independent fresh V6 plan; never a u456 fork')
    args = parser.parse_args()
    if args.mode == 'supervise':
        raise SystemExit(supervise(args))
    result = launch(args) if args.mode == 'launch' else inspect(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
