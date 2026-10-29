"""Explicit pinned recovery of the existing V6 long run, without budget reset.

Reuse full-state validation and frozen screens; preserve every earlier plan.
The same adapter can prepare a later explicitly requested recovery as well.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from diagnostics.v6_long_session import load as load_long, checkpoint_args
from diagnostics.resume_v6_scratch import frozen_screen, now
from diagnostics.v6_scratch_session import ROOT, read, archive_session, training_integrity
from training.checkpoint_protocol import atomic_write_json, atomic_copy_checkpoint, sha256_file
from utils.mut_mappo_session import RunSessionOwner, CarlaOwner
from utils.reward_fork_session import save_snapshot, rows


def recovery_stops(source, original_stops, completed):
    if (type(source) is not int or not 50 <= source < 600
            or original_stops != [100, 200, 400, 600]
            or sorted(completed) != [x for x in original_stops if x <= source]):
        raise ValueError('requires the latest safe boundary and all earlier frozen milestones within u600')
    return [x for x in original_stops if x > source]


def parent_plan(path):
    record = read(path)
    if record['version'] == 'v6_long_full_state_v1':
        parent, _, original = load_long(path)
        return parent, original, str(Path(path).resolve())
    if record['version'] == 'v6_long_recovery_v1':
        parent, original = load(path)
        return parent, original, parent['long_plan']
    raise ValueError('unrecognized long-run recovery parent')


def verify_references(references):
    updates = []
    for reference in references:
        summary = read(reference['path'])
        if (sha256_file(reference['path']) != reference['sha256']
                or summary['update'] != reference['update']
                or summary['aggregate']['valid_case_count'] != 12
                or sha256_file(summary['checkpoint']) != summary['sha256']):
            raise RuntimeError('previous frozen milestone evidence changed')
        updates.append(summary['update'])
    return updates


def load(path):
    record = read(path)
    _, original, root_plan = parent_plan(record['previous_plan'])
    completed = verify_references(record['completed_references'])
    stops = recovery_stops(record['source_update'], [100, 200, 400, 600], completed)
    if (record['version'] != 'v6_long_recovery_v1' or record['stop_updates'] != stops
            or record['long_plan'] != root_plan
            or record['previous_plan_sha256'] != sha256_file(record['previous_plan'])
            or record['adapter_sha256'] != sha256_file(__file__)
            or record['run_dir'] != original['run_dir']
            or record['total_budget_updates'] != 600 or record['total_budget_episodes'] != 9600
            or sha256_file(Path(record['control_dir'])/f"source_u{record['source_update']:06d}.pt") != record['source_sha256']):
        raise RuntimeError('pinned recovery identity or budget changed')
    return record, original


def phase_digest(record, start, stop):
    pairs = list(zip([record['source_update']]+record['stop_updates'][:-1], record['stop_updates']))
    if (start, stop) not in pairs:
        raise ValueError('cannot skip an authorized frozen milestone')
    if start == record['source_update']:
        return record['source_sha256']
    summary_path = Path(record['control_dir'])/f'u{start:06d}_summary.json'
    summary = read(summary_path)
    verify_references([{'path': str(summary_path), 'sha256': sha256_file(summary_path), 'update': start}])
    return summary['sha256']


def prepare(previous, control):
    parent, original, root = parent_plan(previous)
    run = Path(original['run_dir'])
    expected = int(read(run/'checkpoint_manifest.json')['latest']['update'])
    references = list(parent.get('completed_references', []))
    state = read(Path(parent['control_dir'])/'status.json')
    if state['status'] in ('COMPLETED', 'SUPERSEDED_BY_EXPLICIT_RECOVERY'):
        raise RuntimeError('this predecessor is not the interrupted current control')
    for summary in state.get('completed_stages', []):
        path = Path(parent['control_dir'])/f"u{summary['update']:06d}_summary.json"
        references.append({'path': str(path), 'sha256': sha256_file(path), 'update': summary['update']})
    stops = recovery_stops(expected, [100, 200, 400, 600], verify_references(references))
    checkpoint, digest, _ = checkpoint_args(original, expected, stops[0])
    control = Path(control).resolve()
    control.relative_to((ROOT/'diagnostics').resolve())
    control.mkdir(parents=True, exist_ok=False)
    tail = [row for row in rows(run/'episodes.jsonl') if row['update'] > expected]
    record = {'version': 'v6_long_recovery_v1', 'authorized_on': '2026-10-19 Asia/Shanghai',
        'previous_plan': str(Path(previous).resolve()), 'previous_plan_sha256': sha256_file(previous),
        'long_plan': root, 'run_dir': original['run_dir'], 'control_dir': str(control),
        'source_update': expected, 'source_sha256': digest,
        'source_resume_count': checkpoint['run_metadata']['resume_count'],
        'stop_updates': stops, 'remaining_updates': 600-expected, 'remaining_episodes': (600-expected)*16,
        'total_budget_updates': 600, 'total_budget_episodes': 9600, 'schedule_horizon': 600,
        'completed_references': references, 'uncommitted_episode_rows_at_prepare': len(tail),
        'automatic_restart': False, 'adapter_sha256': sha256_file(__file__)}
    atomic_copy_checkpoint(run/'latest.pt', control/f'source_u{expected:06d}.pt')
    atomic_write_json(training_integrity(run, expected), control/'source_integrity.json')
    atomic_write_json(tail, control/'uncommitted_episodes_at_prepare.json')
    for reference in references:
        shutil.copy2(reference['path'], control/f"reference_u{reference['update']:06d}_summary.json")
    atomic_write_json(record, control/'plan.json')
    print(json.dumps({'status': 'RECOVERY_PREPARED_NOT_TRAINED', **record}, indent=2))


def archive_interrupted(record, original):
    run, control = Path(original['run_dir']), Path(record['control_dir'])
    previous = Path(read(record['previous_plan'])['control_dir'])
    archived = control/'interrupted_previous_session'
    archived.mkdir(exist_ok=False)
    for name in ('episodes.jsonl', 'metrics.jsonl', 'checkpoint_manifest.json'):
        shutil.copy2(run/name, archived/name)
    for name in ('launch.json', 'exit.json', 'camera.json', 'monitor_latest.json'):
        if (run/'session'/name).exists():
            shutil.copy2(run/'session'/name, archived/name)
    launch = read(run/'session/launch.json')
    shutil.copy2(launch['log_path'], archived/'interrupted_training.log')
    for name in ('status.json', 'controller.stdout.log', 'controller.stderr.log'):
        if (previous/name).exists():
            shutil.copy2(previous/name, archived/('previous_'+name))
    atomic_write_json({'at': now(), 'safe_update': record['source_update'],
        'old_exit_record_present': (run/'session/exit.json').exists(),
        'meaning': 'interruption evidence only; missing exit is not successful termination',
        'uncommitted_episode_rows': record['uncommitted_episode_rows_at_prepare'],
        'rollback': 'original trainer discards rows beyond the safe checkpoint and redoes that update'},
        archived/'interruption_review.json')
    for name in ('launch.json', 'exit.json'):
        source = run/'session'/name
        if source.exists():
            source.replace(archived/('original_'+name))
    old = read(previous/'status.json')
    old.update(status='SUPERSEDED_BY_EXPLICIT_RECOVERY', recovery_control=str(control), changed_at=now())
    atomic_write_json(old, previous/'status.json')


def train(path, start, stop):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    record, original = load(path)
    expected = phase_digest(record, start, stop)
    checkpoint, digest, args = checkpoint_args(original, start, stop)
    if digest != expected:
        raise RuntimeError('pinned complete resume boundary changed')
    print(json.dumps({'status': 'STRICT_FULL_STATE_RECOVERY', 'resume_from': start, 'next_update': start+1,
        'source_sha256': digest, 'stop_update': stop, 'total_budget_updates': 600,
        'prior_resume_count': checkpoint['run_metadata']['resume_count'], 'lr_horizon': args.lr_decay_horizon}), flush=True)
    with CarlaOwner(run_dir=original['run_dir'], purpose='explicit full-state V6 long recovery'):
        _preflight_idle_world(host='127.0.0.1', port=2000, expected_version='0.9.16')
        trainer.train(args)
    load(path)


def supervise(path, original, start, stop):
    run = Path(original['run_dir'])
    session, stamp = run/'session', uuid.uuid4().hex
    log = session/f'training_{stamp}.log'
    cmd = [sys.executable, '-u', '-m', 'diagnostics.v6_long_recovery', 'train', '--plan', str(path),
           '--from-update', str(start), '--stop-update', str(stop)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    with log.open('x', encoding='utf-8') as output:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env)
        atomic_write_json({'launch_id': stamp, 'trainer_pid': child.pid, 'supervisor_pid': os.getpid(),
            'command': cmd, 'log_path': str(log), 'started_at': now(), 'resume_update': start,
            'stop_update': stop, 'total_budget_updates': 600, 'status': 'DISPATCHED_VERIFY_CHECKPOINT',
            'monitor_interval_seconds': 1200, 'automatic_restart': False}, session/'launch.json')
        with (session/f'camera_{stamp}.log').open('x', encoding='utf-8') as camera_output:
            camera = subprocess.Popen([sys.executable, '-u', '-m', 'utils.follow_reward_fork',
                '--trainer-pid', str(child.pid), '--run-dir', str(run)], cwd=ROOT,
                stdout=camera_output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            deadline = 0
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    try:
                        save_snapshot(run)
                    except (OSError, ValueError, KeyError) as error:
                        atomic_write_json({'at': now(), 'error': str(error)}, session/'monitor_error.json')
                    deadline = time.monotonic()+1200
                time.sleep(2)
            try:
                camera.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        atomic_write_json({'launch_id': stamp, 'trainer_pid': child.pid, 'supervisor_pid': os.getpid(),
            'returncode': child.returncode, 'camera_pid': camera.pid, 'camera_returncode': camera.poll(),
            'status': 'EXITED', 'ended_at': now(), 'automatic_restart': False}, session/'exit.json')
        save_snapshot(run)
        if child.returncode:
            raise RuntimeError(f'trainer exited {child.returncode}; no automatic restart')


def controller(path):
    path = Path(path).resolve()
    record, original = load(path)
    control, run = Path(record['control_dir']), Path(original['run_dir'])
    directories = [control, run, Path(original['control_dir']),
                   Path(read(record['long_plan'])['control_dir']),
                   Path(read(record['previous_plan'])['control_dir'])]
    with ExitStack() as owners:
        for directory in dict.fromkeys(p.resolve() for p in directories):
            owners.enter_context(RunSessionOwner(directory))
        if (control/'status.json').exists():
            raise FileExistsError('recovery already dispatched; explicit review required')
        state = {**record, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': now(), 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            for index, (start, stop) in enumerate(zip([record['source_update']]+record['stop_updates'][:-1], record['stop_updates'])):
                load(path)
                _, digest, _ = checkpoint_args(original, start, stop)
                if digest != phase_digest(record, start, stop):
                    raise RuntimeError('safe boundary changed after preparation')
                if index == 0:
                    archive_interrupted(record, original)
                else:
                    archive_session(run, control)
                state.update(status='TRAINING_DISPATCHED', resume_from_update=start,
                             target_update=stop, changed_at=now())
                atomic_write_json(state, control/'status.json')
                supervise(path, original, start, stop)
                load(path)
                state.update(status='FROZEN_EVALUATION', changed_at=now())
                atomic_write_json(state, control/'status.json')
                state['completed_stages'].append(frozen_screen(original, control, stop))
                atomic_write_json(state, control/'status.json')
            state.update(status='COMPLETED', ended_at=now())
            atomic_write_json(state, control/'status.json')
        except BaseException as error:
            state.update(status='FAILED_NO_RESTART', error=f'{type(error).__name__}: {error}', ended_at=now())
            atomic_write_json(state, control/'status.json')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'validate', 'run', 'train'])
    parser.add_argument('--previous-plan')
    parser.add_argument('--control-dir')
    parser.add_argument('--plan')
    parser.add_argument('--from-update', type=int)
    parser.add_argument('--stop-update', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.previous_plan, args.control_dir)
    elif args.mode == 'train':
        train(args.plan, args.from_update, args.stop_update)
    elif args.mode == 'validate':
        record, original = load(args.plan)
        _, digest, _ = checkpoint_args(original, record['source_update'], record['stop_updates'][0])
        if digest != record['source_sha256']:
            raise RuntimeError('pinned checkpoint changed')
        print(json.dumps({'status': 'PINNED_FULL_RECOVERY_VALIDATED', 'source_update': record['source_update'],
                          'remaining_stops': record['stop_updates'], 'sha256': digest}))
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
