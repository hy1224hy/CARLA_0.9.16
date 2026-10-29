"""Bounded full-state V6 continuation with serial frozen milestone screens.

Only the train subcommand and the existing screen evaluator use CARLA. Plans
pin their completed predecessor and all frozen scientific sources. No retries.
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

from diagnostics.resume_v6_scratch_final import load as completed_plan
from diagnostics.resume_v6_scratch import frozen_screen, now
from diagnostics.v6_scratch_session import ROOT, read, archive_session, training_integrity
from training.checkpoint_protocol import (
    atomic_copy_checkpoint, atomic_write_json, files_fingerprint,
    load_checkpoint_file, sha256_file, validate_checkpoint_payload,
)
from utils.mut_mappo_session import CarlaOwner, RunSessionOwner
from utils.reward_fork_session import save_snapshot


def require_stages(source, stops):
    if (type(source) is not int or source != 50 or not stops
            or any(type(value) is not int for value in stops)
            or stops != sorted(set(stops)) or stops[0] <= source or stops[-1] > 600):
        raise ValueError('requires increasing milestones after u50 within the u600 schedule')


def require_boundary(original, checkpoint, expected, stop):
    metadata, config = checkpoint['run_metadata'], checkpoint['training_config']
    if (type(expected) is not int or type(stop) is not int or not 50 <= expected < stop <= 600
            or checkpoint['update'] != expected or checkpoint['total_episodes'] != expected*16
            or metadata.get('run_id') != Path(original['run_dir']).name
            or metadata.get('source_type') != 'fresh' or metadata.get('source_checkpoint')
            or 'reward_fork' in metadata or metadata.get('code_drift_override')
            or config.get('reward_version') != 'v6' or config.get('pace_profile') != 'instant_pace_v6'
            or checkpoint['lr_schedule']['decay_horizon'] != 600):
        raise ValueError('requires the same fresh V6 full-state run at its exact prior milestone')


def checkpoint_args(original, expected, stop):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    run = Path(original['run_dir'])
    checkpoint, digest = load_checkpoint_file(run/'latest.pt', map_location='cpu')
    latest = read(run/'checkpoint_manifest.json')['latest']
    require_boundary(original, checkpoint, expected, stop)
    if latest['update'] != expected or latest['sha256'] != digest:
        raise RuntimeError('latest checkpoint/manifest mismatch')
    config = original['training_args']
    if config.get('pretrained_checkpoint') or config.get('allow_resume_code_drift'):
        raise ValueError('no weight transfer or source-drift override')
    args = trainer.build_parser().parse_args([])
    for key, value in config.items():
        if key.startswith('_') or not hasattr(args, key):
            raise ValueError(f'unknown frozen argument {key}')
        setattr(args, key, value)
    args.resume, args.pretrained_checkpoint = 'latest', ''
    args.save_dir, args.run_id, args.updates = str(run), run.name, stop
    args.follow_spectator, args.allow_resume_code_drift = True, False
    args.stop_request_file = str(run/'session/STOP_REQUESTED')
    args._explicit_cli_options = {'--'+key.replace('_', '-') for key in config}
    trainer.apply_resume_configuration(args, checkpoint)
    if checkpoint['run_metadata']['code_fingerprint'] != files_fingerprint(trainer.training_source_paths(ROOT)):
        raise RuntimeError('frozen training source identity changed')
    validate_checkpoint_payload(checkpoint, expected_protocol_versions=checkpoint['protocol_versions'],
        expected_action_spec=checkpoint['action_spec'], expected_model_spec=checkpoint['model_spec'],
        expected_training_config=trainer.checkpoint_training_config(args), expected_critic_spec=checkpoint['critic_spec'])
    if Path(args.stop_request_file).exists():
        raise RuntimeError('STOP_REQUESTED exists; no implicit override')
    training_integrity(run, expected)
    return checkpoint, digest, args


def load(path):
    record = read(path)
    predecessor, _, original = completed_plan(record['previous_plan'])
    require_stages(record['source_update'], record['stop_updates'])
    if (record['version'] != 'v6_long_full_state_v1'
            or record['adapter_sha256'] != sha256_file(__file__)
            or record['previous_plan_sha256'] != sha256_file(record['previous_plan'])
            or record['source_summary_sha256'] != sha256_file(record['source_summary'])
            or record['run_dir'] != original['run_dir']
            or record['total_budget_episodes'] != record['stop_updates'][-1]*16
            or record['source_sha256'] != read(record['source_summary'])['sha256']):
        raise RuntimeError('long-run plan/predecessor/source identity changed')
    return record, predecessor, original


def expected_digest(record, start, stop):
    require_stages(record['source_update'], record['stop_updates'])
    pairs = list(zip([record['source_update']]+record['stop_updates'][:-1], record['stop_updates']))
    if (start, stop) not in pairs:
        raise ValueError('stage is outside the authorized milestone sequence')
    if start == record['source_update']:
        return record['source_sha256']
    summary = read(Path(record['control_dir'])/f'u{start:06d}_summary.json')
    if summary['update'] != start or summary['aggregate']['valid_case_count'] != 12:
        raise RuntimeError('previous frozen milestone incomplete')
    if sha256_file(summary['checkpoint']) != summary['sha256']:
        raise RuntimeError('previous frozen checkpoint changed')
    return summary['sha256']


def prepare(previous, control, stops):
    require_stages(50, stops)
    predecessor, _, original = completed_plan(previous)
    previous_control = Path(predecessor['control_dir'])
    status = read(previous_control/'status.json')
    summary_path = previous_control/'u000050_summary.json'
    summary = read(summary_path)
    if status['status'] != 'COMPLETED' or summary['update'] != 50 or summary['aggregate']['valid_case_count'] != 12:
        raise RuntimeError('u50 pilot must have completed training and frozen evaluation')
    checkpoint, digest, _ = checkpoint_args(original, 50, stops[0])
    if digest != summary['sha256']:
        raise RuntimeError('latest is not the evaluated u50 checkpoint')
    launch = read(Path(original['run_dir'])/'session/launch.json')
    exited = read(Path(original['run_dir'])/'session/exit.json')
    if not launch.get('launch_id') or launch['launch_id'] != exited.get('launch_id') or exited['returncode'] != 0:
        raise RuntimeError('u50 training lacks a matching successful session exit')
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    record = {'version': 'v6_long_full_state_v1', 'authorized_on': '2026-10-18 Asia/Shanghai',
        'human_budget': 'continue unchanged V6 from complete u50 to cumulative u600',
        'previous_plan': str(Path(previous).resolve()), 'previous_plan_sha256': sha256_file(previous),
        'source_summary': str(summary_path), 'source_summary_sha256': sha256_file(summary_path),
        'run_dir': original['run_dir'], 'control_dir': str(control), 'source_update': 50,
        'source_sha256': digest, 'stop_updates': stops, 'total_budget_episodes': stops[-1]*16,
        'new_updates': stops[-1]-50, 'new_episodes': (stops[-1]-50)*16,
        'evaluation_seeds': original['evaluation_seeds'], 'adapter_sha256': sha256_file(__file__),
        'source_resume_count': checkpoint['run_metadata']['resume_count'], 'automatic_restart': False,
        'schedule_horizon': 600, 'scientific_configuration': 'unchanged original 40 sources, 12 scenes, V6 and 20/2'}
    atomic_copy_checkpoint(Path(original['run_dir'])/'latest.pt', control/'source_u000050.pt')
    shutil.copy2(summary_path, control/'reference_u000050_summary.json')
    atomic_write_json(training_integrity(Path(original['run_dir']), 50), control/'source_integrity.json')
    atomic_write_json(checkpoint['curriculum_state'], control/'source_curriculum_state.json')
    atomic_write_json(record, control/'plan.json')
    print(json.dumps({'status': 'LONG_RUN_PREPARED_NOT_TRAINED', **record}, indent=2))


def train(path, start, stop):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    record, _, original = load(path)
    expected = expected_digest(record, start, stop)
    checkpoint, digest, args = checkpoint_args(original, start, stop)
    if digest != expected:
        raise RuntimeError('exact milestone resume checkpoint changed')
    print(json.dumps({'status': 'STRICT_FULL_STATE_RESUME', 'resume_from': start, 'next_update': start+1,
        'stop_update': stop, 'total_budget_updates': record['stop_updates'][-1],
        'source_sha256': digest, 'lr_horizon': args.lr_decay_horizon,
        'prior_resume_count': checkpoint['run_metadata']['resume_count']}, indent=2), flush=True)
    with CarlaOwner(run_dir=original['run_dir'], purpose='authorized V6 long full-state continuation'):
        _preflight_idle_world(host='127.0.0.1', port=2000, expected_version='0.9.16')
        trainer.train(args)
    load(path)


def supervise(path, original, start, stop):
    run = Path(original['run_dir'])
    session = run/'session'
    stamp = uuid.uuid4().hex
    log = session/f'training_{stamp}.log'
    cmd = [sys.executable, '-u', '-m', 'diagnostics.v6_long_session', 'train', '--plan', str(path),
           '--from-update', str(start), '--stop-update', str(stop)]
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    with log.open('x', encoding='utf-8') as output:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=environment)
        atomic_write_json({'launch_id': stamp, 'trainer_pid': child.pid, 'supervisor_pid': os.getpid(),
            'command': cmd, 'log_path': str(log), 'started_at': now(), 'stop_update': stop,
            'resume_update': start, 'total_budget_updates': read(path)['stop_updates'][-1],
            'status': 'DISPATCHED_VERIFY_CHECKPOINT', 'monitor_interval_seconds': 1200,
            'automatic_restart': False}, session/'launch.json')
        with (session/f'camera_{stamp}.log').open('x', encoding='utf-8') as camera_output:
            camera = subprocess.Popen([sys.executable, '-u', '-m', 'utils.follow_reward_fork',
                '--trainer-pid', str(child.pid), '--run-dir', str(run)], cwd=ROOT,
                stdout=camera_output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=environment)
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
    record, predecessor, original = load(path)
    control, run = Path(record['control_dir']), Path(original['run_dir'])
    with ExitStack() as owners:
        for directory in (control, Path(original['control_dir']), Path(predecessor['control_dir']), run):
            owners.enter_context(RunSessionOwner(directory))
        if (control/'status.json').exists():
            raise FileExistsError('already dispatched; explicit review required')
        state = {**record, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': now(), 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            for start, stop in zip([50]+record['stop_updates'][:-1], record['stop_updates']):
                load(path)
                _, digest, _ = checkpoint_args(original, start, stop)
                if digest != expected_digest(record, start, stop):
                    raise RuntimeError('milestone boundary changed')
                archive_session(run, control)
                state.update(status='TRAINING_DISPATCHED', resume_from_update=start,
                             target_update=stop, changed_at=now())
                atomic_write_json(state, control/'status.json')
                supervise(path, original, start, stop)
                load(path)
                state.update(status='FROZEN_EVALUATION', changed_at=now())
                atomic_write_json(state, control/'status.json')
                state['completed_stages'].append(frozen_screen(original, control, stop))
                state.update(changed_at=now())
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
    parser.add_argument('--stops', type=int, nargs='+', default=[100, 200, 400, 600])
    parser.add_argument('--from-update', type=int)
    parser.add_argument('--stop-update', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.previous_plan, args.control_dir, args.stops)
    elif args.mode == 'train':
        train(args.plan, args.from_update, args.stop_update)
    elif args.mode == 'validate':
        record, _, original = load(args.plan)
        _, digest, _ = checkpoint_args(original, 50, record['stop_updates'][0])
        if digest != record['source_sha256']:
            raise RuntimeError('pinned source changed')
        print(json.dumps({'status': 'LONG_RUN_FULL_STATE_VALIDATED', 'source_update': 50,
                          'stop_updates': record['stop_updates'], 'sha256': digest}))
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
