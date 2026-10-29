"""Explicit recovery from a pinned latest safe update in the final u25->u50 stage.

This reusable adapter preserves all older frozen sources and recovery records.
Only the checkpoint named in its new plan can be resumed; no automatic retry.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack

from diagnostics.resume_v6_scratch_screen25 import load as previous_plan
from diagnostics.resume_v6_scratch import frozen_screen, now
from diagnostics.v6_scratch_session import ROOT, read, training_integrity
from training.checkpoint_protocol import (
    atomic_write_json, atomic_copy_checkpoint, load_checkpoint_file,
    validate_checkpoint_payload, sha256_file, files_fingerprint,
)
from utils.mut_mappo_session import RunSessionOwner, CarlaOwner
from utils.reward_fork_session import save_snapshot


def require_boundary(plan, checkpoint, expected, stop):
    metadata, config = checkpoint['run_metadata'], checkpoint['training_config']
    if (isinstance(expected, bool) or not 25 <= expected < 50 or stop != 50
            or checkpoint['update'] != expected
            or metadata.get('run_id') != Path(plan['run_dir']).name
            or metadata.get('source_type') != 'fresh' or metadata.get('source_checkpoint')
            or 'reward_fork' in metadata or config.get('reward_version') != 'v6'
            or config.get('pace_profile') != 'instant_pace_v6'):
        raise ValueError('requires this fresh V6 run at the pinned latest boundary, ending at u50')


def checkpoint_args(plan, expected):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    run = Path(plan['run_dir'])
    checkpoint, digest = load_checkpoint_file(run/'latest.pt', map_location='cpu')
    latest = read(run/'checkpoint_manifest.json')['latest']
    require_boundary(plan, checkpoint, expected, 50)
    if checkpoint['update'] != latest['update'] or digest != latest['sha256']:
        raise RuntimeError('latest safe checkpoint/manifest mismatch')
    config = plan['training_args']
    if config.get('pretrained_checkpoint') or config.get('allow_resume_code_drift'):
        raise ValueError('pretrained transfer and source-drift override are forbidden')
    args = trainer.build_parser().parse_args([])
    for key, value in config.items():
        if not hasattr(args, key) or key.startswith('_'):
            raise ValueError(f'unknown frozen argument {key}')
        setattr(args, key, value)
    args.resume, args.pretrained_checkpoint = 'latest', ''
    args.updates, args.save_dir, args.run_id = 50, str(run), run.name
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
    previous, original = previous_plan(record['previous_plan'])
    if (record['version'] != 'v6_scratch_final_recovery_v1' or record['stop_update'] != 50
            or record['adapter_sha256'] != sha256_file(__file__)
            or record['previous_plan_sha256'] != sha256_file(record['previous_plan'])
            or record['prior_u25_summary_sha256'] != sha256_file(record['prior_u25_summary'])):
        raise RuntimeError('frozen recovery plan/source/evaluation identity changed')
    return record, previous, original


def prepare(previous, control):
    previous_record, original = previous_plan(previous)
    expected = int(read(Path(original['run_dir'])/'checkpoint_manifest.json')['latest']['update'])
    checkpoint, digest, _ = checkpoint_args(original, expected)
    prior = Path(previous_record['control_dir'])/'u000025_summary.json'
    summary = read(prior)
    if summary['update'] != 25 or summary['aggregate']['valid_case_count'] != 12:
        raise RuntimeError('the original u25 frozen stage has not been completed')
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    record = {'version': 'v6_scratch_final_recovery_v1', 'authorized_on': '2026-10-18 Asia/Shanghai',
        'previous_plan': str(Path(previous).resolve()), 'previous_plan_sha256': sha256_file(previous),
        'control_dir': str(control), 'run_dir': original['run_dir'], 'resume_update': expected,
        'resume_sha256': digest, 'stop_update': 50, 'remaining_updates': 50-expected,
        'remaining_episodes': (50-expected)*16, 'total_budget_episodes': 800,
        'prior_u25_summary': str(prior), 'prior_u25_summary_sha256': sha256_file(prior),
        'adapter_sha256': sha256_file(__file__), 'automatic_restart': False,
        'source_resume_count': checkpoint['run_metadata']['resume_count']}
    atomic_write_json(record, control/'plan.json')
    atomic_copy_checkpoint(Path(original['run_dir'])/'latest.pt', control/f'source_u{expected:06d}.pt')
    shutil.copy2(prior, control/'reference_u000025_summary.json')
    atomic_write_json(training_integrity(Path(original['run_dir']), expected), control/'source_integrity.json')
    print(json.dumps({'status': 'RECOVERY_PREPARED_NOT_TRAINED', **record}, indent=2))


def train(path):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    record, _, original = load(path)
    checkpoint, digest, args = checkpoint_args(original, record['resume_update'])
    if digest != record['resume_sha256']:
        raise RuntimeError('pinned resume checkpoint changed')
    print(json.dumps({'resume_from': checkpoint['update'], 'next_update': checkpoint['update']+1,
                      'total_episodes': checkpoint['total_episodes'], 'stop_update': 50,
                      'lr_horizon': args.lr_decay_horizon}), flush=True)
    with CarlaOwner(run_dir=original['run_dir'], purpose='explicit pinned final-stage V6 recovery'):
        _preflight_idle_world(host='127.0.0.1', port=2000, expected_version='0.9.16')
        trainer.train(args)
    load(path)


def supervise(path, original):
    run, session = Path(original['run_dir']), Path(original['run_dir'])/'session'
    stamp = uuid.uuid4().hex
    log = session/f'training_{stamp}.log'
    cmd = [sys.executable, '-u', '-m', 'diagnostics.resume_v6_scratch_final', 'train', '--plan', str(path)]
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    with log.open('x', encoding='utf-8') as output:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=environment)
        atomic_write_json({'launch_id': stamp, 'trainer_pid': child.pid, 'supervisor_pid': os.getpid(),
            'command': cmd, 'log_path': str(log), 'started_at': now(), 'stop_update': 50,
            'resume_update': read(path)['resume_update'], 'status': 'DISPATCHED_VERIFY_CHECKPOINT',
            'monitor_interval_seconds': 1200, 'automatic_restart': False}, session/'launch.json')
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
    record, previous, original = load(path)
    control, run = Path(record['control_dir']), Path(original['run_dir'])
    with ExitStack() as owners:
        for directory in (control, Path(original['control_dir']), Path(previous['control_dir']), run):
            owners.enter_context(RunSessionOwner(directory))
        if (control/'status.json').exists():
            raise FileExistsError('recovery already dispatched; explicit review required')
        state = {**record, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': now(), 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            _, digest, _ = checkpoint_args(original, record['resume_update'])
            if digest != record['resume_sha256']:
                raise RuntimeError('pinned resume boundary changed')
            archived = control/'interrupted_previous_session'
            archived.mkdir(exist_ok=False)
            for name in ('episodes.jsonl', 'metrics.jsonl', 'checkpoint_manifest.json'):
                shutil.copy2(run/name, archived/name)
            for name in ('launch.json', 'exit.json', 'camera.json', 'monitor_latest.json'):
                if (run/'session'/name).exists():
                    shutil.copy2(run/'session'/name, archived/name)
            previous_control = Path(previous['control_dir'])
            for name in ('status.json', 'controller.stdout.log', 'controller.stderr.log'):
                shutil.copy2(previous_control/name, archived/('previous_'+name))
            for name in ('launch.json', 'exit.json'):
                source = run/'session'/name
                if source.exists():
                    source.replace(archived/('original_'+name))
            old_state = read(previous_control/'status.json')
            old_state.update(status='SUPERSEDED_BY_EXPLICIT_RECOVERY', recovery_control=str(control), changed_at=now())
            atomic_write_json(old_state, previous_control/'status.json')
            state.update(status='TRAINING_DISPATCHED', target_update=50,
                         resume_from_update=record['resume_update'], changed_at=now())
            atomic_write_json(state, control/'status.json')
            supervise(path, original)
            load(path)
            state.update(status='FROZEN_EVALUATION', changed_at=now())
            atomic_write_json(state, control/'status.json')
            state['completed_stages'].append(frozen_screen(original, control, 50))
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
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.previous_plan, args.control_dir)
    elif args.mode == 'train':
        train(args.plan)
    elif args.mode == 'validate':
        record, _, original = load(args.plan)
        _, digest, _ = checkpoint_args(original, record['resume_update'])
        if digest != record['resume_sha256']:
            raise RuntimeError('pinned boundary changed')
        print(json.dumps({'status': 'PINNED_FINAL_RECOVERY_VALIDATED', 'resume_update': record['resume_update']}))
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
