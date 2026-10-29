"""Explicit recovery after u25 training succeeded but simulator preflight failed.

Reuses the reviewed recovery adapter unchanged. First redo the zero-case frozen
screen, then strictly resume u25->50 and screen u50. No training is repeated.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from diagnostics.resume_v6_scratch import (
    recovery_plan, validate_checkpoint, frozen_screen, train_segment, now,
)
from diagnostics.v6_scratch_session import ROOT, read, archive_session
from training.checkpoint_protocol import atomic_write_json, atomic_copy_checkpoint, sha256_file
from utils.mut_mappo_session import RunSessionOwner


def require_finished_u25(checkpoint, launch, exited):
    if (checkpoint['update'] != 25 or not launch.get('launch_id')
            or launch['launch_id'] != exited.get('launch_id') or exited.get('returncode') != 0):
        raise ValueError('requires complete u25 and a successful matching training exit')


def load(path):
    current = read(path)
    _, original = recovery_plan(current['previous_plan'])
    if (current['version'] != 'v6_scratch_screen25_recovery_v1'
            or current['adapter_sha256'] != sha256_file(__file__)
            or current['previous_plan_sha256'] != sha256_file(current['previous_plan'])
            or current['stop_updates'] != [25, 50]):
        raise RuntimeError('screen recovery plan/source changed')
    return current, original


def prepare(previous_path, control):
    previous, original = recovery_plan(previous_path)
    checkpoint, digest = validate_checkpoint(original, 50)
    run = Path(original['run_dir'])
    require_finished_u25(checkpoint, read(run/'session/launch.json'), read(run/'session/exit.json'))
    failure = read(Path(previous['control_dir'])/'screen_u000025/manifest.json')
    if failure['status'] != 'FAILED' or failure['completed_cases'] != 0:
        raise RuntimeError('expected reviewed failed preflight with zero evaluated cases')
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    record = {'version': 'v6_scratch_screen25_recovery_v1', 'authorized_on': '2026-10-17 Asia/Shanghai',
        'previous_plan': str(Path(previous_path).resolve()), 'previous_plan_sha256': sha256_file(previous_path),
        'control_dir': str(control), 'run_dir': original['run_dir'], 'resume_update': 25,
        'resume_sha256': digest, 'stop_updates': [25, 50], 'remaining_training_updates': 25,
        'total_training_budget_updates': 50, 'total_training_budget_episodes': 800,
        'evaluation_seeds': original['evaluation_seeds'], 'adapter_sha256': sha256_file(__file__),
        'automatic_restart': False, 'first_action': 'repeat u25 frozen screen after failed connection; no training rerun'}
    atomic_write_json(record, control/'plan.json')
    atomic_copy_checkpoint(run/'latest.pt', control/'source_u000025.pt')
    print(json.dumps({'status': 'SCREEN_RECOVERY_PREPARED_NOT_EVALUATED', **record}, indent=2))


def validate(path):
    current, original = load(path)
    run = Path(original['run_dir'])
    checkpoint, digest = validate_checkpoint(original, 50)
    require_finished_u25(checkpoint, read(run/'session/launch.json'), read(run/'session/exit.json'))
    if digest != current['resume_sha256']:
        raise RuntimeError('u25 boundary changed')
    return current, original


def controller(path):
    current, original = validate(path)
    previous = read(current['previous_plan'])
    control, run = Path(current['control_dir']), Path(original['run_dir'])
    with (RunSessionOwner(control), RunSessionOwner(Path(original['control_dir'])),
          RunSessionOwner(Path(previous['control_dir'])), RunSessionOwner(run)):
        if (control/'status.json').exists():
            raise FileExistsError('one-shot screen recovery was already dispatched')
        state = {**current, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': now(), 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            archived = control/'previous_failed_control'
            archived.mkdir(exist_ok=False)
            old_control = Path(previous['control_dir'])
            for name in ('status.json', 'controller.stdout.log', 'controller.stderr.log'):
                shutil.copy2(old_control/name, archived/name)
            shutil.copy2(old_control/'screen_u000025/manifest.json', archived/'failed_screen_manifest.json')
            # The old FAILED record remains intact; link its explicit replacement.
            atomic_write_json({'superseded_at': now(), 'new_control_dir': str(control)}, old_control/'superseded_by.json')
            state.update(status='FROZEN_EVALUATION', target_update=25, changed_at=now())
            atomic_write_json(state, control/'status.json')
            state['completed_stages'].append(frozen_screen(original, control, 25))
            atomic_write_json(state, control/'status.json')
            load(path)
            checkpoint, _ = validate_checkpoint(original, 50)
            archive_session(run, control)
            state.update(status='TRAINING_DISPATCHED', target_update=50, resume_from_update=25, changed_at=now())
            atomic_write_json(state, control/'status.json')
            train_segment(original, checkpoint, 50)
            load(path)
            state.update(status='FROZEN_EVALUATION', target_update=50, changed_at=now())
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
    parser.add_argument('mode', choices=['prepare', 'validate', 'run'])
    parser.add_argument('--previous-plan')
    parser.add_argument('--control-dir')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.previous_plan, args.control_dir)
    elif args.mode == 'validate':
        current, _ = validate(args.plan)
        print(json.dumps({'status': 'U25_SCREEN_RECOVERY_VALIDATED', 'sha256': current['resume_sha256']}))
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
