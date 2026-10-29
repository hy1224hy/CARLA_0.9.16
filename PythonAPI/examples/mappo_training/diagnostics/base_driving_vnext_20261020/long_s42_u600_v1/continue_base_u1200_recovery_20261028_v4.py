"""Human-authorized simulator-crash recovery1051->1200; no automatic restart."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
CONTROL = ROOT / 'diagnostics/base_driving_vnext_20261020/long_s42_u600_v1'
PLAN = CONTROL / 'plan.json'
EXTENSION = CONTROL / 'extension_u1200_recovery_v1.json'
EXTENSION_SHA = '4bd640c454fc393dcd9425e984cede232a98928728feda8b1bd4aecdfe62e189'
PRIOR_RECORD = CONTROL / 'recovery_completed_eval_u1000_20261027_v3.json'
PRIOR_RECORD_SHA = '0a516b33f012056a9fdcfa0bfb27af30aa04d672d9786e1e16622b1ba34e5ecd'
RECORD = CONTROL / 'recovery_u1051_to_u1200_20261028_v4.json'
SOURCE = CONTROL / 'u001051_recovery_20261028_v4.pt'
SOURCE_SHA = '75600173d4d2287e4fb7240d380ac7c0f91839b3c4be06f1450562bf77844eba'
FINGERPRINT = '8d35d03c4a09111331b00ed308fa08f04a659070b2205e4ff61e6749fb865700'
OLD_UUID = '106f3ce262a24ecbb3b812e9ba606a70'
VERSION = 'base_driving_u1051_crash_recovery_u1200_20261028_v4'
COMPLETED = [50, 100, 200, 400, 600, 800, 1000]
REMAINING_EPISODES = 2384


def recovery_actions():
    return [('train', 1051, 1200), ('screen', 1200, 1200)]


def validate_source(info):
    if (info['safe_update'] != 1051 or info['episodes'] != 16816 or info['sha256'] != SOURCE_SHA
            or info['actor_adam_steps'] != {'0': [5255], '1': [5255]} or info['critic_adam_steps'] != [5255]
            or info['value_normalizer_update_count'] != 1051
            or set(info['rng_components']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
            or info['lr_schedule']['decay_horizon'] != 600 or info['source_fingerprint'] != FINGERPRINT):
        raise ValueError('Complete recovery source must remain exactly safe1051 with full original state')


def validate_interrupted_session(launch, exited):
    if (launch.get('launch_id') != OLD_UUID or launch.get('actual_worker_pid') != 17916
            or launch.get('resume_update') != 1000 or launch.get('stop_update') != 1200 or exited is not None):
        raise ValueError('Interrupted identity/exit changed; preserve facts and require a new review')


def runtime(require_record=True):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('original_u1200_adapter', CONTROL / 'continue_base_u1200_recovery.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    record = previous.effective_record()
    if (previous.sha256_file(EXTENSION) != EXTENSION_SHA or previous.sha256_file(PRIOR_RECORD) != PRIOR_RECORD_SHA
            or record['total_budget_updates'] != 1200 or record['total_budget_episodes'] != 19200):
        raise ValueError('Prior explicit budget/recovery binding changed')
    prior = previous.base.read(PRIOR_RECORD)
    if previous.sha256_file(prior['controller_source']) != prior['controller_source_sha256']:
        raise ValueError('Prior controller source changed')
    if require_record:
        auth = previous.base.read(RECORD)
        if (auth['version'] != VERSION or auth['source_checkpoint_sha256'] != SOURCE_SHA
                or auth['source_update'] != 1051 or auth['total_budget_updates'] != 1200
                or auth['total_budget_episodes'] != 19200 or auth['extension_sha256'] != EXTENSION_SHA
                or auth['prior_recovery_record_sha256'] != PRIOR_RECORD_SHA
                or auth['controller_source_sha256'] != previous.sha256_file(__file__)
                or previous.sha256_file(SOURCE) != SOURCE_SHA):
            raise ValueError('Current immutable recovery authorization/source changed')
    return previous, previous.base, record


def prepare():
    if RECORD.exists(): raise FileExistsError('Recovery already prepared; no implicit retry')
    previous, base, record = runtime(False)
    run = Path(record['run_dir'])
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        info = previous.checkpoint_summary(record)
        validate_source(info)
        launch = base.read(run / 'session/launch.json')
        exited = base.read(run / 'session/exit.json') if (run / 'session/exit.json').exists() else None
        validate_interrupted_session(launch, exited)
        state = base.read(CONTROL / 'status.json')
        if (state['version'] != 'base_driving_completed_u1000_evaluation_resume_u1200_v3'
                or state['status'] != 'TRAINING' or state['target_update'] != 1200):
            raise ValueError('Not the human-reported interrupted current training segment')
        world = previous.preflight(record)
        if base._completed_screens(record) != COMPLETED:
            raise ValueError('Completed frozen milestones changed; do not rerun or skip them')
        summaries = {str(stop): previous.sha256_file(CONTROL / f'u{stop:06d}_summary.json') for stop in COMPLETED}
        if SOURCE.exists():
            if previous.sha256_file(SOURCE) != SOURCE_SHA: raise ValueError('Fixed recovery source collision')
        else: previous.atomic_copy_checkpoint(run / 'latest.pt', SOURCE)
        tails = [r for r in base.rows(run / 'episodes.jsonl') if r['update'] > 1051]
        payload = {'version': VERSION, 'prepared_at': base.now(),
            'authorization': 'On2026-10-28 human reported CARLA crash, restarted CARLA, and explicitly requested resuming the same training and monitor.',
            'source_update': 1051, 'source_episodes': 16816, 'source_checkpoint': str(SOURCE),
            'source_checkpoint_sha256': SOURCE_SHA, 'complete_source_state': info,
            'remaining_updates': 149, 'remaining_episodes': REMAINING_EPISODES,
            'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'budget_extension': str(EXTENSION), 'extension_sha256': EXTENSION_SHA,
            'prior_recovery_record': str(PRIOR_RECORD), 'prior_recovery_record_sha256': PRIOR_RECORD_SHA,
            'controller_source': str(Path(__file__).resolve()), 'controller_source_sha256': previous.sha256_file(__file__),
            'prior_status': state, 'prior_status_sha256': previous.sha256_file(CONTROL / 'status.json'),
            'interrupted_launch': launch, 'interrupted_launch_sha256': previous.sha256_file(run / 'session/launch.json'),
            'interrupted_exit_present': False, 'uncommitted_episode_count': len(tails),
            'completed_stages': COMPLETED, 'completed_summary_sha256': summaries,
            'initialization_sha256': previous.sha256_file(run / 'initialization.json'),
            'live_preflight': world, 'actions': recovery_actions(), 'automatic_restart': False,
            'tail_policy': 'Archive original interruption including missing exit; native recorder retains all16816 committed episodes and reconciles only uncommitted tails; redo1052 fully.',
            'schedule_policy': 'Preserve original600 cumulative LR/entropy horizon and terminal values; no reset/stretch.'}
        previous.atomic_write_json(payload, RECORD)
        print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'source': info, 'preflight': world,
                          'record': str(RECORD), 'remaining_training_episodes': REMAINING_EPISODES,
                          'uncommitted_episode_count': len(tails)}), flush=True)


def snapshot(previous, base, record):
    run = Path(record['run_dir'])
    data = previous.inspect_files(run)
    data.update(protocol=VERSION, inspection_mode='FILES_ONLY', authorized_extension=str(EXTENSION),
        authorized_recovery=str(RECORD), automatic_restart=False, current_control=base.read(CONTROL / 'status.json'),
        safe_update=base.read(run / 'checkpoint_manifest.json')['latest']['update'])
    previous.atomic_write_json(data, run / 'session/monitor_latest.json')
    with (run / 'session/monitor_history.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(data, allow_nan=False) + '\n')


def supervise(previous, base, record):
    run = Path(record['run_dir'])
    session = run / 'session'
    if (session / 'launch.json').exists() or (session / 'exit.json').exists():
        raise ValueError('Raw prior session must be archived before dispatch')
    stamp = uuid.uuid4().hex
    log = session / f'training_{stamp}.log'
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'worker', '--from-update', '1051',
               '--to-update', '1200', '--launch-id', stamp]
    launch = {'launch_id': stamp, 'spawned_pid': None, 'actual_worker_pid': None, 'trainer_pid': None,
        'supervisor_pid': os.getpid(), 'command': command, 'log_path': str(log), 'started_at': base.now(),
        'resume_update': 1051, 'stop_update': 1200, 'status': 'DISPATCHED_NOT_TRAINING_SUCCESS',
        'automatic_restart': False, 'extension_record': str(EXTENSION), 'recovery_record': str(RECORD)}
    previous.atomic_write_json(launch, session / 'launch.json')
    camera = None
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with log.open('x', encoding='utf-8') as out, (session / f'camera_{stamp}.log').open('x', encoding='utf-8') as camout:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        launch['spawned_pid'] = child.pid
        previous.atomic_write_json(launch, session / 'launch.json')
        deadline = 0
        while child.poll() is None:
            actual = base.read(session / 'launch.json').get('actual_worker_pid')
            if actual and camera is None:
                camera = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'utils.follow_reward_fork',
                    '--trainer-pid', str(actual), '--run-dir', str(run)], cwd=ROOT, stdout=camout,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, creationflags=flags)
            if time.monotonic() >= deadline:
                try: snapshot(previous, base, record)
                except Exception as error:
                    previous.atomic_write_json({'at': base.now(), 'error': f'{type(error).__name__}: {error}'}, session / 'monitor_error.json')
                deadline = time.monotonic() + 1200
            time.sleep(2)
        if camera is not None:
            try: camera.wait(timeout=8)
            except subprocess.TimeoutExpired: pass
        actual = base.read(session / 'launch.json').get('actual_worker_pid')
        previous.atomic_write_json({'launch_id': stamp, 'spawned_pid': child.pid, 'trainer_pid': actual,
            'supervisor_pid': os.getpid(), 'returncode': child.returncode, 'ended_at': base.now(),
            'camera_pid': camera.pid if camera else None, 'camera_returncode': camera.poll() if camera else None,
            'automatic_restart': False}, session / 'exit.json')
    if child.returncode or not actual or not base._matching_exit(run, stop=1200):
        raise RuntimeError('Training failed or lacks matching actual worker exit0; no automatic retry')
    base.safe_boundary(record, expected=1200)


def controller():
    previous, base, record = runtime()
    run = Path(record['run_dir'])
    authorization = base.read(RECORD)
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        validate_source(previous.checkpoint_summary(record))
        validate_interrupted_session(base.read(run / 'session/launch.json'),
            base.read(run / 'session/exit.json') if (run / 'session/exit.json').exists() else None)
        if (previous.sha256_file(CONTROL / 'status.json') != authorization['prior_status_sha256']
                or previous.sha256_file(run / 'session/launch.json') != authorization['interrupted_launch_sha256']
                or previous.sha256_file(run / 'initialization.json') != authorization['initialization_sha256']):
            raise ValueError('Prior state changed after explicit recovery preparation')
        for stop, digest in authorization['completed_summary_sha256'].items():
            if previous.sha256_file(CONTROL / f'u{int(stop):06d}_summary.json') != digest:
                raise ValueError('Previously completed frozen summary changed')
        if (run / 'session/STOP_REQUESTED').exists(): raise ValueError('STOP_REQUESTED requires review')
        previous.preflight(record)
        state = {'version': VERSION, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': base.now(),
            'initial_safe_update': 1051, 'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'extension_sha256': EXTENSION_SHA, 'recovery_record': str(RECORD),
            'recovery_record_sha256': previous.sha256_file(RECORD), 'completed_stages': list(COMPLETED),
            'automatic_restart': False, 'explicit_interruption_recovery': True}
        previous.atomic_write_json(state, CONTROL / 'status.json')
        try:
            state['interrupted_session_archive'] = base.archive_session(record, reason='interrupted')
            state.update(status='TRAINING', target_update=1200, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            supervise(previous, base, record)
            state.update(status='FROZEN_EVALUATION', target_update=1200, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            result = base.frozen_screen(record, 1200)
            state['completed_stages'].append(1200)
            state['last_stage_selection'] = result['selection']
            state.update(status='COMPLETED', ended_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            snapshot(previous, base, record)
        except BaseException as error:
            state.update(status='FAILED_NO_RESTART', error=f'{type(error).__name__}: {error}', ended_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--from-update', type=int)
    parser.add_argument('--to-update', type=int)
    parser.add_argument('--launch-id')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'run': controller()
    else:
        if (args.from_update, args.to_update) != (1051, 1200) or not args.launch_id:
            raise ValueError('Worker cannot exceed explicit1051->1200 recovery segment')
        previous, base, record = runtime()
        with previous.RunSessionOwner(Path(record['run_dir']) / 'worker_guard'):
            base._worker_owned(PLAN, record, 1051, 1200, args.launch_id)
