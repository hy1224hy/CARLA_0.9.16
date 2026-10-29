"""Explicit u11 interruption recovery; leaves the original frozen sources intact.

Keeps the same run, strict full-state trainer, original configuration and u25/u50
screens. This one-shot adapter never retries, expands the budget, or starts CARLA.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from diagnostics.v6_scratch_session import (
    ROOT, load_plan, training_integrity, archive_session, read, command,
)
from training.checkpoint_protocol import (
    atomic_copy_checkpoint, atomic_write_json, load_checkpoint_file,
    sha256_file, validate_checkpoint_payload, files_fingerprint,
)
from utils.mut_mappo_session import RunSessionOwner, CarlaOwner
from utils.reward_fork_session import inspect, save_snapshot


def now():
    return datetime.now(timezone.utc).isoformat()


def require_boundary(plan, checkpoint, stop):
    expected = {25: 11, 50: 25}
    metadata = checkpoint['run_metadata']
    if (stop not in expected or isinstance(stop, bool) or checkpoint['update'] != expected[stop]
            or metadata.get('source_type') != 'fresh' or metadata.get('source_checkpoint')
            or 'reward_fork' in metadata or metadata.get('run_id') != Path(plan['run_dir']).name
            or checkpoint['training_config'].get('pace_profile') != 'instant_pace_v6'):
        raise ValueError('recovery requires this fresh V6 run at u11->25 or u25->50')


def resume_command(plan, checkpoint, stop):
    require_boundary(plan, checkpoint, stop)
    config = dict(plan['training_args'])
    if config.get('pretrained_checkpoint') or config.get('allow_resume_code_drift'):
        raise ValueError('recovery cannot load pretrained weights or override source identity')
    run = Path(plan['run_dir']).resolve()
    config.update(resume='latest', pretrained_checkpoint='', updates=stop,
                  save_dir=str(run), run_id=run.name, follow_spectator=True,
                  stop_request_file=str(run/'session/STOP_REQUESTED'))
    result = [sys.executable, '-u', '-m', 'training.train_no_rllib_mappo_robust_v2']
    for name, value in config.items():
        if isinstance(value, bool):
            if name == 'lr_decay':
                result.append('--lr-decay' if value else '--no-lr-decay')
            elif value:
                result.append('--'+name.replace('_', '-'))
        elif value is not None and value != '':
            result.extend(('--'+name.replace('_', '-'), str(value)))
    return result


def validate_checkpoint(plan, stop):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    run = Path(plan['run_dir'])
    checkpoint, digest = load_checkpoint_file(run/'latest.pt', map_location='cpu')
    latest = read(run/'checkpoint_manifest.json')['latest']
    require_boundary(plan, checkpoint, stop)
    if latest['sha256'] != digest or latest['update'] != checkpoint['update']:
        raise RuntimeError('manifest/checkpoint identity mismatch')
    if checkpoint['run_metadata']['code_fingerprint'] != files_fingerprint(trainer.training_source_paths(ROOT)):
        raise RuntimeError('training sources changed; strict resume required')
    args = trainer.build_parser().parse_args(resume_command(plan, checkpoint, stop)[4:])
    args._explicit_cli_options = {'--'+key.replace('_', '-') for key in plan['training_args']}
    trainer.apply_resume_configuration(args, checkpoint)
    validate_checkpoint_payload(checkpoint,
        expected_protocol_versions=checkpoint['protocol_versions'], expected_action_spec=checkpoint['action_spec'],
        expected_model_spec=checkpoint['model_spec'], expected_training_config=trainer.checkpoint_training_config(args),
        expected_critic_spec=checkpoint['critic_spec'])
    if (run/'session/STOP_REQUESTED').exists():
        raise RuntimeError('STOP_REQUESTED exists; no implicit override')
    training_integrity(run, checkpoint['update'])
    return checkpoint, digest


def recovery_plan(path):
    recovery = read(path)
    plan = load_plan(recovery['original_plan'])
    if (recovery['version'] != 'v6_scratch_recovery_u11_v1'
            or recovery['stop_updates'] != [25, 50]
            or recovery['adapter_sha256'] != sha256_file(__file__)
            or recovery['original_plan_sha256'] != sha256_file(recovery['original_plan'])):
        raise RuntimeError('frozen recovery/original plan changed')
    initial = read(Path(plan['run_dir'])/'initialization.json')
    if initial['plan_sha256'] != recovery['original_plan_sha256']:
        raise RuntimeError('original initialization/plan identity mismatch')
    return recovery, plan


def prepare(original, control):
    plan = load_plan(original)
    checkpoint, digest = validate_checkpoint(plan, 25)
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    record = {'version': 'v6_scratch_recovery_u11_v1', 'authorized_on': '2026-10-17 Asia/Shanghai',
              'original_plan': str(Path(original).resolve()), 'original_plan_sha256': sha256_file(original),
              'run_dir': plan['run_dir'], 'control_dir': str(control), 'resume_update': 11,
              'resume_sha256': digest, 'stop_updates': [25, 50], 'remaining_updates': 39,
              'original_total_episodes_budget': 800, 'evaluation_seeds': plan['evaluation_seeds'],
              'adapter_sha256': sha256_file(__file__), 'automatic_restart': False,
              'training_sources': 'all original 40 source and 12 scene hashes unchanged',
              'original_session': 'abrupt interruption, missing exit; never infer success from old PID'}
    atomic_write_json(record, control/'plan.json')
    atomic_copy_checkpoint(Path(plan['run_dir'])/'latest.pt', control/'u000011.pt')
    atomic_write_json(training_integrity(Path(plan['run_dir']), 11), control/'u000011_integrity.json')
    print(json.dumps({'status': 'RECOVERY_PREPARED_NOT_TRAINED', **record}, indent=2))


def archive_interrupted(plan, control):
    """Caller holds both the original controller and run locks; never invent exit0."""
    run, original = Path(plan['run_dir']), Path(plan['control_dir'])
    destination = control/'interrupted_original_session'
    destination.mkdir(exist_ok=False)
    for name in ('launch.json', 'exit.json', 'camera.json', 'monitor_latest.json'):
        source = run/'session'/name
        if source.exists():
            shutil.copy2(source, destination/name)
    for name in ('status.json', 'controller.stdout.log', 'controller.stderr.log'):
        shutil.copy2(original/name, destination/('controller_'+name))
    for name in ('episodes.jsonl', 'metrics.jsonl', 'checkpoint_manifest.json'):
        shutil.copy2(run/name, destination/name)
    atomic_write_json({'at': now(), 'safe_update': 11,
        'prior_exit': 'MISSING_OR_UNCONFIRMED_NOT_SUCCESS',
        'ownership': 'original controller and run OS locks acquired; live client audit and global CARLA lock required',
        'uncommitted_data': 'original episodes preserved here; trainer discards rows after u11 on strict resume'},
        destination/'interruption_review.json')
    for name in ('launch.json', 'exit.json'):
        source = run/'session'/name
        if source.exists():
            source.replace(destination/('original_'+name))
    state = read(original/'status.json')
    state.update(status='SUPERSEDED_BY_EXPLICIT_RECOVERY', recovery_control=str(control), changed_at=now())
    atomic_write_json(state, original/'status.json')


def train_segment(plan, checkpoint, stop):
    """Hold the machine-wide clock lease while the original trainer child runs."""
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    run, session = Path(plan['run_dir']), Path(plan['run_dir'])/'session'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    log = session/f'training_{stamp}.log'
    cmd = resume_command(plan, checkpoint, stop)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    with CarlaOwner(run_dir=run, purpose='explicit strict V6 scratch recovery'):
        _preflight_idle_world(host='127.0.0.1', port=2000, expected_version='0.9.16')
        with log.open('x', encoding='utf-8') as stream:
            trainer = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env)
            atomic_write_json({'launch_id': stamp, 'trainer_pid': trainer.pid, 'supervisor_pid': os.getpid(),
                'command': cmd, 'log_path': str(log), 'started_at': now(), 'stop_update': stop,
                'resume_update': checkpoint['update'], 'status': 'DISPATCHED_VERIFY_CHECKPOINT',
                'monitor_interval_seconds': 1200, 'automatic_restart': False}, session/'launch.json')
            with (session/f'camera_{stamp}.log').open('x', encoding='utf-8') as camera_stream:
                camera = subprocess.Popen([sys.executable, '-u', '-m', 'utils.follow_reward_fork',
                    '--trainer-pid', str(trainer.pid), '--run-dir', str(run)], cwd=ROOT,
                    stdout=camera_stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
                deadline = 0
                while trainer.poll() is None:
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
            atomic_write_json({'launch_id': stamp, 'trainer_pid': trainer.pid, 'supervisor_pid': os.getpid(),
                'returncode': trainer.returncode, 'camera_pid': camera.pid, 'camera_returncode': camera.poll(),
                'status': 'EXITED', 'ended_at': now(), 'automatic_restart': False}, session/'exit.json')
            save_snapshot(run)
            if trainer.returncode:
                raise RuntimeError(f'trainer exited {trainer.returncode}; no restart')


def frozen_screen(plan, control, stop):
    from diagnostics.reward_instant_v6_20261015.round2_v1.driver import lateral_analysis
    run = Path(plan['run_dir'])
    evidence = inspect(run)
    atomic_write_json(evidence, control/f'u{stop:06d}_training_inspect.json')
    if (evidence['safe_update'] != stop or evidence['integrity_issues'] or evidence['metrics_commit_pending']
            or not evidence['exit_is_current_launch'] or evidence['exit'].get('returncode') != 0):
        raise RuntimeError('training did not finish at a healthy exact boundary')
    integrity = training_integrity(run, stop)
    atomic_write_json(integrity, control/f'u{stop:06d}_all_updates_integrity.json')
    frozen = control/f'u{stop:06d}.pt'
    if frozen.exists():
        raise FileExistsError(frozen)
    digest = atomic_copy_checkpoint(run/'latest.pt', frozen)
    atomic_write_json(read(run/'checkpoint_manifest.json'), control/f'u{stop:06d}_checkpoint_manifest.json')
    evaluation = control/f'screen_u{stop:06d}'
    command('diagnostics.run_reward_screen', '--checkpoints', frozen, '--output', evaluation,
            '--seeds', *plan['evaluation_seeds'])
    command('diagnostics.analyze_reward_screen', '--evaluation-dir', evaluation,
            '--output', evaluation/'behavior_analysis.json')
    analysis = read(evaluation/'behavior_analysis.json')
    if analysis['analysis_evidence_status'] != 'COMPLETE_VALID_EVIDENCE' or analysis['errors']:
        raise RuntimeError('incomplete frozen evaluation evidence')
    lateral = lateral_analysis(evaluation)
    atomic_write_json(lateral, evaluation/'lateral_analysis.json')
    result = {'update': stop, 'source_type': 'fresh', 'checkpoint': str(frozen), 'sha256': digest,
              'aggregate': analysis['aggregate'], 'lateral': lateral, 'training_integrity': integrity,
              'completed_at': now(), 'qualification': 'NOT_ASSIGNED_KNOWN_SCENE_PILOT_ONLY'}
    atomic_write_json(result, control/f'u{stop:06d}_summary.json')
    with (ROOT/'docs/基础红策略_V6从零分阶段实验_20261015.md').open('a', encoding='utf-8') as stream:
        stream.write(f"\n恢复后u{stop}完成（UTC {now()}），累计{stop*16}回合；冻结共同到达"
                     f"{analysis['aggregate']['both_reached_count_valid_cases']}/12。证据`{control/f'u{stop:06d}_summary.json'}`；未授予驾驶/泛化资格。\n")
    return result


def controller(path):
    recovery, plan = recovery_plan(path)
    run, control = Path(plan['run_dir']), Path(recovery['control_dir'])
    # Old supervisors own these same locks. Never recover alongside them.
    with RunSessionOwner(control), RunSessionOwner(Path(plan['control_dir'])), RunSessionOwner(run):
        if (control/'status.json').exists():
            raise FileExistsError('recovery already dispatched; explicit review required')
        state = {**recovery, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': now(), 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            checkpoint, digest = validate_checkpoint(plan, 25)
            if digest != recovery['resume_sha256']:
                raise RuntimeError('u11 boundary changed after preparation')
            archive_interrupted(plan, control)
            for stop in recovery['stop_updates']:
                recovery_plan(path)
                checkpoint, _ = validate_checkpoint(plan, stop)
                if state['completed_stages']:
                    archive_session(run, control)
                state.update(status='TRAINING_DISPATCHED', target_update=stop,
                             resume_from_update=checkpoint['update'], changed_at=now())
                atomic_write_json(state, control/'status.json')
                train_segment(plan, checkpoint, stop)
                recovery_plan(path)
                state.update(status='FROZEN_EVALUATION', changed_at=now())
                atomic_write_json(state, control/'status.json')
                state['completed_stages'].append(frozen_screen(plan, control, stop))
                atomic_write_json(state, control/'status.json')
            state.update(status='COMPLETED', ended_at=now())
            atomic_write_json(state, control/'status.json')
        except BaseException as error:
            state.update(status='FAILED_NO_RESTART', error=f'{type(error).__name__}: {error}', ended_at=now())
            atomic_write_json(state, control/'status.json')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'validate', 'run'])
    parser.add_argument('--original-plan')
    parser.add_argument('--control-dir')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.original_plan, args.control_dir)
    elif args.mode == 'validate':
        _, plan = recovery_plan(args.plan)
        _, digest = validate_checkpoint(plan, 25)
        print(json.dumps({'status': 'STRICT_U11_RECOVERY_VALIDATED_NOT_TRAINED', 'sha256': digest}))
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
