"""Finite fresh V6 pilot, using the existing trainer and serial frozen screens.

prepare/validate/controller bookkeeping do not connect to CARLA. Only the train
subcommand and the existing evaluator may own the shared simulator clock.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from training.checkpoint_protocol import (
    atomic_copy_checkpoint, atomic_write_json, ensure_fresh_run_directory,
    load_checkpoint_file, sha256_file, validate_checkpoint_header,
)
from utils.reward_fork_session import inspect, rows
from utils.mut_mappo_session import RunSessionOwner

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ('train/update_skipped', 'reward_audit/nonfinite_count',
           'reward_audit/max_abs_step_residual',
           'policy/agent_0/preupdate_max_abs_ratio_error',
           'policy/agent_1/preupdate_max_abs_ratio_error')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def make_training_args(plan, stop, checkpoint=None):
    """Use only frozen configuration, never weights from the reference run."""
    from training import train_no_rllib_mappo_robust_v2 as trainer
    stops = plan['stop_updates']
    if stop not in stops or isinstance(stop, bool):
        raise ValueError('stop update is outside the approved stage budget')
    config = plan['training_args']
    if config.get('resume') or config.get('pretrained_checkpoint'):
        raise ValueError('fresh plan cannot inject a previous training source')
    args = trainer.build_parser().parse_args([])
    for name, value in config.items():
        if not hasattr(args, name) or name.startswith('_'):
            raise ValueError(f'unknown training argument {name}')
        setattr(args, name, value)
    if args.reward_version != 'v6' or args.pace_profile != 'instant_pace_v6':
        raise ValueError('scratch experiment requires unchanged V6')
    run = Path(plan['run_dir']).resolve()
    args.save_dir, args.run_id, args.updates = str(run), run.name, stop
    args.pretrained_checkpoint = ''
    args.stop_request_file = str(run/'session/STOP_REQUESTED')
    args.follow_spectator, args.allow_resume_code_drift = True, False
    args._explicit_cli_options = {'--'+key.replace('_', '-') for key in config}
    index = stops.index(stop)
    if index == 0:
        if checkpoint is not None:
            raise ValueError('fresh stage cannot load a checkpoint')
        if run.exists():
            ensure_fresh_run_directory(run)
        args.resume = ''
    else:
        if checkpoint is None:
            raise ValueError('resume stage requires a complete checkpoint')
        metadata = checkpoint['run_metadata']
        if (checkpoint['update'] != stops[index-1]
                or metadata.get('source_type') != 'fresh'
                or metadata.get('source_checkpoint') or 'reward_fork' in metadata
                or metadata.get('run_id') != run.name):
            raise ValueError('resume requires this fresh run at its exact previous stage boundary')
        args.resume = 'latest'
    return args


def training_integrity(run, stop):
    committed = [r for r in rows(run/'metrics.jsonl') if 0 < r['update'] <= stop]
    if [r['update'] for r in committed] != list(range(1, stop+1)):
        raise RuntimeError('incomplete or duplicate safe metric sequence')
    maxima = {key: 0.0 for key in CHECKED}
    for row in committed:
        for key, value in row['scalars'].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"u{row['update']} nonfinite {key}")
        for key in CHECKED:
            value = abs(row['scalars'][key])
            maxima[key] = max(maxima[key], value)
            limit = 1e-6 if 'ratio_error' in key or 'residual' in key else 0
            if value > limit:
                raise RuntimeError(f"u{row['update']} invalid {key}={value}")
    episodes = [r for r in rows(run/'episodes.jsonl') if 0 < r['update'] <= stop]
    counts = Counter(r['update'] for r in episodes)
    if counts != Counter({update: 16 for update in range(1, stop+1)}):
        raise RuntimeError('incomplete or duplicate episode evidence')
    return {'updates': len(committed), 'episodes': len(episodes), 'maxima': maxima,
            'outcomes': {key: sum(bool(r.get(key, False)) for r in episodes)
                         for key in ('success', 'collision', 'offroad', 'road_violation', 'timeout')}}


def load_plan(path):
    plan = read(path)
    if plan.get('version') != 'v6_scratch_v1' or plan.get('stop_updates') != [25, 50]:
        raise ValueError('unregistered scratch plan or budget')
    run = Path(plan['run_dir']).resolve()
    if run.parent != (ROOT/'logs').resolve():
        raise ValueError('fresh run must use the project logs directory')
    for name, digest in plan['source_sha256'].items():
        if sha256_file(ROOT/name) != digest:
            raise RuntimeError(f'frozen source changed: {name}')
    for name, digest in plan['scenario_sha256'].items():
        if sha256_file(ROOT/name) != digest:
            raise RuntimeError(f'frozen scenario changed: {name}')
    return plan


def prepare(reference, run, control):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    reference = Path(reference).resolve()
    checkpoint, reference_sha = load_checkpoint_file(reference, map_location='cpu')
    validate_checkpoint_header(checkpoint)
    if checkpoint['training_config'].get('pace_profile') != 'instant_pace_v6':
        raise ValueError('configuration reference must use V6')
    run, control = Path(run).resolve(), Path(control).resolve()
    if run.exists() or control.exists():
        raise FileExistsError('independent run/control directory already exists')
    config = {**checkpoint['args'], 'resume': '', 'pretrained_checkpoint': '',
              'save_dir': str(run), 'run_id': run.name, 'updates': 50,
              'config_path': str(control/'plan.json'), 'experiment_contract': '',
              'continuation_contract': '', 'allow_resume_code_drift': False,
              'stop_request_file': str(run/'session/STOP_REQUESTED')}
    sources = trainer.training_source_paths(ROOT) + [Path(__file__),
        ROOT/'utils/reward_fork_session.py', ROOT/'utils/mut_mappo_session.py',
        ROOT/'utils/follow_reward_fork.py', ROOT/'utils/r8c1_training_session.py',
        ROOT/'evaluation/evaluate_reward_fork.py', ROOT/'evaluation/diagnose_r8c1_behavior.py',
        ROOT/'evaluation/evaluate_base_policy_c0.py', ROOT/'diagnostics/run_reward_screen.py',
        ROOT/'diagnostics/analyze_reward_screen.py',
        ROOT/'diagnostics/reward_instant_v6_20261015/round2_v1/driver.py']
    scenes = sorted((ROOT/'data/training_scenarios_town05_r8a_v1').glob('*.json'))
    # Pin sampler metadata as well as the 12 scenario files.
    plan = {'version': 'v6_scratch_v1', 'authorized_on': '2026-10-15 Asia/Shanghai',
            'run_dir': str(run), 'control_dir': str(control), 'source_update': 0,
            'initialization': 'fresh actors/critic/optimizers/ValueNorm/RNG/curriculum; no weights loaded',
            'configuration_reference': str(reference), 'configuration_reference_sha256': reference_sha,
            'training_args': config, 'stop_updates': [25, 50], 'total_episodes_budget': 800,
            'evaluation_seeds': [62001], 'automatic_restart': False,
            'source_sha256': {str(p.resolve().relative_to(ROOT)).replace('\\','/'): sha256_file(p) for p in sources},
            'scenario_sha256': {str(p.relative_to(ROOT)).replace('\\','/'): sha256_file(p) for p in scenes},
            'qualification': 'known-scene single-seed learning pilot; not generalization'}
    control.mkdir(parents=True, exist_ok=False)
    atomic_write_json(plan, control/'plan.json')
    print(json.dumps({'status': 'PREPARED_NOT_TRAINED', 'plan': str(control/'plan.json')}, indent=2))


def run_training(plan_path, stop):
    from training import train_no_rllib_mappo_robust_v2 as trainer
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    plan = load_plan(plan_path)
    run = Path(plan['run_dir'])
    initial = read(run/'initialization.json')
    if initial['plan_sha256'] != sha256_file(plan_path):
        raise ValueError('scratch plan changed after initialization')
    checkpoint = None
    if stop != plan['stop_updates'][0]:
        checkpoint, digest = load_checkpoint_file(run/'latest.pt', map_location='cpu')
        latest = read(run/'checkpoint_manifest.json')['latest']
        if digest != latest['sha256'] or checkpoint['update'] != latest['update']:
            raise RuntimeError('resume checkpoint/manifest mismatch')
    args = make_training_args(plan, stop, checkpoint)
    # The trainer validates the full payload and restores every state before connecting.
    if checkpoint is not None:
        trainer.apply_resume_configuration(args, checkpoint)
    if Path(args.stop_request_file).exists():
        raise RuntimeError('STOP_REQUESTED exists; no implicit resume')
    print(json.dumps({'status': 'INITIALIZATION_VERIFIED', 'source_type': 'fresh',
                      'resume_from_update': checkpoint['update'] if checkpoint else None,
                      'stop_update': stop, 'lr_horizon': args.lr_decay_horizon}), flush=True)
    with CarlaOwner(run_dir=run, purpose='independent fresh V6 training'):
        _preflight_idle_world(host=args.host, port=args.port, expected_version='0.9.16')
        trainer.train(args)
    load_plan(plan_path)


def command(module, *arguments):
    cmd = [sys.executable, '-u', '-m', module, *map(str, arguments)]
    print(json.dumps({'at': now(), 'command': cmd}), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, stdin=subprocess.DEVNULL,
                   env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))


def archive_session(run, control):
    launch, exited = read(run/'session/launch.json'), read(run/'session/exit.json')
    if not launch.get('launch_id') or launch['launch_id'] != exited.get('launch_id') or exited.get('returncode') != 0:
        raise RuntimeError('previous session is incomplete or failed')
    destination = control/('completed_session_'+launch['launch_id'])
    destination.mkdir(exist_ok=False)
    for name in ('camera.json', 'monitor_latest.json'):
        if (run/'session'/name).exists():
            shutil.copy2(run/'session'/name, destination/name)
    for name in ('launch.json', 'exit.json'):
        (run/'session'/name).replace(destination/name)


def controller(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    run, control = Path(plan['run_dir']), Path(plan['control_dir'])
    with RunSessionOwner(control):
        if run.exists() or (control/'status.json').exists():
            raise FileExistsError('one-shot scratch controller already dispatched; inspect before recovery')
        state = {**plan, 'plan_sha256': sha256_file(plan_path), 'pid': os.getpid(),
                 'started_at': now(), 'status': 'STARTING', 'completed_stages': []}
        atomic_write_json(state, control/'status.json')
        try:
            run.mkdir(exist_ok=False)
            (run/'session').mkdir()
            atomic_write_json({'version': 'v6_scratch_v1', 'source_update': 0,
                               'source_type': 'fresh', 'plan_sha256': state['plan_sha256'],
                               'plan_path': str(plan_path), 'created_at': now(),
                               'status': 'PREPARED_NOT_TRAINED'}, run/'initialization.json')
            for stop in plan['stop_updates']:
                load_plan(plan_path)
                if (run/'session/STOP_REQUESTED').exists():
                    raise RuntimeError('STOP_REQUESTED prevents next stage')
                if state['completed_stages']:
                    archive_session(run, control)
                state.update(status='TRAINING_DISPATCHED', target_update=stop, changed_at=now())
                atomic_write_json(state, control/'status.json')
                command('utils.reward_fork_session', 'supervise', '--run-dir', run,
                        '--stop-update', stop, '--scratch-plan', plan_path)
                evidence = inspect(run)
                atomic_write_json(evidence, control/f'u{stop:06d}_training_inspect.json')
                if (evidence['safe_update'] != stop or evidence['integrity_issues']
                        or evidence['metrics_commit_pending'] or not evidence['exit_is_current_launch']
                        or evidence['exit'].get('returncode') != 0):
                    raise RuntimeError('training failed to reach a healthy exact boundary')
                integrity = training_integrity(run, stop)
                atomic_write_json(integrity, control/f'u{stop:06d}_all_updates_integrity.json')
                checkpoint, digest = load_checkpoint_file(run/'latest.pt', map_location='cpu')
                validate_checkpoint_header(checkpoint)
                if checkpoint['run_metadata']['source_type'] != 'fresh' or checkpoint['update'] != stop:
                    raise RuntimeError('unexpected trained checkpoint origin/update')
                frozen = control/f'u{stop:06d}.pt'
                if frozen.exists():
                    raise FileExistsError(frozen)
                if atomic_copy_checkpoint(run/'latest.pt', frozen) != digest:
                    raise RuntimeError('frozen checkpoint copy mismatch')
                atomic_write_json(read(run/'checkpoint_manifest.json'), control/f'u{stop:06d}_checkpoint_manifest.json')
                evaluation = control/f'screen_u{stop:06d}'
                state.update(status='FROZEN_EVALUATION', checkpoint=str(frozen),
                             evaluation_dir=str(evaluation), changed_at=now())
                atomic_write_json(state, control/'status.json')
                command('diagnostics.run_reward_screen', '--checkpoints', frozen, '--output', evaluation,
                        '--seeds', *plan['evaluation_seeds'])
                command('diagnostics.analyze_reward_screen', '--evaluation-dir', evaluation,
                        '--output', evaluation/'behavior_analysis.json')
                analysis = read(evaluation/'behavior_analysis.json')
                if analysis['analysis_evidence_status'] != 'COMPLETE_VALID_EVIDENCE' or analysis['errors']:
                    raise RuntimeError('incomplete frozen evaluation evidence')
                # Reuse the already-reviewed pure-file lateral analysis, including missing-control flags.
                from diagnostics.reward_instant_v6_20261015.round2_v1.driver import lateral_analysis
                lateral = lateral_analysis(evaluation)
                atomic_write_json(lateral, evaluation/'lateral_analysis.json')
                summary = {'update': stop, 'source_type': 'fresh', 'checkpoint': str(frozen),
                           'sha256': digest, 'aggregate': analysis['aggregate'], 'lateral': lateral,
                           'training_integrity': integrity, 'completed_at': now(),
                           'qualification': 'NOT_ASSIGNED_KNOWN_SCENE_PILOT_ONLY'}
                atomic_write_json(summary, control/f'u{stop:06d}_summary.json')
                state['completed_stages'].append(summary)
                atomic_write_json(state, control/'status.json')
                doc = ROOT/'docs/基础红策略_V6从零分阶段实验_20261015.md'
                with doc.open('a', encoding='utf-8') as stream:
                    stream.write(f"\n阶段u{stop}已完成，UTC {now()}：{stop*16}回合，"
                                 f"冻结共同到达{analysis['aggregate']['both_reached_count_valid_cases']}/12。"
                                 f"证据：`{control/f'u{stop:06d}_summary.json'}`。未授予驾驶或泛化资格。\n")
                # Zero early successes is learning evidence, not a numerical/runtime failure.
            state.update(status='COMPLETED', ended_at=now())
            atomic_write_json(state, control/'status.json')
        except BaseException as error:
            state.update(status='FAILED_NO_RESTART', error=f'{type(error).__name__}: {error}', ended_at=now())
            atomic_write_json(state, control/'status.json')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'validate', 'run', 'train'])
    parser.add_argument('--plan')
    parser.add_argument('--reference')
    parser.add_argument('--run-dir')
    parser.add_argument('--control-dir')
    parser.add_argument('--stop-update', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not all((args.reference, args.run_dir, args.control_dir)):
            parser.error('prepare requires --reference, --run-dir and --control-dir')
        prepare(args.reference, args.run_dir, args.control_dir)
    elif not args.plan:
        parser.error('--plan is required')
    elif args.mode == 'validate':
        plan = load_plan(args.plan)
        make_training_args(plan, plan['stop_updates'][0])
        print(json.dumps({'status': 'OFFLINE_VALIDATED_NOT_TRAINED', 'stages': plan['stop_updates'],
                          'source_files': len(plan['source_sha256']), 'scenario_files': len(plan['scenario_sha256'])}))
    elif args.mode == 'train':
        run_training(args.plan, args.stop_update)
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
