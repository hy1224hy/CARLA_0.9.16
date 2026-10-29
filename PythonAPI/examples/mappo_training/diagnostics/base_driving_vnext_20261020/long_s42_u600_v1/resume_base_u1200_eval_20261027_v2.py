"""Explicit recovery of interrupted u1000 validation; same authorized u1200 budget."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
CONTROL = ROOT / 'diagnostics/base_driving_vnext_20261020/long_s42_u600_v1'
RECORD = CONTROL / 'recovery_eval_u1000_20261027_v2.json'
PREVIOUS = CONTROL / 'continue_base_u1200_recovery.py'
EXTENSION = CONTROL / 'extension_u1200_recovery_v1.json'
EXTENSION_SHA = '4bd640c454fc393dcd9425e984cede232a98928728feda8b1bd4aecdfe62e189'
SOURCE = CONTROL / 'u001000.pt'
SOURCE_SHA = '0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a'
FINGERPRINT = '8d35d03c4a09111331b00ed308fa08f04a659070b2205e4ff61e6749fb865700'
VERSION = 'base_driving_u1000_validation_recovery_u1200_v2'
COMPLETED = [50, 100, 200, 400, 600, 800]


def recovery_actions():
    return [('validation', 1000, 1000), ('train', 1000, 1200), ('screen', 1200, 1200)]


def require_within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or parent not in path.parents:
        raise ValueError('Recovery path must remain strictly inside control directory')
    return path


def validate_resume_evidence(*, info, parallel_manifest, parallel, validation_manifest,
                             validation_rows, matching_exit, completed):
    if (info['safe_update'] != 1000 or info['episodes'] != 16000 or info['sha256'] != SOURCE_SHA
            or info['actor_adam_steps'] != {'0': [5000], '1': [5000]}
            or info['critic_adam_steps'] != [5000] or info['value_normalizer_update_count'] != 1000
            or set(info['rng_components']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
            or info['lr_schedule']['decay_horizon'] != 600 or info['source_fingerprint'] != FINGERPRINT):
        raise ValueError('Full-state u1000 source differs from the completed safe checkpoint')
    reports = parallel.get('reports', [])
    if (parallel_manifest.get('status') != 'COMPLETED' or parallel_manifest.get('completed_cases') != 32
            or len(reports) != 1 or reports[0].get('case_count') != 32
            or reports[0].get('checkpoint_sha256') != SOURCE_SHA
            or reports[0].get('evidence_status') != 'COMPLETE_VALID_EVIDENCE' or reports[0].get('errors')):
        raise ValueError('Cannot retain incomplete or mismatched parallel evaluation')
    if (validation_manifest.get('status') != 'RUNNING' or validation_manifest.get('completed_cases') != 0
            or validation_rows):
        raise ValueError('Interrupted validation changed; review evidence rather than discard results')
    if not matching_exit or completed != COMPLETED:
        raise ValueError('Missing matching actual training exit or earlier completed frozen stages')


def runtime():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('original_u1200_controller', PREVIOUS)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    return previous, previous.base


def tree_hashes(directory, sha256_file):
    return {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def get_evidence(previous, base, record):
    from evaluation.base_driving_qualification import analyze_evaluation
    run = Path(record['run_dir'])
    stage = CONTROL / 'screen_u001000'
    qualification = ROOT / base.load_contract(record['contract_path'], ROOT)['payload']['qualification_contract']
    if previous.sha256_file(EXTENSION) != EXTENSION_SHA or previous.sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError('Immutable authorized extension or u1000 checkpoint changed')
    if record['total_budget_updates'] != 1200 or record['total_budget_episodes'] != 19200:
        raise ValueError('Recovery must retain original explicit u1200 budget')
    if (CONTROL / 'u001000_summary.json').exists():
        raise ValueError('u1000 already completed; do not repeat its evaluation')
    info = previous.checkpoint_summary(record)
    parallel = analyze_evaluation(stage / 'parallel', qualification)
    validate_resume_evidence(info=info, parallel_manifest=base.read(stage / 'parallel/manifest.json'),
        parallel=parallel, validation_manifest=base.read(stage / 'validation/manifest.json'),
        validation_rows=base.rows(stage / 'validation/episodes.jsonl'),
        matching_exit=base._matching_exit(run, stop=1000), completed=base._completed_screens(record))
    launch = base.read(run / 'session/launch.json')
    if launch['launch_id'] != '9597351f69b44b9e9b86a7d1af257d37' or launch['actual_worker_pid'] != 28496:
        raise ValueError('Completed u1000 training identity changed')
    return info, parallel, qualification


def prepare():
    previous, base = runtime()
    record = previous.effective_record()
    run = Path(record['run_dir'])
    if RECORD.exists():
        raise FileExistsError('Recovery already prepared; no implicit retry')
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        info, _, _ = get_evidence(previous, base, record)
        world = previous.preflight(record)
        stage = CONTROL / 'screen_u001000'
        archive = require_within(CONTROL / ('interrupted_validation_u001000_20261027_v2_' + uuid.uuid4().hex[:8]), CONTROL)
        payload = {
            'version': VERSION, 'prepared_at': base.now(),
            'authorization': 'Human reported CARLA crash, restarted it, and explicitly requested resuming training and monitoring on2026-10-27.',
            'budget_extension': str(EXTENSION), 'extension_sha256': EXTENSION_SHA,
            'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'remaining_training_updates': 200, 'remaining_training_episodes': 3200,
            'source_checkpoint': str(SOURCE), 'source_checkpoint_sha256': SOURCE_SHA,
            'complete_source_state': info, 'controller_source': str(Path(__file__).resolve()),
            'controller_source_sha256': previous.sha256_file(__file__),
            'prior_status': base.read(CONTROL / 'status.json'),
            'prior_status_sha256': previous.sha256_file(CONTROL / 'status.json'),
            'prior_launch': base.read(run / 'session/launch.json'),
            'prior_exit': base.read(run / 'session/exit.json'),
            'prior_launch_sha256': previous.sha256_file(run / 'session/launch.json'),
            'prior_exit_sha256': previous.sha256_file(run / 'session/exit.json'),
            'retained_parallel_file_sha256': tree_hashes(stage / 'parallel', previous.sha256_file),
            'interrupted_validation_file_sha256': tree_hashes(stage / 'validation', previous.sha256_file),
            'interrupted_validation_archive': str(archive),
            'live_preflight': world, 'actions': recovery_actions(), 'automatic_restart': False,
            'recovery_policy': 'Retain parallel32 byte-for-byte; archive incomplete validation0/64 with original status; rerun validation64, then native full-state1000->1200 and complete frozen1200.',
        }
        previous.atomic_write_json(payload, RECORD)
        print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'record': str(RECORD), 'source': info, 'preflight': world}), flush=True)


def finish_validation(previous, base, record, authorization, parallel, qualification):
    from diagnostics.v6_scratch_session import training_integrity
    from evaluation.evaluate_base_driving_vnext import evaluate
    from evaluation.base_driving_qualification import select_candidate
    stage = CONTROL / 'screen_u001000'
    interrupted = require_within(stage / 'validation', CONTROL)
    archive = require_within(authorization['interrupted_validation_archive'], CONTROL)
    if archive.exists(): raise FileExistsError('Validation archive already exists; no implicit retry')
    if tree_hashes(interrupted, previous.sha256_file) != authorization['interrupted_validation_file_sha256']:
        raise ValueError('Interrupted validation evidence changed after preparation')
    interrupted.replace(archive)
    previous.atomic_write_json({'at': base.now(), 'reason': 'Human-confirmed CARLA crash/restart',
        'original_manifest_status': base.read(archive / 'manifest.json')['status'],
        'completed_cases': 0, 'evaluation_success_claimed': False,
        'original_files_sha256': authorization['interrupted_validation_file_sha256'],
        'recovery_record': str(RECORD)}, archive / 'interruption_review.json')
    validation = evaluate([SOURCE], stage / 'validation',
        Path(record['validation_scenario_dir']) / 'protocol/manifest.json', qualification,
        geometry_report=record['geometry_reports']['validation'])
    reports = validation['reports']
    if (len(reports) != 1 or reports[0]['checkpoint_sha256'] != SOURCE_SHA
            or reports[0]['case_count'] != 64 or reports[0]['evidence_status'] != 'COMPLETE_VALID_EVIDENCE'
            or reports[0]['errors']): raise ValueError('Recovered validation is not complete valid64 evidence')
    if tree_hashes(stage / 'parallel', previous.sha256_file) != authorization['retained_parallel_file_sha256']:
        raise ValueError('Preserved parallel evidence changed')
    if previous.sha256_file(SOURCE) != SOURCE_SHA or not base._matching_exit(record['run_dir'], stop=1000):
        raise ValueError('Frozen source or real matching training exit changed')
    selection = select_candidate(reports, regression_reports=parallel['reports'])
    previous.atomic_write_json(selection, stage / 'selection.json')
    summary = {
        'status': 'COMPLETED', 'update': 1000, 'ended_at': base.now(),
        'checkpoint': str(SOURCE), 'checkpoint_sha256': SOURCE_SHA,
        'parallel_directory': str(stage / 'parallel'), 'validation_directory': str(stage / 'validation'),
        'parallel_qualification_sha256': previous.sha256_file(stage / 'parallel/qualification.json'),
        'validation_qualification_sha256': previous.sha256_file(stage / 'validation/qualification.json'),
        'selection': selection, 'training_integrity': training_integrity(record['run_dir'], 1000),
        'normal_training_matching_exit_0': True, 'explicit_safe_boundary_recovery': False,
        'prior_session_archive': None, 'recovery_record': str(RECORD),
        'interrupted_validation_archive': str(archive), 'retained_parallel_cases': 32,
        'qualification_scope': 'engineering validation and training regression; not formal generalization',
    }
    previous.atomic_write_json(summary, CONTROL / 'u001000_summary.json')
    return summary


def controller():
    previous, base = runtime()
    record = previous.effective_record()
    run = Path(record['run_dir'])
    authorization = base.read(RECORD)
    if authorization['version'] != VERSION or authorization['controller_source_sha256'] != previous.sha256_file(__file__):
        raise ValueError('Recovery controller binding changed')
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        if (run / 'session/STOP_REQUESTED').exists(): raise ValueError('STOP_REQUESTED requires review')
        for path, key in ((CONTROL / 'status.json', 'prior_status_sha256'),
                          (run / 'session/launch.json', 'prior_launch_sha256'),
                          (run / 'session/exit.json', 'prior_exit_sha256')):
            if previous.sha256_file(path) != authorization[key]: raise ValueError('Prior session changed after preparation')
        info, parallel, qualification = get_evidence(previous, base, record)
        if tree_hashes(CONTROL / 'screen_u001000/parallel', previous.sha256_file) != authorization['retained_parallel_file_sha256']:
            raise ValueError('Preserved parallel changed after preparation')
        previous.preflight(record)
        state = {'version': VERSION, 'status': 'STARTING', 'pid': os.getpid(), 'started_at': base.now(),
            'initial_safe_update': 1000, 'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'extension_sha256': EXTENSION_SHA, 'recovery_record': str(RECORD),
            'recovery_record_sha256': previous.sha256_file(RECORD), 'completed_stages': list(COMPLETED),
            'automatic_restart': False, 'explicit_interruption_recovery': True}
        previous.atomic_write_json(state, CONTROL / 'status.json')
        try:
            state.update(status='FROZEN_EVALUATION', target_update=1000, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            result = finish_validation(previous, base, record, authorization, parallel, qualification)
            if base._completed_screens(record) != COMPLETED + [1000]: raise ValueError('u1000 summary verification failed')
            state['completed_stages'].append(1000)
            state['last_stage_selection'] = result['selection']
            state.update(status='TRAINING', target_update=1200, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            previous.effective_record()
            base.safe_boundary(record, expected=1000)
            state['completed_u1000_training_archive'] = base.archive_session(record, reason='completed')
            previous.atomic_write_json(state, CONTROL / 'status.json')
            previous.supervise(record, 1000, 1200)
            state.update(status='FROZEN_EVALUATION', target_update=1200, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            result = base.frozen_screen(record, 1200)
            state['completed_stages'].append(1200)
            state['last_stage_selection'] = result['selection']
            state.update(status='COMPLETED', ended_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            previous.snapshot(record)
        except BaseException as error:
            state.update(status='FAILED_NO_RESTART', error=f'{type(error).__name__}: {error}', ended_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else controller()
