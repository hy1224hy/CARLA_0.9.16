"""Finalize already complete32+64 u1000 results; continue full-state training to1200."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
CONTROL = ROOT / 'diagnostics/base_driving_vnext_20261020/long_s42_u600_v1'
RECORD = CONTROL / 'recovery_completed_eval_u1000_20261027_v3.json'
V2_RECORD = CONTROL / 'recovery_eval_u1000_20261027_v2.json'
SOURCE = CONTROL / 'u001000.pt'
SOURCE_SHA = '0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a'
VERSION = 'base_driving_completed_u1000_evaluation_resume_u1200_v3'
COMPLETED = [50, 100, 200, 400, 600, 800]


def validate_reports(parallel, validation):
    for report, count in ((parallel, 32), (validation, 64)):
        if (report.get('checkpoint_sha256') != SOURCE_SHA or report.get('case_count') != count
                or report.get('evidence_status') != 'COMPLETE_VALID_EVIDENCE' or report.get('errors')):
            raise ValueError('Both registered frozen suites must have complete valid matching evidence')


def make_summary(record, selection, training_integrity, sha256_file, now, provenance):
    run = Path(record['run_dir'])
    stage = CONTROL / 'screen_u001000'
    return dict({
        'status': 'COMPLETED', 'update': 1000, 'ended_at': now(), 'checkpoint': str(SOURCE),
        'checkpoint_sha256': SOURCE_SHA, 'parallel_directory': str(stage / 'parallel'),
        'validation_directory': str(stage / 'validation'),
        'parallel_qualification_sha256': sha256_file(stage / 'parallel/qualification.json'),
        'validation_qualification_sha256': sha256_file(stage / 'validation/qualification.json'),
        'selection': selection, 'training_integrity': training_integrity(run, 1000),
        'normal_training_matching_exit_0': True, 'explicit_safe_boundary_recovery': False,
        'prior_session_archive': None, 'retained_parallel_cases': 32, 'retained_validation_cases': 64,
        'qualification_scope': 'engineering validation and training regression; not formal generalization',
    }, **provenance)


def runtime():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('validation_recovery_v2', CONTROL / 'resume_base_u1200_eval_20261027_v2.py')
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    previous, base = v2.runtime()
    return v2, previous, base


def verified_evidence(v2, previous, base, record):
    from evaluation.base_driving_qualification import analyze_evaluation, select_candidate
    authorization = base.read(V2_RECORD)
    if previous.sha256_file(CONTROL / 'resume_base_u1200_eval_20261027_v2.py') != authorization['controller_source_sha256']:
        raise ValueError('Original failed controller evidence changed')
    if record['total_budget_updates'] != 1200 or record['total_budget_episodes'] != 19200:
        raise ValueError('Cannot change existing authorized budget')
    info = previous.checkpoint_summary(record)
    if (info['safe_update'] != 1000 or info['episodes'] != 16000 or info['sha256'] != SOURCE_SHA
            or previous.sha256_file(SOURCE) != SOURCE_SHA
            or info['actor_adam_steps'] != {'0': [5000], '1': [5000]} or info['critic_adam_steps'] != [5000]
            or info['value_normalizer_update_count'] != 1000
            or set(info['rng_components']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
            or info['lr_schedule']['decay_horizon'] != 600 or info['source_fingerprint'] != v2.FINGERPRINT):
        raise ValueError('Complete source1000 state changed')
    if (CONTROL / 'u001000_summary.json').exists() or base._completed_screens(record) != COMPLETED:
        raise ValueError('Completed stages changed; no repeated summary or evaluation')
    run = Path(record['run_dir'])
    if not base._matching_exit(run, stop=1000): raise ValueError('Missing real matched training exit')
    for name in ('launch', 'exit'):
        if previous.sha256_file(run / f'session/{name}.json') != authorization[f'prior_{name}_sha256']:
            raise ValueError('Original training identity changed')
    qualification = ROOT / base.load_contract(record['contract_path'], ROOT)['payload']['qualification_contract']
    stage = CONTROL / 'screen_u001000'
    parallel = analyze_evaluation(stage / 'parallel', qualification)['reports']
    validation = analyze_evaluation(stage / 'validation', qualification)['reports']
    if len(parallel) != 1 or len(validation) != 1: raise ValueError('One frozen checkpoint required')
    validate_reports(parallel[0], validation[0])
    if v2.tree_hashes(stage / 'parallel', previous.sha256_file) != authorization['retained_parallel_file_sha256']:
        raise ValueError('Retained parallel32 evidence changed')
    selection = select_candidate(validation, regression_reports=parallel)
    if base.read(stage / 'selection.json') != selection: raise ValueError('Persisted selection changed')
    return info, selection


def prepare():
    if RECORD.exists(): raise FileExistsError('Already prepared; no implicit retry')
    v2, previous, base = runtime()
    record = previous.effective_record()
    run = Path(record['run_dir'])
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        info, selection = verified_evidence(v2, previous, base, record)
        prior = base.read(CONTROL / 'status.json')
        if (prior['version'] != v2.VERSION or prior['status'] != 'FAILED_NO_RESTART'
                or not prior.get('error', '').startswith("TypeError: unsupported operand type(s) for /")):
            raise ValueError('Not the reviewed summary Path/string failure')
        world = previous.preflight(record)
        stage = CONTROL / 'screen_u001000'
        payload = {'version': VERSION, 'prepared_at': base.now(),
            'authorization': 'Continue the human-authorized simulator crash recovery; repair summary Path conversion without repeating any complete evaluation.',
            'controller_source': str(Path(__file__).resolve()), 'controller_source_sha256': previous.sha256_file(__file__),
            'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'source_checkpoint': str(SOURCE), 'source_checkpoint_sha256': SOURCE_SHA, 'complete_source_state': info,
            'prior_status': prior, 'prior_status_sha256': previous.sha256_file(CONTROL / 'status.json'),
            'prior_recovery_record': str(V2_RECORD), 'prior_recovery_record_sha256': previous.sha256_file(V2_RECORD),
            'retained_parallel_file_sha256': v2.tree_hashes(stage / 'parallel', previous.sha256_file),
            'retained_validation_file_sha256': v2.tree_hashes(stage / 'validation', previous.sha256_file),
            'live_preflight': world, 'verified_selection': selection, 'automatic_restart': False}
        previous.atomic_write_json(payload, RECORD)
        print(json.dumps({'status': 'PREPARED_NOT_STARTED', 'source': info, 'preflight': world,
                          'retained_complete_evaluation_cases': 96, 'selection': selection}), flush=True)


def controller():
    v2, previous, base = runtime()
    record = previous.effective_record()
    run = Path(record['run_dir'])
    authorization = base.read(RECORD)
    if authorization['version'] != VERSION or authorization['controller_source_sha256'] != previous.sha256_file(__file__):
        raise ValueError('Recovery controller binding changed')
    with ExitStack() as owners:
        for directory in (run, CONTROL): owners.enter_context(previous.RunSessionOwner(directory))
        if previous.sha256_file(CONTROL / 'status.json') != authorization['prior_status_sha256']:
            raise ValueError('Prior state changed after explicit preparation')
        if previous.sha256_file(V2_RECORD) != authorization['prior_recovery_record_sha256']:
            raise ValueError('Prior recovery evidence changed')
        _, selection = verified_evidence(v2, previous, base, record)
        for suite in ('parallel', 'validation'):
            if v2.tree_hashes(CONTROL / f'screen_u001000/{suite}', previous.sha256_file) != authorization[f'retained_{suite}_file_sha256']:
                raise ValueError('Complete frozen evidence changed after preparation')
        if (run / 'session/STOP_REQUESTED').exists(): raise ValueError('STOP_REQUESTED requires review')
        previous.preflight(record)
        state = {'version': VERSION, 'status': 'FINALIZING_COMPLETE_EVALUATION', 'pid': os.getpid(),
            'started_at': base.now(), 'initial_safe_update': 1000, 'target_update': 1000,
            'total_budget_updates': 1200, 'total_budget_episodes': 19200,
            'recovery_record': str(RECORD), 'recovery_record_sha256': previous.sha256_file(RECORD),
            'extension_sha256': v2.EXTENSION_SHA, 'completed_stages': list(COMPLETED),
            'automatic_restart': False, 'explicit_interruption_recovery': True}
        previous.atomic_write_json(state, CONTROL / 'status.json')
        try:
            from diagnostics.v6_scratch_session import training_integrity
            summary = make_summary(record, selection, training_integrity, previous.sha256_file, base.now,
                {'recovery_record': str(RECORD), 'prior_recovery_record': str(V2_RECORD),
                 'interrupted_validation_archive': base.read(V2_RECORD)['interrupted_validation_archive']})
            previous.atomic_write_json(summary, CONTROL / 'u001000_summary.json')
            if base._completed_screens(record) != COMPLETED + [1000]: raise ValueError('Complete summary recheck failed')
            state['completed_stages'].append(1000)
            state['last_stage_selection'] = selection
            state.update(status='TRAINING', target_update=1200, changed_at=base.now())
            previous.atomic_write_json(state, CONTROL / 'status.json')
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
