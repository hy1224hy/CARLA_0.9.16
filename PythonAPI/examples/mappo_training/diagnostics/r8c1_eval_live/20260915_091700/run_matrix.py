"""Execute the approved finite diagnostic matrix, sequentially on one CARLA world."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from evaluation.validate_base_c0_run import compare_c0_runs
from training.checkpoint_protocol import atomic_write_json

OUTPUT = Path(__file__).resolve().parent
CONTRACT = 'configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json'
TRAINING = 'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/periodic'
SMOKES = ['P0_BASE_C0_smoke_u000100_20260915_091700', 'P0_BASE_C0_smoke_u000100_20260915_091745']
state = {'status': 'RUNNING', 'purpose': 'diagnostic_only', 'selection_allowed': False,
         'performance_decision': 'NOT_EVALUATED', 'smoke_eval_ids': SMOKES, 'runs': [], 'comparisons': []}
if '--continue-independent-pairs' in sys.argv:
    state = json.loads((OUTPUT / 'matrix_status.json').read_text(encoding='utf-8'))
    if any(item['status'] != 'DIAGNOSTIC_PASS' for item in state['runs']):
        raise RuntimeError('cannot continue after an incomplete or invalid evaluation')
    state['continuation_reason'] = 'Retain u20 repeatability HOLD; complete other already-approved independent pairs without changing thresholds or rerunning completed cases.'
    state['status'] = 'RUNNING'


def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_write_json(state, OUTPUT / 'matrix_status.json')


try:
    smoke = compare_c0_runs(*(PROJECT / 'evaluations' / name for name in SMOKES))
    if not smoke['gate_pass']:
        raise RuntimeError('smoke repeatability gate failed')
    save()
    for profile, update in [('short_probe', 20), ('short_probe', 60), ('short_probe', 100), ('long_probe', 100)]:
        previous = [item for item in state['runs'] if item['profile'] == profile and item['update'] == update]
        if previous:
            if len(previous) != 2:
                raise RuntimeError('incomplete pair requires separate recovery')
            compare_c0_runs(*(Path(item['output_dir']) for item in previous))
            continue
        pair = []
        for repeat in (1, 2):
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_id = f'P0_BASE_C0_{profile}_u{update:06d}_{stamp}'
            directory = PROJECT / 'evaluations' / eval_id
            command = [sys.executable, '-B', '-u', '-m', 'evaluation.evaluate_base_policy_c0',
                       '--eval-contract', CONTRACT, '--checkpoint', f'{TRAINING}/ckpt_update_{update:06d}.pt',
                       '--profile', profile, '--eval-id', eval_id, '--output-dir', str(directory),
                       '--host', '127.0.0.1', '--port', '2000']
            item = {'eval_id': eval_id, 'profile': profile, 'update': update, 'repeat': repeat,
                    'command': command, 'output_dir': str(directory), 'status': 'RUNNING'}
            state['runs'].append(item)
            save()
            print(f'START {eval_id} repeat={repeat}', flush=True)
            with (OUTPUT / f'{eval_id}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            item['exit_code'] = result.returncode
            if result.returncode:
                item['status'] = 'HOLD'
                raise RuntimeError(f'evaluation failed: {eval_id}, exit={result.returncode}')
            manifest = json.loads((directory / 'evaluation_manifest.json').read_text(encoding='utf-8'))
            item.update(status=manifest['status'], completed_cases=manifest['completed_cases'],
                        cleanup=manifest['post_cleanup'])
            pair.append(directory)
            save()
            print(f'FINISH {eval_id} cases={item["completed_cases"]}', flush=True)
        comparison = compare_c0_runs(*pair)
        comparison_path = OUTPUT / f'{profile}_u{update:06d}_repeatability.json'
        atomic_write_json(comparison, comparison_path)
        state['comparisons'].append({'profile': profile, 'update': update, 'status': comparison['status'],
                                     'gate_pass': comparison['gate_pass'], 'path': str(comparison_path)})
        save()
        print(f'REPEAT {profile} u{update} {comparison["status"]}', flush=True)
        if not comparison['gate_pass']:
            if '--continue-independent-pairs' not in sys.argv:
                raise RuntimeError('frozen repeatability gate failed; remaining matrix stopped')
            print('HOLD retained; continuing independent approved pairs', flush=True)
    state['status'] = 'COMPLETED_WITH_REPEATABILITY_HOLD' if any(not x['gate_pass'] for x in state['comparisons']) else 'COMPLETED'
except BaseException as error:
    state.update(status='HOLD', error=f'{type(error).__name__}: {error}')
    save()
    raise
save()
print('MATRIX COMPLETED', flush=True)
