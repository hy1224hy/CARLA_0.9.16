"""Explicit R4 recovery: preserve runtime identity and resume complete boundary."""
from pathlib import Path
import json
import os
import sys
from datetime import datetime

ROOT = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
HERE = ROOT / 'experiments/reward_iteration_20261107/R4_three_dimension'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['PYTHONFAULTHANDLER'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from training.mut_mappo_config import load_config, validate_config, run_identity
from training.mut_mappo_checkpoint import CheckpointStore
from utils.mut_mappo_session import RunSessionOwner

with RunSessionOwner(HERE / 'run'):
    config = validate_config(load_config(HERE / 'config.json'))
    payload = CheckpointStore(HERE / 'run', run_identity(config)).load()
    if not 0 < payload['completed_round'] < 66:
        raise RuntimeError('Expected an incomplete R4 pilot at a verified complete boundary')
    record = dict(at=datetime.now().isoformat(), resumed_from=payload['completed_round'],
                  target_round=66, unchanged_runtime_identity=True,
                  original_exit_code=3221226505, root_cause='not established',
                  authorization='User requested resume training, monitoring and evaluation on 2026-11-07')
    (HERE / 'RESUME_RECORD.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    original = HERE / 'run_trial.py'
    code = original.read_text(encoding='utf-8')
    before = "'--run-dir',HERE/'run','--allow-candidate'])"
    if code.count(before) != 1:
        raise RuntimeError('Unexpected original controller; refusing ambiguous resume patch')
    code = code.replace(before, "'--run-dir',HERE/'run','--allow-candidate','--resume'])")
    code = code.replace(".open('w',encoding='utf-8')", ".open('a',encoding='utf-8')")
    exec(compile(code, str(original), 'exec'), {'__file__': str(original), '__name__': '__main__'})
