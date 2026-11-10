"""Explicit, reproducible reward-repair pilots; never overwrite prior studies.

Run from PythonAPI/examples with ``python -m highwaytest.straight.repair_study``.
The parent normal driver is requalified in the current environment, not silently
given a new source identity. Parent identity and unchanged actor hash are saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from .config import StraightConfig
from .runner import (load_state, model_from_state, evaluate_model, evaluate,
                     source_digest, runtime_versions, save)


def actor_digest(actor):
    digest = hashlib.sha256()
    for key, value in sorted(actor.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(root, parent, steps):
    root, parent = Path(root), Path(parent)
    if root.exists():
        raise FileExistsError(root)
    state = load_state(parent)
    if (state['config']['level'] != 0 or not state.get('qualified')
            or state.get('attack_level') is not None or state.get('repertoire')):
        raise ValueError('Only a qualified ordinary-driving initializer is allowed')
    cfg = StraightConfig(level=0, seed=state['config']['seed'],
                         hidden=state['config']['hidden'], obs_dim=state['config']['obs_dim'])
    for key in ('duration', 'policy_frequency', 'simulation_frequency',
                'target_speed', 'target_time_gap'):
        if state['config'][key] != getattr(cfg, key):
            raise ValueError('Parent physical contract differs: ' + key)
    model = model_from_state(state)
    torch.set_num_threads(1)
    gate = evaluate_model(model, cfg, 'gate', 32)
    lane = evaluate_model(model, cfg, 'gate', 32, demonstration=True)
    qualified = (gate['overall']['success_rate'] >= .9
                 and gate['overall']['invalid_rate'] <= .05
                 and gate['overall']['collision_rate'] <= .05
                 and lane['overall']['success_rate'] >= .9
                 and lane['overall']['lane_change_rate'] >= .5)
    if not qualified:
        raise ValueError('Requalification failed: ' + json.dumps(
            dict(driving=gate['overall'], lane=lane['overall'])))
    root.mkdir(parents=True)
    (root / 'configs').mkdir()
    (root / 'base').mkdir()
    origin = dict(path=str(parent.resolve()), file_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                  source_sha256=state['source_sha256'], actor_sha256=actor_digest(state['actor']))
    new_state = dict(state, config=cfg.to_dict(), source_sha256=source_digest(),
                     packages=runtime_versions(), qualification=gate,
                     requalification=dict(parent=origin, driving=gate, lane=lane))
    save(root / 'base' / 'driver.pt', new_state)
    manifest = dict(source_sha256=source_digest(), parent=origin, gate=gate['overall'],
                    lane_gate=lane['overall'], budget='A+B environment decisions',
                    initialization='unchanged ordinary-driving actor; current-source gate requalification')
    (root / 'study.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for level in (1, 2, 3):
        train_cfg = replace(cfg, level=level, method='independent', total_steps=steps,
                            lr=1e-4, entropy_coef=.003, clip=.15,
                            geometry_weight=.5, approach_weight=3., potential_ly=1.5,
                            eval_interval=20000, lane_cross_bonus=0.)
        train_cfg.validate()
        name = f'L{level}_independent_s{cfg.seed}'
        (root / 'configs' / (name + '.json')).write_text(
            json.dumps(train_cfg.to_dict(), indent=2), encoding='utf-8')
    return manifest


def train_all(root):
    root = Path(root).resolve()
    (root / 'logs').mkdir(exist_ok=True)
    def job(config):
        output = root / 'train' / config.stem
        cmd = [sys.executable, '-m', 'highwaytest.straight', 'train',
               '--config', str(config), '--output', str(output)]
        if (output / 'latest.pt').exists():
            cmd += ['--resume', str(output / 'latest.pt')]
        else:
            cmd += ['--initialize-from', str(root / 'base' / 'driver.pt')]
        with (root / 'logs' / (config.stem + '.log')).open('a', encoding='utf-8') as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        return dict(run=config.stem, exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(job, sorted((root / 'configs').glob('*.json'))))
    (root / 'batch_status.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if any(row['exit_code'] for row in results):
        raise RuntimeError('Training failed; inspect study logs: ' + str(results))
    return results


def status(root):
    rows = []
    for run in sorted((Path(root) / 'train').glob('L*')):
        path = run / 'validation.jsonl'
        validations = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                validations.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not validations:
            rows.append(dict(run=run.name, status='starting'))
            continue
        last = validations[-1]
        best = max(r['reports'][team]['overall']['attack_success_rate']
                   for r in validations for team in ('A', 'B'))
        rows.append(dict(run=run.name, steps=last['train_steps'], best_asr=best,
                         latest={team: last['reports'][team]['overall'] for team in ('A', 'B')}))
    return rows


def final_evaluate(root):
    rows = []
    for run in sorted((Path(root) / 'train').glob('L*')):
        result = evaluate(run / 'selected.pt', 'test', 100, run / 'test', gif=True)
        families = {}
        for family in ('follow', 'adjacent'):
            episodes = [e for e in result['episodes'] if e['family'] == family]
            families[family] = {key: sum(float(e.get(key, 0) or 0) for e in episodes) / len(episodes)
                                for key in ('attack_success', 'collision', 'invalid',
                                            'grade_violation', 'cut_count', 'blocking_time')}
        rows.append(dict(run=run.name, overall=result['overall'], families=families))
    (Path(root) / 'final_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'status', 'evaluate'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--steps', type=int, default=200000)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.parent:
            parser.error('prepare requires --parent')
        result = prepare(args.root, args.parent, args.steps)
    else:
        result = {'train': train_all, 'status': status, 'evaluate': final_evaluate}[args.command](args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
