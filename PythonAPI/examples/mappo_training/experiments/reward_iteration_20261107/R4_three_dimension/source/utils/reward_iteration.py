"""Auditable 66-update local reward experiments; no LLM or quota dependency."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_rows(path):
    result = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                pass  # The active writer may have an incomplete final line.
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def average(values):
    return statistics.mean(values) if values else 0.


def summarize(records):
    result = {}
    for team in sorted({r.get('team') for r in records if r.get('team')}):
        cohort = [r for r in records if r.get('team') == team]
        valid = [r for r in cohort if r.get('valid') is True]
        roles = {}
        for aid in ('2', '3'):
            values = [r.get('individual_adversarial_metrics', {}).get(aid, {}) for r in valid]
            roles[aid] = dict(
                exposure=average([v.get('critical_exposure_s', 0.) for v in values]),
                qualified_event=average([v.get('qualified_primary_event_credit', 0.) for v in values]),
                qualified_exposure=average([v.get('critical_exposure_s', 0.) if v.get('qualified_primary_event_credit') else 0. for v in values]))
        result[team] = dict(n=len(valid), attempted=len(cohort),
            cases=sorted(r.get('case_key', '') for r in valid), roles=roles,
            bad_rate=average([float(not r.get('blue_eligible', False)) for r in valid]),
            collision_rate=average([float(r.get('safety', {}).get('collision', False)) for r in valid]))
    return result


def improved(baseline, candidate):
    """Provisional engineering screen, not statistical proof or safety approval."""
    teams = {}
    for team, data in candidate.items():
        base = baseline.get(team, {})
        complete = data['n'] >= 24 and data['n'] == data['attempted'] and data.get('cases') == base.get('cases')
        per_role = {}
        for aid in ('2', '3'):
            old = base.get('roles', {}).get(aid, {}).get('exposure', 0.)
            new = data['roles'][aid]
            per_role[aid] = new['exposure'] >= max(old * 1.25, old + .05) and new['qualified_event'] >= .125
        teams[team] = dict(passed=complete and all(per_role.values()), complete=complete, roles=per_role)
    return dict(passed=any(t['passed'] for t in teams.values()), teams=teams,
        rule='same 24-case cohort; each role exposure +25% and +0.05s; each role qualified events >=3/24; no hard bad-rate gate per user preference')


def snapshot_sources(directory):
    hashes = {}
    for folder in ('environment', 'training', 'models', 'evaluation', 'utils'):
        for path in (ROOT / folder).glob('*.py'):
            relative = path.relative_to(ROOT)
            target = directory / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(directory / 'source_hashes.json', hashes)


def progress(root, version, phase, five_hour=False):
    directory = root / version
    metrics = [r for r in read_rows(directory / 'run/metrics.jsonl') if 'teams' in r]
    report = dict(at=now(), version=version, phase=phase,
        completed_updates=metrics[-1]['completed_round'] if metrics else 0, target_updates=66, teams={})
    for team in ('A', 'B'):
        report['teams'][team] = {}
        for name, window in [('previous10', metrics[-20:-10]), ('latest10', metrics[-10:])]:
            if not window:
                continue
            values = [r['training_window'][team] for r in window]
            report['teams'][team][name] = dict(
                b=average([v['b'] for v in values]),
                exposure=average([v.get('critical_exposure_s', 0.) for v in values]),
                safe_event=average([v.get('safe_critical_event', 0.) for v in values]),
                rewards={k: average([v.get('reward_components_mean', {}).get(k, 0.) for v in values])
                    for k in ('base_soft', 'base_hard', 'individual_adversarial')},
                explained_variance=average([r['teams'][team]['explained_variance'] for r in window]))
    output = root / 'control' / ('five_hour' if five_hour else 'twenty_minute')
    write(output / 'latest.json', report)
    with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + '\n')
    return report


def execute(root, version, phase, module, args, clocks):
    log = root / version / (phase + '_console.log')
    command = [sys.executable, '-X', 'utf8', '-u', '-m', module, *map(str, args)]
    write(root / 'control/state.json', dict(at=now(), status='RUNNING', version=version, phase=phase, command=command))
    with log.open('w', encoding='utf-8') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        write(root / 'control/worker.json', dict(pid=child.pid, at=now(), command=command))
        while child.poll() is None:
            current = time.monotonic()
            if current >= clocks['twenty']:
                progress(root, version, phase)
                clocks['twenty'] = current + 1200
            if current >= clocks['five']:
                progress(root, version, phase, five_hour=True)
                write(root / 'control/RESUME.json', dict(at=now(), version=version, phase=phase, pid=child.pid,
                    instruction='Inspect worker identity and state; never duplicate or automatically resume a failed CARLA operation.'))
                clocks['five'] = current + 18000
            time.sleep(5)
    if child.returncode:
        raise RuntimeError(f'{version}/{phase} exited {child.returncode}; see {log}')
    progress(root, version, phase)


def run(root):
    root = root.resolve()
    if (root / 'control/controller.started').exists():
        raise RuntimeError('Controller already started; inspect state instead of starting a duplicate')
    (root / 'control/controller.started').write_text(now())
    clocks = dict(twenty=0, five=time.monotonic()+18000)
    # Baseline evaluation was launched by the agent. Do not compete for its world.
    base_manifest = root / 'base_u0/evaluation/manifest.json'
    deadline = time.monotonic()+3600
    while True:
        if base_manifest.exists():
            status = json.loads(base_manifest.read_text())['status']
            if status == 'COMPLETED':
                break
            if status == 'FAILED':
                raise RuntimeError('Baseline evaluation failed; inspect its error before continuing')
        if time.monotonic() > deadline:
            raise RuntimeError('Baseline evaluation did not finish within one hour')
        time.sleep(5)
    baseline = summarize(read_rows(root / 'base_u0/evaluation/cases.jsonl'))
    write(root / 'base_u0/analysis.json', baseline)
    config = json.loads((root / 'R3a_alpha010/config.json').read_text())
    versions = [
        ('R3a_alpha010', .1, 3., 5., 'Reduce base-soft weight only, 0.3 to 0.1, to test objective competition.'),
        ('R3b_alpha003', .03, 3., 5., 'Previous 66-update candidate did not pass the exposure screen; further reduce base-soft weight while retaining original hard costs.'),
        ('R3c_exposure005', .03, 5., 5., 'Low base weighting was insufficient; increase bounded continuous TTC exposure reward from 3 to 5, preserving event timing and hard costs.'),
        ('R3d_event007', .03, 5., 7., 'Exposure weighting was insufficient; test stronger once-per-role event credit, 5 to 7. Positive budget is checked by the existing validator.')]
    for version, alpha, exposure, bonus, reason in versions:
        directory = root / version
        directory.mkdir(exist_ok=True)
        config['task']['team_reward'].update(base_soft_weight=alpha, critical_weight=exposure, primary_event_reward=bonus)
        write(directory / 'config.json', config)
        (directory / 'REASON.md').write_text(reason+'\nFresh source-policy initialization; B0, seed42, 66 updates, LR horizon66. Selection comparisons are exploratory single-seed screens.\n', encoding='utf-8')
        snapshot_sources(directory)
        execute(root, version, 'validation', 'training.train_mut_mappo', ['--config', directory/'config.json', '--validate-only'], clocks)
        execute(root, version, 'training', 'training.train_mut_mappo', ['--config', directory/'config.json', '--run-dir', directory/'run', '--allow-candidate'], clocks)
        status = json.loads((directory/'run/session/status.json').read_text())
        if status.get('completed_round') != 66:
            raise RuntimeError('Training did not reach exactly 66 updates')
        execute(root, version, 'evaluation', 'evaluation.evaluate_mut_mappo', ['--config', directory/'config.json', '--checkpoint', directory/'run/latest.pt', '--team', 'both', '--split', 'selection', '--conditions', 'C2', '--output-dir', directory/'evaluation'], clocks)
        result = summarize(read_rows(directory/'evaluation/cases.jsonl'))
        verdict = improved(baseline, result)
        write(directory/'analysis.json', dict(at=now(), baseline=baseline, candidate=result, verdict=verdict))
        if verdict['passed']:
            # Test has not been used for tuning. It needs prepared reference files.
            write(root/'control/state.json', dict(at=now(), status='CANDIDATE_READY_FOR_HELDOUT', version=version,
                instruction='Selection screen passed; prepare heldout references and evaluate baseline and frozen candidate on test. No test-driven retuning. Not yet a verified success.'))
            return
    write(root/'control/state.json', dict(at=now(), status='NEEDS_TRAJECTORY_DIAGNOSIS',
        instruction='All four parameter candidates failed. Do not repeat blindly or claim success. Inspect per-case opportunity, reward gates and trajectories before designing shaping.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root)
    except BaseException as error:
        write(args.root/'control/state.json', dict(at=now(), status='FAILED', error=repr(error)))
        raise
