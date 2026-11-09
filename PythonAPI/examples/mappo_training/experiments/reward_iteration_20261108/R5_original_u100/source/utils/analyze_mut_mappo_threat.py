"""Offline paired threat diagnostics from independent MutMAPPO evaluation summaries.

No CARLA/PyTorch dependency. Does not change selection, checkpoints or training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))
              and not isinstance(v, bool) and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def analyze_report(report, source, label=None):
    meta = report.get('metadata', {})
    if (report.get('schema_version') != 'mut_mappo_evaluation_v1'
            or report.get('evidence') != 'independent_evaluation'
            or meta.get('status') != 'COMPLETED'):
        raise ValueError(f'{source}: requires a completed independent evaluation summary')
    teams = report.get('paired', {}).get('teams', {})
    if not teams:
        raise ValueError(f'{source}: no paired teams; evaluate C1 and C2 first')
    config_name = str(meta.get('config_path') or source).replace('\\', '/').split('/')[-1]
    label = label or Path(config_name).stem
    candidates = {r['team']: r for r in report.get('selection', {}).get('candidates', [])}
    summaries, cases = [], []
    for team, data in teams.items():
        included = data.get('included')
        if not isinstance(included, list) or len(included) != data.get('n_common_valid'):
            raise ValueError(f'{source}/{team}: paired case count mismatch')
        records, seen = [], set()
        for pair in included:
            success = pair.get('red_success', {})
            if any(type(success.get(c)) is not bool for c in ('C1', 'C2')):
                raise ValueError(f'{source}/{team}: missing C1/C2 success flags')
            eligible = pair.get('blue_eligible_pair')
            if type(eligible) is not bool:
                raise ValueError(f'{source}/{team}: missing paired blue qualification')
            c0 = success.get('C0')
            if c0 is not None and type(c0) is not bool:
                raise ValueError('C0 success must be boolean or missing')
            key = (pair['case_key'], pair.get('paired_attempt', 0))
            if key in seen:
                raise ValueError(f'{source}/{team}: duplicate paired case {key}')
            seen.add(key)
            c1, c2 = success['C1'], success['C2']
            delta = int(c1) - int(c2)
            record = dict(report=label, source=str(source), split=meta.get('split'), team=team,
                case_key=key[0], paired_attempt=key[1], family=pair.get('family'),
                c0_success=c0, c1_success=c1, c2_success=c2, blue_eligible_pair=eligible,
                safe_contribution=delta if eligible else 0,
                outcome='new_failure' if delta > 0 else 'recovery' if delta < 0 else 'unchanged',
                arrival_delay_s=pair.get('arrival_delay_C2_minus_C1_both_success_seconds'),
                completion_loss=pair.get('red_completion_difference_C1_minus_C2'))
            records.append(record)
        n = len(records)
        safe_new = sum(r['outcome'] == 'new_failure' and r['blue_eligible_pair'] for r in records)
        safe_recovery = sum(r['outcome'] == 'recovery' and r['blue_eligible_pair'] for r in records)
        c0_known = all(r['c0_success'] is not None for r in records) and n > 0
        c2 = data.get('conditions', {}).get('C2', {})
        candidate = candidates.get(team, {})
        both_success = [r for r in records if r['c1_success'] and r['c2_success']]
        safe_both_success = [r for r in both_success if r['blue_eligible_pair']]
        row = dict(report=label, source=str(source), split=meta.get('split'), team=team,
            completed_round=meta.get('checkpoint_completed_round'), n_common_valid=n,
            n_excluded=len(data.get('excluded', [])),
            safe_new_failures=safe_new, safe_recoveries=safe_recovery,
            safe_net_gain=(safe_new-safe_recovery)/n if n else None,
            unsafe_new_failures=sum(r['outcome'] == 'new_failure' and not r['blue_eligible_pair'] for r in records),
            unsafe_recoveries=sum(r['outcome'] == 'recovery' and not r['blue_eligible_pair'] for r in records),
            c0_failure_cases=sum(not r['c0_success'] for r in records) if c0_known else None,
            c1_incremental_failure_cases=sum(r['c0_success'] and not r['c1_success'] for r in records) if c0_known else None,
            reference_success_cases=sum(r['c1_success'] for r in records),
            safe_reference_success_cases=sum(r['c1_success'] and r['blue_eligible_pair'] for r in records),
            c1_success_rate=mean([int(r['c1_success']) for r in records]),
            c2_success_rate=mean([int(r['c2_success']) for r in records]),
            blue_bad_rate=c2.get('b'),
            safe_efficiency_utility=c2.get('safe_efficiency_utility'),
            both_success_pairs=len(both_success), arrival_delay_s=mean([r['arrival_delay_s'] for r in both_success]),
            safe_both_success_pairs=len(safe_both_success),
            safe_arrival_delay_s=mean([r['arrival_delay_s'] for r in safe_both_success]),
            completion_loss=mean([r['completion_loss'] for r in records]),
            safe_completion_loss=mean([r['completion_loss'] for r in records if r['blue_eligible_pair']]),
            qualified=candidate.get('qualified'), rejection=candidate.get('rejection'))
        for kind, count in c2.get('blue_violation_episode_counts', {}).items():
            row['blue_violation_' + kind] = count
        summaries.append(row)
        cases.extend(records)
    return summaries, cases


def discover_reports(root):
    """Use the latest recorded boundary per run; no mixing retries or old boundaries."""
    root = Path(root).resolve()
    read_json(root / 'experiment.json')  # Require an actual experiment, not an arbitrary tree.
    latest = {}
    for file in sorted((root / 'results').glob('*.json')):
        result = read_json(file)
        name = result['run_id']
        if name not in latest or result['completed_round'] > latest[name][1]['completed_round']:
            latest[name] = (file, result)
    found = []
    for file, record in latest.values():
        for field in ('selection_dir', 'test_dir'):
            raw = record.get(field)
            if not raw:
                continue
            normalized = str(raw).replace('\\', '/')
            saved = Path(normalized) / 'summary.json'
            # Rebase saved absolute server paths when an experiment was downloaded/moved.
            relocated = root / 'evaluations' / file.stem / normalized.rstrip('/').split('/')[-1] / 'summary.json'
            candidate = saved if saved.is_file() else relocated
            if not candidate.is_file():
                raise FileNotFoundError(f'Missing recorded report: {saved}; also tried {relocated}. Use --reports for other layouts.')
            found.append((candidate, record['run_id']))
    if not found:
        raise ValueError('No recorded evaluations. Run evaluate first, or provide --reports explicitly.')
    return found


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    esc = lambda value: html.escape('' if value is None else str(value))
    return '<table><thead><tr>' + ''.join('<th>'+esc(f)+'</th>' for f in fields) + '</tr></thead><tbody>' + ''.join(
        '<tr>'+''.join('<td>'+esc(r.get(f))+'</td>' for f in fields)+'</tr>' for r in rows) + '</tbody></table>'


def generate(inputs, output):
    summaries, cases, provenance = [], [], []
    seen = set()
    for path, label in inputs:
        path = Path(path).resolve()
        if path in seen:
            raise ValueError(f'Duplicate input report: {path}')
        seen.add(path)
        report = read_json(path)
        rows, paired = analyze_report(report, path, label)
        summaries.extend(rows); cases.extend(paired)
        meta = report['metadata']
        provenance.append(dict(path=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata=meta))
    result = dict(schema='mut_mappo_threat_diagnostics_v1', summaries=summaries, provenance=provenance,
        independent_unit='outer training seed; A/B are dependent candidates; no pooling',
        interpretation=[
            'safe_net_gain = (safe new failures - safe recoveries) / all common-valid pairs.',
            'Blue qualification is required on BOTH C1 and C2 for safe contributions.',
            'Positive arrival_delay/completion_loss means red was delayed/hindered.',
            'C0/C1 success is a single observed outcome, NOT a pre-registered reliable baseline subset.',
            'Selection and test, different reds/case lists/budgets must not be pooled.',
            'This report is descriptive, with no statistical significance or causal blame claim.',
            'Proximity/braking and eligible-only efficiency cannot be recovered from summary alone; use raw traces.',
        ])
    # Validate all inputs before creating output. Existing reports are never overwritten.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'threat_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output / 'teams.csv', summaries)
    write_csv(output / 'cases.csv', cases)
    fields = ['report', 'split', 'team', 'n_common_valid', 'blue_bad_rate', 'safe_new_failures',
              'safe_recoveries', 'safe_net_gain', 'unsafe_new_failures', 'c0_failure_cases',
              'c1_incremental_failure_cases', 'safe_arrival_delay_s', 'qualified']
    bars = []
    for row in summaries:
        text = html.escape(f"{row['report']} / {row['split']} / {row['team']}")
        n = row['n_common_valid']
        bars.append(f'<p>{text} (N={n})</p>')
        for field, color in [('safe_new_failures', '#c74637'), ('safe_recoveries', '#3181bc')]:
            count = row[field]
            width = 100 * count/n if n else 0
            bars.append(f'<div>{field}: {count}<div style="background:{color};width:{width}%;height:12px"></div></div>')
    page = '<!doctype html><meta charset="utf-8"><title>Paired threat diagnostics</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}section{max-width:900px}</style><h1>Paired threat diagnostics</h1>'
    page += '<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in result['interpretation'])+'</ul>'
    page += table(summaries, fields)+'<section><h2>Safe new failures vs recoveries (counts)</h2>'+''.join(bars)+'</section>'
    (output / 'index.html').write_text(page, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--experiment', type=Path, help='Experiment root; discover latest selection/test results for every run')
    source.add_argument('--reports', type=Path, nargs='+', help='Any completed evaluation summary JSON files')
    parser.add_argument('--labels', nargs='+', help='Optional labels, one per explicit report')
    parser.add_argument('--output-dir', required=True, type=Path, help='New output directory (never overwritten)')
    args = parser.parse_args()
    if args.labels and (not args.reports or len(args.labels) != len(args.reports)):
        parser.error('--labels requires exactly one label per --reports file')
    inputs = discover_reports(args.experiment) if args.experiment else list(zip(args.reports, args.labels or [None]*len(args.reports)))
    result = generate(inputs, args.output_dir)
    print(json.dumps(dict(output=str(args.output_dir.resolve()), teams=len(result['summaries']), reports=len(inputs)), indent=2))


if __name__ == '__main__': main()
