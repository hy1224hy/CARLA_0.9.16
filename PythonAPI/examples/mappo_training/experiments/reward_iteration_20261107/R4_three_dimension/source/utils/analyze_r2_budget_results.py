"""File-only review of all four registered R2 milestones, with signed inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def analyze(root, output):
    status = read(root/'control/pipeline_status.json')
    if status['status'] not in {'COMPLETED_FOR_ANALYSIS', 'EVIDENCE_COMPLETE', 'COMPLETED'}:
        raise ValueError('the registered pipeline has not completed')
    plan = read(root/'control/experiment_plan.json')
    if sha(plan['config']) != plan['config_sha256']:
        raise ValueError('configuration changed')
    nodes = read(root/'control/milestones.json')
    if [n['target_train_ticks'] for n in nodes] != [0,24000,48000,72000]:
        raise ValueError('all four predeclared milestones are required')
    inputs, series, initial = {}, [], {}
    for node in nodes:
        name = 'frozen_'+str(node['target_train_ticks']).zfill(6)
        marker = root/'control'/(name+'_completed.json')
        done = read(marker)
        if done['returncode'] != 0:
            raise ValueError('evaluation did not exit normally')
        report_path = root/'evaluations'/name/'analysis/critical_report.json'
        expected = next(v for k, v in done['artifact_hashes'].items() if k.replace('\\','/').endswith('critical_report.json'))
        if sha(report_path) != expected or sha(node['checkpoint']) != node['sha256']:
            raise ValueError('signed evaluation/model SHA differs')
        inputs[str(report_path.relative_to(root))] = expected
        report = read(report_path)
        cohort = report['reports'][0]
        if cohort['manifest']['checkpoint_sha256'] != node['sha256']:
            raise ValueError('evaluation is not of this milestone')
        for team in ('A','B'):
            legacy = next(r for r in report['team_rows'] if r['team'] == team)
            quality = next(r for r in report['interaction_quality_team_rows'] if r['team'] == team)
            if legacy['n_measured_pairs'] != 24 or legacy['measurement_coverage'] != 1:
                raise ValueError('missing paired cases')
            cases = [r for r in cohort['cases'] if r['team'] == team and r['condition'] == 'C2']
            if len(cases) != 24:
                raise ValueError('C2 denominator differs')
            key = lambda r:(r['case_key'],r['seed'],r['paired_attempt'])
            if node['target_train_ticks'] == 0:
                initial[team] = {key(r):r for r in cases}
            if {key(r) for r in cases} != set(initial[team]):
                raise ValueError('milestones do not use identical paired cases')
            driving = {}
            for aid in ('2','3'):
                failures, unknown = Counter(), Counter()
                for case in cases:
                    for check, value in case['interaction_quality']['driving_quality']['per_blue'][aid]['checks'].items():
                        if value is False:
                            failures[check] += 1
                        elif value is None:
                            unknown[check] += 1
                driving[aid] = dict(failed_checks_overlapping=dict(failures),unknown_checks=dict(unknown),
                    **{field:mean([c.get('driving_metrics',{}).get(aid,{}).get(field) for c in cases])
                       for field in ('mean_speed_mps','above_preferred_speed_fraction','preferred_speed_fraction',
                                     'reference_lateral_p95_m','actual_steer_rate_p95_per_second')})
            transitions = {}
            for field in ('predicted_conflict','observed_contact_free_resolved','joint_normal_resolved'):
                pairs = [(initial[team][key(c)]['interaction_quality'][field],c['interaction_quality'][field]) for c in cases]
                transitions[field] = dict(added=sum(a is False and b is True for a,b in pairs),
                    removed=sum(a is True and b is False for a,b in pairs),
                    unchanged_true=sum(a is True and b is True for a,b in pairs),unknown=sum(a is None or b is None for a,b in pairs))
            row = dict(**node, team=team, n=24, legacy=legacy, interaction_quality=quality,
                blue_event_eligible_count=sum(c['blue_eligible'] for c in cases),
                red_both_reached_count=sum(c['red_success'] for c in cases),
                raw_critical_count=sum(c['critical_interaction']['critical_event'] for c in cases),
                legacy_qualified_critical_count=sum(c['blue_eligible'] and c['critical_interaction']['critical_event'] for c in cases),
                blue_violations_overlapping={event:sum(c['safety'].get(event) is True for c in cases)
                    for event in ('collision','offroad','road_rule','task_violation')},
                driving=driving,paired_to_initial=transitions)
            series.append(row)
    result = dict(scope=plan['scope'],inputs_sha256=inputs,source_sha256=sha(Path(__file__)),series=series,
        limitations=['one outer seed, A/B dependent; gate cases used for development',
            'old24k has a different LR horizon; not a pure budget comparator',
            'observational TTC/clearance and preferences are not causal blame or formal driving qualification'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# R2 72k fixed-reward engineering review','',
        'All four registered milestones; each row has 24 paired cases. A/B are not independent seeds.','',
        '| Target/actual train ticks | Team | Raw danger | Legacy G×H | Observed resolved/no-contact | Full driving | Joint goal | Blue eligible | Red both reached |',
        '|---|---|---|---|---|---|---|---|---|']
    def count(rate):
        return 'unmeasured' if rate is None else str(round(24*rate))+'/24'
    for r in series:
        q=r['interaction_quality']
        lines.append('| '+ ' | '.join([str(r['target_train_ticks'])+'/'+str(r['actual_train_ticks']),r['team'],
            str(r['raw_critical_count'])+'/24',str(r['legacy_qualified_critical_count'])+'/24',
            count(q['observed_contact_free_resolved_C2']),count(q['driving_quality_rate_C2']),
            count(q['joint_normal_resolved_C2']),str(r['blue_event_eligible_count'])+'/24',
            str(r['red_both_reached_count'])+'/24'])+' |')
    lines += ['', 'See summary.json for paired gains/losses, contact/censoring, C1 differences, and overlapping per-blue driving failures.',
              'Do not infer convergence or reward superiority from this one-seed curve.']
    (output/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'experiments/adversarial_u1000_R2_budget_20261104_v1')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.root.resolve(),args.output.resolve())
    print(json.dumps(dict(status='COMPLETE',rows=len(result['series']),output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    main()
