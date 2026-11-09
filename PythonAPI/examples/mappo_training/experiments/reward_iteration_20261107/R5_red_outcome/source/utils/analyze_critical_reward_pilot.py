"""File-only descriptive R1/R0 comparison; A/B are not independent seeds."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    if not path.exists():return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
            if l.endswith('\n') and l.strip()]


def mean(values):return statistics.mean(values) if values else None


def summarize_group(root,group):
    from evaluation.mut_mappo_metrics import _safety_reasons
    marker=root/'control'/f'{group}_gate_completed.json'
    if not marker.exists():return None
    completion=read(marker)
    reports=[name for name in completion['artifact_hashes'] if name.replace('\\','/').endswith('analysis/critical_report.json')]
    if len(reports)!=1:raise ValueError('completed gate must identify one independent report')
    report_path=root/reports[0]
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!=completion['artifact_hashes'][reports[0]]:
        raise ValueError('completed independent report changed')
    report=read(report_path)
    if len(report['reports'])!=1:raise ValueError('Do not pool multiple evaluation cohorts')
    source=report['reports'][0]; cases=source['cases']; result={}
    for row in report['team_rows']:
        team=row['team']; pairs=source['paired']['teams'][team]['pairs']
        keys={(p['case_key'],p['seed'],p['paired_attempt']) for p in pairs}
        subset=[r for r in cases if r['team']==team and (r['case_key'],r['seed'],r['paired_attempt']) in keys]
        c2=[r for r in subset if r['condition']=='C2']
        if len(c2)!=len(pairs):raise ValueError('C2 denominator differs from measured pairs')
        driving={}
        for aid in ('2','3'):
            driving[aid]={}
            for label,records in (('all_C2',c2),('event_eligible_C2',[r for r in c2 if r['blue_eligible']])):
                driving[aid][label]={}
                for metric in ('mean_speed_mps','above_preferred_speed_fraction','preferred_speed_fraction',
                    'reference_lateral_p95_m','actual_steer_rate_p95_per_second',
                    'unblocked_low_speed_fraction','abs_acceleration_p95_mps2','abs_jerk_p95_mps3'):
                    values=[r.get('driving_metrics',{}).get(aid,{}).get(metric) for r in records]
                    values=[v for v in values if v is not None]
                    driving[aid][label][metric]={'case_mean':mean(values),'n_measured':len(values)}
        item=dict(row)
        item['measured_pair_identities']=sorted(keys)
        item.update(blue_violation_episode_counts_overlapping={name:sum(name in _safety_reasons(r) for r in c2)
                    for name in ('collision','offroad','road_rule','task_violation')},
            red_success_rates={c:mean([int(r['red_success']) for r in subset if r['condition']==c]) for c in ('C0','C1','C2')},
            mean_event_count_C2=mean([len(r['critical_interaction']['events']) for r in c2]),
            mean_event_count_C1=mean([len(r['critical_interaction']['events']) for r in subset if r['condition']=='C1']),
            driving=driving)
        result[team]=item
    costs={}
    for r in reversed(lines(root/'runs'/group/'actual_cost_audit.jsonl')):
        if 'actual_counters' in r:costs['train_process']=r['actual_counters'];break
    counts=collections.Counter()
    for r in lines(root/'references'/group/'world_tick_audit.jsonl'):counts[r['kind']]+=r.get('ticks',1)
    costs['reference_explicit_ticks']=dict(counts)
    costs['independent_eval_ticks_all_attempts']={k:sum(r.get('tick_counts',{}).get(k,0) for r in cases)
                                                for k in ('total','decision','setup')}
    failed_roots=completion.get('retained_failed_evaluation_roots',
        [completion['retained_failed_evaluation_root']] if completion.get('retained_failed_evaluation_root') else [])
    if failed_roots:
        failed_rows=[r for path in failed_roots for r in lines(root/path/'evaluations/eval_000/cases.jsonl')]
        costs['retained_failed_eval_committed_tick_lower_bound']={
            k:sum(r.get('tick_counts',{}).get(k,0) for r in failed_rows)
            for k in ('total','decision','setup')}
        costs['failed_eval_cost_limitation']='Native interruption may leave constructor/cleanup ticks unrecorded'
    return dict(teams=result,costs=costs,report=str(report_path),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())


def analyze(root):
    groups={g:summarize_group(root,g) for g in ('R1','R0')}
    result=dict(status='BOTH_EVALUATED' if all(groups.values()) else 'PARTIAL_NOT_COMPARABLE',
        scope='Weak-red descriptive engineering pilot; one outer seed, A/B dependent; no formal algorithm claim',
        qualification_scope='Event eligibility only; speed/lane/control quality reported separately',
        driving_statistic='Means of per-case measurements; a mean of case P95 is not a trajectory-wide P95',
        groups=groups,comparisons=None)
    if all(groups.values()):
        cfg={g:read(root/'configs'/f'{g}_B0_s42.json') for g in groups}
        for key in ('red_checkpoint','case_manifest','seed'):
            if cfg['R0'][key]!=cfg['R1'][key]:raise ValueError('comparison identity differs: '+key)
        if cfg['R0']['task'].get('c1_controller')!=cfg['R1']['task'].get('c1_controller'):
            raise ValueError('normal C1 policy differs between reward groups')
        result['comparisons']={}
        for team in ('A','B'):
            a,b=groups['R1']['teams'][team],groups['R0']['teams'][team]
            if a['measured_pair_identities']!=b['measured_pair_identities']:
                raise ValueError('measured case/seed/attempt identities differ')
            fields=('qualified_critical_rate_C2','raw_critical_rate_C2','delta_qualified_critical_rate',
                    'delta_raw_critical_rate','blue_eligible_rate_C2','mean_tet_C2','mean_tit_C2')
            result['comparisons'][team]={f'R1_minus_R0_{k}':a[k]-b[k]
                if a[k] is not None and b[k] is not None else None for k in fields}
            result['comparisons'][team]['C1_qualified_rate_drift']=a['qualified_critical_rate_C1']-b['qualified_critical_rate_C1']
            result['comparisons'][team]['C1_raw_rate_drift']=a['raw_critical_rate_C1']-b['raw_critical_rate_C1']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('experiments/adversarial_u1000_critical_reward_20261101_v3'))
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();root=args.root if args.root.is_absolute() else ROOT/args.root
    result=analyze(root)
    if args.output_dir:
        output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
        output.mkdir(parents=True,exist_ok=False)
        (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
