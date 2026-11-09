"""Read-only completed R2/R0/R1 comparison; write a new review artifact directory."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import importlib.util

from utils.analyze_critical_reward_pilot import ROOT,read,lines,summarize_group


def event_count(rate,n):
    count=round(rate*n)
    if abs(rate*n-count)>1e-7:raise ValueError('nonintegral event count')
    return count


def gpu_summary(path):
    samples=[]
    for record in lines(path):
        if record.get('returncode')!=0:continue
        for row in record.get('rows',[]):
            samples.append(dict(at=record['at'],index=int(row[0]),used=float(row[3]),free=float(row[4])))
    if not samples:return dict(status='UNMEASURED'),[]
    indices={r['index'] for r in samples}
    if len(indices)!=1:raise ValueError('review each GPU separately')
    start=datetime.fromisoformat(samples[0]['at'])
    for row in samples:row['seconds']=(datetime.fromisoformat(row['at'])-start).total_seconds()
    steady=[r for r in samples if r['seconds']>=30]
    if not steady:steady=samples
    return dict(status='MEASURED_WHOLE_GPU_NOT_PROCESS_ATTRIBUTION',samples=len(samples),
        observed_seconds=samples[-1]['seconds'],first_used_MiB=samples[0]['used'],
        peak_used_MiB=max(r['used'] for r in samples),
        after_30s_used_min_MiB=min(r['used'] for r in steady),
        after_30s_used_max_MiB=max(r['used'] for r in steady),
        after_30s_used_median_MiB=statistics.median(r['used'] for r in steady),
        last_used_MiB=samples[-1]['used']),samples


def analyze(root):
    plan=read(root/'control/experiment_plan.json')
    previous=ROOT/plan['predecessor']
    groups={g:summarize_group(previous,g) for g in ('R0','R1')}
    groups['R2']=summarize_group(root,'R2')
    if not all(groups.values()):raise ValueError('all three fixed-end evaluations must be completed')
    configs={g:read((root if g=='R2' else previous)/'configs'/f'{g}_B0_s42.json') for g in groups}
    for g in ('R0','R1'):
        for key in ('red_checkpoint','case_manifest','seed','model_spec','algorithm','method','device'):
            if configs[g][key]!=configs['R2'][key]:raise ValueError('comparison differs: '+key)
        for key in ('c1_controller','timestep','max_steps'):
            if configs[g]['task'][key]!=configs['R2']['task'][key]:raise ValueError('task differs: '+key)
    rows=[];contrasts={}
    for g,summary in groups.items():
        for team,item in summary['teams'].items():
            n=item['n_measured_pairs']
            if n!=24 or item['measurement_coverage']!=1:raise ValueError('incomplete paired coverage')
            if item['measured_pair_identities']!=groups['R2']['teams'][team]['measured_pair_identities']:
                raise ValueError('case/seed/attempt mismatch')
            rows.append(dict(group=g,team=team,n=n,
                raw_danger=event_count(item['raw_critical_rate_C2'],n),
                event_qualified_danger=event_count(item['qualified_critical_rate_C2'],n),
                blue_event_eligible=event_count(item['blue_eligible_rate_C2'],n),
                C1_raw_danger=event_count(item['raw_critical_rate_C1'],n),
                C1_event_qualified_danger=event_count(item['qualified_critical_rate_C1'],n),
                red_both_reached=event_count(item['red_success_rates']['C2'],n),
                candidate_bad_rate_screen_pass=(1-item['blue_eligible_rate_C2']<=configs[g]['evaluation']['selection_max_bad_rate']+1e-9),
                mean_TET_s=item['mean_tet_C2'],mean_TIT_s=item['mean_tit_C2']))
    for baseline in ('R0','R1'):
        contrasts[baseline]={}
        for team in ('A','B'):
            new,old=groups['R2']['teams'][team],groups[baseline]['teams'][team]
            fields=('raw_critical_rate_C2','qualified_critical_rate_C2','blue_eligible_rate_C2',
                    'delta_raw_critical_rate','delta_qualified_critical_rate','mean_tet_C2','mean_tit_C2')
            contrasts[baseline][team]={name:new[name]-old[name] for name in fields}
            contrasts[baseline][team]['C1_qualified_drift']=new['qualified_critical_rate_C1']-old['qualified_critical_rate_C1']
    operation=read(root/'evaluations/R2_gate_retry1/operation.json')
    gpu,samples=gpu_summary(root/'evaluations/R2_gate_retry1/gpu_memory.jsonl')
    result=dict(scope='Weak-red engineering, one outer seed; A/B dependent; no formal superiority claim',
        all_fixed_evaluations_complete=True,rows=rows,groups=groups,R2_minus_baseline=contrasts,
        qualification_scope='Event eligibility only; speed/lane/control quality remain separate',
        all_R2_teams_pass_existing_bad_rate_screen=all(r['candidate_bad_rate_screen_pass'] for r in rows if r['group']=='R2'),
        operating_mode_limitation='R2 retry reuses idle Town05; R0/R1 completed with per-episode reload. Task/configuration are fixed; report baseline drift and do not claim an isolated causal reward effect.',
        R2_operation=operation,gpu=gpu)
    return result,samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/adversarial_u1000_critical_reward_20261101_v4'))
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();root=args.root if args.root.is_absolute() else ROOT/args.root
    result,samples=analyze(root)
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if samples:
        with (output/'gpu_memory_samples.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=('at','index','seconds','used','free'))
            writer.writeheader();writer.writerows(samples)
    if samples and importlib.util.find_spec('matplotlib') is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(8,3))
        axis.plot([r['seconds']/60 for r in samples],[r['used']/1024 for r in samples],linewidth=1.5)
        axis.set(xlabel='Evaluation elapsed time (minutes)',ylabel='Whole GPU memory used (GiB)',
                 title='R2 frozen evaluation: Town05 reuse')
        axis.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'gpu_memory.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(rows=result['rows'],gpu=result['gpu'],output=str(output)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
