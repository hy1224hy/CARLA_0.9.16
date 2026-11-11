"""Seed-level aggregation, paired estimates and reproducible scientific figures."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np

METRICS=['attack_success_rate','canonical_return','collision_rate','grade_violation_rate','utility',
    'blocking_time','cut_count','repeated_cut_rate','red_delay_m','time','completed_lane_changes',
    'style_success_rate','style_speed_fraction','style_lane_fraction','front_effect_time','command_jerk_mean','collision_latency_capped']


def mean_sd(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        sample_sd=statistics.stdev(values) if len(values)>1 else None)


def paired_interval(values):
    rng=np.random.default_rng(20261111);values=np.asarray(values,dtype=float)
    if len(values)<2:return [None,None]
    bootstrap=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
    return np.quantile(bootstrap,[.025,.975]).tolist()


def save_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        if fields:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def build_report(root,output=None,allow_partial=False):
    root=Path(root).resolve();output=Path(output) if output else root/'report'
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'study.json').read_text());expected=plan['seeds'];run_rows=[];index={}
    for path in sorted((root/'audits').glob('**/metrics.json')):
        report=json.loads(path.read_text());cfg=report['config'];task=cfg['task']
        group=cfg['run_id'].rsplit('_s',1)[0]
        key=(group,report['split'],report['checkpoint'],report['condition'],report['execution'],task['seed'])
        index.setdefault(key,[]).append(report)
        run_rows.append(dict(group=group,seed=task['seed'],level=task['level'],arm=cfg['arm'],run=cfg['run_id'],
            checkpoint=report['checkpoint'],team=report['team'],split=report['split'],condition=report['condition'],
            execution=report['execution'],action_seed=report['action_seed'],**{k:report['overall'].get(k) for k in METRICS}))
    seed_rows=[]
    for key,reports in index.items():
        group,split,checkpoint,condition,execution,seed=key;cfg=reports[0]['config']
        expected_teams={'A','B'} if checkpoint not in ('selected','init') else {reports[0]['team']}
        if {r['team'] for r in reports}!=expected_teams:
            if not allow_partial:raise ValueError('Missing replica audit: '+str(key))
            continue
        counts={t:sum(r['team']==t for r in reports) for t in expected_teams}
        if len(set(counts.values()))>1:raise ValueError('Unequal stochastic replicates: '+str(key))
        row=dict(group=group,split=split,checkpoint=checkpoint,condition=condition,execution=execution,seed=seed,
            level=cfg['task']['level'],arm=cfg['arm'],beta=cfg['beta'] if cfg['mutual']!='off' else 0.,label=cfg['algorithm'].upper() if cfg['mutual']=='off' and not cfg['self_kl'] else 'Mut-PPO' if cfg['mutual']=='both' else 'one_way' if cfg['mutual']=='one_way' else 'PPO-selfKL')
        for metric in METRICS:
            values=[r['overall'].get(metric) for r in reports]
            if all(v is not None for v in values):row[metric]=statistics.mean(values)
        seed_rows.append(row)
    grouped={}
    for row in seed_rows:
        key=tuple(row[k] for k in ['group','split','checkpoint','condition','execution'])
        grouped.setdefault(key,[]).append(row)
    group_rows=[];paired=[]
    for key,rows in grouped.items():
        available=sorted(r['seed'] for r in rows)
        complete=available==sorted(expected)
        if not complete and not allow_partial:raise ValueError('Missing training seeds: '+str(key))
        for metric in METRICS:
            vals=[r[metric] for r in rows if metric in r]
            if vals:group_rows.append(dict(group=key[0],split=key[1],checkpoint=key[2],condition=key[3],execution=key[4],
                level=rows[0]['level'],arm=rows[0]['arm'],metric=metric,complete=complete,**mean_sd(vals)))
        reference=(f'L{rows[0]["level"]}_PPO',*key[1:])
        if reference in grouped and reference!=key:
            controls={r['seed']:r for r in grouped[reference]}
            for metric in METRICS:
                differences=[r[metric]-controls[r['seed']][metric] for r in rows if r['seed'] in controls and metric in r and metric in controls[r['seed']]]
                if differences:
                    low,high=paired_interval(differences)
                    paired.append(dict(group=key[0],split=key[1],checkpoint=key[2],condition=key[3],execution=key[4],metric=metric,
                        mean_difference=statistics.mean(differences),ci_low=low,ci_high=high,n=len(differences),
                        note='exploratory paired-seed bootstrap; small n; no multiplicity adjustment'))
    save_csv(output/'run_rows.csv',run_rows);save_csv(output/'seed_rows.csv',seed_rows)
    save_csv(output/'group_stats.csv',group_rows);save_csv(output/'paired_differences.csv',paired)
    nominal={(r['group'],r['split'],r['checkpoint'],r['execution'],r['seed']):r for r in seed_rows if r['condition']=='nominal'}
    deltas=[]
    for r in seed_rows:
        key=(r['group'],r['split'],r['checkpoint'],r['execution'],r['seed'])
        if r['condition']!='nominal' and key in nominal:
            deltas.append(dict(group=r['group'],seed=r['seed'],split=r['split'],checkpoint=r['checkpoint'],condition=r['condition'],
                execution=r['execution'],asr_delta=r['attack_success_rate']-nominal[key]['attack_success_rate']))
    save_csv(output/'robustness_deltas.csv',deltas)
    curves=training_curves(root,plan)
    save_csv(output/'training_curves.csv',curves)
    behavior_figures(seed_rows,output,allow_partial,expected)
    curve_figures(curves,output,expected,allow_partial)
    retention=retention_rows(plan);save_csv(output/'stability_and_cost.csv',retention)
    stability_figures(retention,output,expected,allow_partial)
    coverage=[]
    for row in plan['runs']:
        summary=Path(row['output'])/'summary.json'
        state=json.loads(summary.read_text()) if summary.exists() else {}
        coverage.append(dict(run=row['run_id'],arm=row['arm'],status=state.get('status','not_complete'),steps=state.get('train_steps'),wall_seconds=state.get('wall_seconds')))
    save_csv(output/'coverage.csv',coverage)
    lines=['# Research point one results','',f'Planned training seeds: {expected}; frozen runs: {len(plan["runs"])}.',
        f'Audits found: {len(run_rows)}; seed-condition aggregates: {len(seed_rows)}.',
        'A/B and action seeds are averaged within each training seed. Error bars use sample SD over training seeds.',
        'Missing runs are not zero-filled. Partial results are exploratory. Selected/latest, nominal/perturbed and deterministic/stochastic remain separate.',
        'Reward-ablation comparisons use canonical return and physical task metrics; the optimized return changes across ablations.',
        'Paired bootstrap intervals are exploratory at n=5; no claim of significance from bar overlap or a single best seed.','',
        '| Group | Split/checkpoint/condition/execution | Seeds | ASR mean ± sample SD |',
        '|---|---|---:|---:|']
    for r in group_rows:
        if r['metric']=='attack_success_rate':
            sd='NA' if r['sample_sd'] is None else f'{r["sample_sd"]:.4f}'
            lines.append(f'| {r["group"]} | {r["split"]}/{r["checkpoint"]}/{r["condition"]}/{r["execution"]} | {r["n"]} | {r["mean"]:.4f} ± {sd} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(audits=len(run_rows),seed_aggregates=len(seed_rows),output=str(output))


def training_curves(root,plan):
    records=[]
    for entry in plan['runs']:
        run=Path(entry['output']);path=run/'validation.jsonl'
        if not path.exists():continue
        cfg=json.loads(Path(entry['config']).read_text())
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            for metric in ['attack_success_rate','collision_rate','grade_violation_rate','canonical_return','style_success_rate','cut_count']:
                values=[row['reports'][t]['overall'].get(metric) for t in ['A','B']]
                if all(v is not None for v in values):records.append(dict(group=entry['run_id'].rsplit('_s',1)[0],arm=entry['arm'],level=entry['level'],seed=entry['seed'],steps=row['train_steps'],metric=metric,value=statistics.mean(values)))
        for filename,fields in [('episodes.jsonl',['return','canonical_return','attack_success']),
                                ('metrics.jsonl',['effective_beta_lane','mutual_kl_acceleration','mutual_kl_lane_decision','lane_decision_fraction'])]:
            source=run/filename
            if not source.exists():continue
            buckets={}
            for line in source.read_text().splitlines():
                try:r=json.loads(line)
                except ValueError:continue
                step=((r['train_steps']-1)//20000+1)*20000
                for metric in fields:
                    if metric in r:buckets.setdefault((step,metric,r['team']),[]).append(float(r[metric]))
            for step,metric in sorted({(k[0],k[1]) for k in buckets}):
                if all((step,metric,t) in buckets for t in ['A','B']):
                    value=statistics.mean(statistics.mean(buckets[step,metric,t]) for t in ['A','B'])
                    records.append(dict(group=entry['run_id'].rsplit('_s',1)[0],arm=entry['arm'],level=entry['level'],seed=entry['seed'],steps=step,metric='sampled_'+metric,value=value))
    return records


def retention_rows(plan):
    rows=[]
    for entry in plan['runs']:
        root=Path(entry['output']);path=root/'validation.jsonl'
        if not path.exists():continue
        data=[json.loads(line) for line in path.read_text().splitlines()]
        late=[r for r in data if r['train_steps']>=.75*plan['steps']]
        if not late:continue
        drawdowns=[]
        for team in ['A','B']:
            vals=[r['reports'][team]['overall']['attack_success_rate'] for r in late]
            peak=vals[0];drop=0.
            for v in vals:peak=max(peak,v);drop=max(drop,peak-v)
            drawdowns.append(drop)
        reached=next((r['train_steps'] for r in data if statistics.mean(r['reports'][t]['overall']['attack_success_rate'] for t in ['A','B'])>=.5),None)
        summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
        rows.append(dict(run=entry['run_id'],arm=entry['arm'],seed=entry['seed'],level=entry['level'],
            late_asr_mean=statistics.mean(r['reports'][t]['overall']['attack_success_rate'] for r in late for t in ['A','B']),
            late_drawdown=statistics.mean(drawdowns),first_50pct_steps=reached,wall_seconds=summary.get('wall_seconds')))
        rows[-1]['reached_50pct']=reached is not None
        rows[-1]['first_50pct_steps_capped']=reached if reached is not None else plan['steps']
    return rows


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    return plt


def behavior_figures(rows,output,partial,expected):
    if not rows:return
    plt=plotting()
    contexts=sorted({tuple(r[k] for k in ['split','checkpoint','condition','execution']) for r in rows})
    for context in contexts:
        selected=[r for r in rows if tuple(r[k] for k in ['split','checkpoint','condition','execution'])==context]
        for arm in sorted({r['arm'] for r in selected}):
            candidates=[r for r in selected if r['arm']==arm or (arm!='main' and r['group'].endswith(('_MUT','_PPO')))]
            for metric in ['attack_success_rate','collision_rate','canonical_return','style_success_rate','cut_count','time']:
                if not any(metric in r for r in candidates):continue
                fig,axes=plt.subplots(1,3,figsize=(15,4.8));drawn=False
                for level,ax in zip([1,2,3],axes):
                    groups=sorted({r['group'] for r in candidates if r['level']==level and metric in r})
                    for x,group in enumerate(groups):
                        items=[r for r in candidates if r['group']==group and metric in r]
                        if not partial and sorted(r['seed'] for r in items)!=sorted(expected):continue
                        vals=[r[metric] for r in items];s=mean_sd(vals)
                        ax.bar(x,s['mean'],yerr=s['sample_sd'],capsize=3,alpha=.7)
                        ax.scatter(x+np.linspace(-.1,.1,len(vals)),vals,color='#233247',s=15);drawn=True
                    ax.set_xticks(range(len(groups)),[g.replace(f'L{level}_','') for g in groups],rotation=35,ha='right')
                    ax.set_title(f'L{level}');ax.grid(axis='y',alpha=.2)
                    if not groups:ax.text(.5,.5,'N/A',transform=ax.transAxes,ha='center',va='center',color='gray')
                    if 'rate' in metric or 'fraction' in metric:ax.set_ylim(0,1.1)
                if drawn:
                    name='_'.join([arm,metric,*context]);fig.suptitle(name+(' [PARTIAL]' if partial else ''))
                    fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.pdf'))
                plt.close(fig)


def curve_figures(rows,output,expected,partial):
    if not rows:return
    plt=plotting()
    for metric in sorted({r['metric'] for r in rows}):
        selected=[r for r in rows if r['arm']=='main' and r['metric']==metric]
        if not selected:continue
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for level,ax in zip([1,2,3],axes):
            for group in sorted({r['group'] for r in selected if r['level']==level}):
                items=[r for r in selected if r['group']==group]
                if not partial and sorted({r['seed'] for r in items})!=sorted(expected):continue
                steps=sorted({r['steps'] for r in items});means=[];sds=[]
                for step in steps:
                    values=[r['value'] for r in items if r['steps']==step]
                    means.append(statistics.mean(values));sds.append(statistics.stdev(values) if len(values)>1 else 0.)
                x=np.array(steps)/1000;mean=np.array(means);sd=np.array(sds)
                ax.plot(x,mean,label=group.replace(f'L{level}_',''));ax.fill_between(x,mean-sd,mean+sd,alpha=.12)
            ax.set(title=f'L{level}',xlabel='A+B steps (k)');ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'N/A / no data',transform=ax.transAxes,ha='center',va='center',color='gray')
        fig.suptitle('Selection curve: '+metric+' (mean ± seed sample SD; no best-of-two envelope)')
        fig.tight_layout();fig.savefig(output/f'training_{metric}.png',dpi=160);fig.savefig(output/f'training_{metric}.pdf');plt.close(fig)


def stability_figures(rows,output,expected,partial):
    rows=[r for r in rows if r['arm']=='main']
    if not rows:return
    plt=plotting()
    for metric in ['late_asr_mean','late_drawdown','wall_seconds','first_50pct_steps_capped']:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5));drawn=False
        for level,ax in zip([1,2,3],axes):
            groups=sorted({r['run'].rsplit('_s',1)[0] for r in rows if r['level']==level})
            for i,group in enumerate(groups):
                selected=[r for r in rows if r['run'].rsplit('_s',1)[0]==group]
                if not partial and sorted(r['seed'] for r in selected)!=sorted(expected):continue
                vals=[r[metric] for r in selected if r.get(metric) is not None]
                if not vals:continue
                s=mean_sd(vals);ax.bar(i,s['mean'],yerr=s['sample_sd'],capsize=3,alpha=.7)
                ax.scatter(i+np.linspace(-.1,.1,len(vals)),vals,color='#233247',s=15);drawn=True
            ax.set_xticks(range(len(groups)),[g.replace(f'L{level}_','') for g in groups],rotation=30,ha='right')
            ax.set_title(f'L{level}');ax.grid(axis='y',alpha=.2)
        if drawn:
            fig.suptitle(metric+(' [PARTIAL]' if partial else ''))
            if metric=='first_50pct_steps_capped':fig.text(.5,.01,'Unreached threshold is capped at full budget; reached_50pct is stored separately.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(output/f'stability_{metric}.png',dpi=160);fig.savefig(output/f'stability_{metric}.pdf')
        plt.close(fig)
