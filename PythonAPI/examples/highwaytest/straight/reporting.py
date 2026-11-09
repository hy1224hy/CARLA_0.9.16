"""Matched independent-seed analysis; pool selection never uses final tests."""
import csv
import json
from pathlib import Path
import numpy as np


def winner_series(rows,level):
    """Per-round best-of-A/B metric series from validation rows: [(train_steps, rate)]."""
    from .runner import rank_report
    key='success_rate' if level==0 else 'attack_success_rate'
    series=[]
    for row in rows:
        winner=max(row['reports'],key=lambda k:rank_report(row['reports'][k]['overall'],level))
        overall=row['reports'][winner]['overall']
        if key in overall:series.append((int(row['train_steps']),float(overall[key])))
    return series


def selected_lineage(rows,level,team):
    """One evolving policy lineage's per-round metric series: [(train_steps, rate)]."""
    key='success_rate' if level==0 else 'attack_success_rate'
    series=[]
    for row in rows:
        overall=row['reports'].get(team,{}).get('overall',{})
        if key in overall:series.append((int(row['train_steps']),float(overall[key])))
    return series


def drawdown(values,half_window=None):
    """Late-training regression on the second half of a metric series."""
    values=list(values)
    window=len(values)//2 if half_window is None else int(half_window)
    tail=values[-window:] if window>0 else []
    if not tail:return {'max_to_min':0.,'mean_adjacent_drop':0.}
    return {'max_to_min':float(max(tail)-min(tail)),
            'mean_adjacent_drop':float(np.mean([max(0.,tail[i]-tail[i+1]) for i in range(len(tail)-1)])) if len(tail)>1 else 0.}


def ablation_label(cfg):
    if cfg.get('risk_weight',1.)==0:return 'no_risk'
    if cfg.get('event_weight',1.)==0:return 'no_event'
    if cfg.get('effect_weight',1.)==0:return 'no_effect'
    return 'full'


def main_row(row):
    """True only for main-comparison rows: full reward, independent (effective beta=0) or mutual beta=0.02."""
    return (row.get('level') in (1,2,3) and row.get('method') in ('independent','mutual')
            and row.get('risk_weight',1.)==1. and row.get('event_weight',1.)==1.
            and row.get('effect_weight',1.)==1. and row.get('geometry_weight',0.)==0.
            and (row.get('method')=='independent' or row.get('beta',.02)==.02))


def report(runs,output):
    from .figures import (final_metric_figure,learning_curve_figure,stability_figure,
                          beta_panel,ablation_panel,trajectory_figure)
    from ..training.checkpoint import source_digest
    current_source=source_digest()
    roots=[runs] if isinstance(runs,(str,Path)) else list(runs)
    rows=[];seen=set();contracts={};case_sets={};series_by_run={};series_rows=[]
    for root in roots:
        for path in sorted(Path(root).rglob('evaluation.json')):
            data=json.loads(path.read_text(encoding='utf-8'))
            if data.get('split')!='test':continue
            if data.get('source_changed'):raise ValueError('Do not mix source-changed evaluations')
            if data['overall'].get('baseline_valid_rate',1.)<1.:
                raise ValueError('Normal reference failed; do not aggregate invalid paired effects')
            cfg=data['config'];level=cfg['level'];beta=cfg['beta'] if cfg['method']=='mutual' else 0.
            source=data.get('source_sha256')
            if not source or data.get('evaluation_source_sha256')!=source:
                raise ValueError('Missing or inconsistent evaluation source identity')
            if source!=current_source:
                raise ValueError('Evaluation predates the current source; freeze the source and re-run evaluate')
            treatment=tuple(cfg.get(k,default) for k,default in (('geometry_weight',0.),('effect_weight',1.),('risk_weight',1.),('event_weight',1.)))
            identity=(level,cfg['method'],beta,*treatment,cfg['seed'])
            if identity in seen:raise ValueError('Duplicate independent training seed/evaluation')
            seen.add(identity)
            contract={k:v for k,v in cfg.items() if k not in ('seed','method','beta','geometry_weight','effect_weight','event_weight','risk_weight')}
            if level in contracts and contracts[level]!=contract:raise ValueError('Unmatched experiment configuration')
            contracts[level]=contract
            cases=[(e['case_seed'],e['family']) for e in data['episodes']]
            if level in case_sets and case_sets[level]!=cases:raise ValueError('Test cases are not paired')
            case_sets[level]=cases
            row=dict(level=level,method=cfg['method'],beta=beta,attack_weight=treatment[1],ablation=ablation_label(cfg),
                     geometry_weight=treatment[0],effect_weight=treatment[1],risk_weight=treatment[2],event_weight=treatment[3],
                     seed=cfg['seed'],train_steps_total=data.get('train_steps_total'),**data['overall'])
            run_dir=None
            for candidate in (path.parent,path.parent.parent):
                if (candidate/'validation.jsonl').is_file():run_dir=candidate;break
            if run_dir is not None:
                validation=[json.loads(line) for line in (run_dir/'validation.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                summary_file=run_dir/'summary.json'
                if summary_file.is_file():
                    team=json.loads(summary_file.read_text(encoding='utf-8')).get('selection') or {}
                    team=team.get('team')
                    if team:
                        selected=selected_lineage(validation,level,team)
                        dd=drawdown([value for _,value in selected])
                        row['drawdown_max_to_min']=dd['max_to_min']
                        row['drawdown_mean_adjacent_drop']=dd['mean_adjacent_drop']
                if main_row(row):
                    winner=winner_series(validation,level)
                    series_by_run.setdefault((level,cfg['method']),[]).append((cfg['seed'],winner))
                    for team in ('A','B'):
                        for steps,rate in selected_lineage(validation,level,team):
                            series_rows.append(dict(level=level,method=cfg['method'],beta=beta,seed=cfg['seed'],team=team,train_steps=steps,rate=rate))
            rows.append(row)
    if not rows:raise ValueError('No final test evaluations found')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with (output/'results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    if series_rows:
        series_columns=['level','method','beta','seed','team','train_steps','rate']
        with (output/'results_series.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=series_columns);writer.writeheader();writer.writerows(series_rows)
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in ('level','method','beta','geometry_weight','effect_weight','risk_weight','event_weight')),[]).append(row)
    summary=[]
    for (level,method,beta,geometry,effect,risk,event),items in groups.items():
        result=dict(level=level,method=method,beta=beta,geometry_weight=geometry,effect_weight=effect,risk_weight=risk,event_weight=event,seeds=[r['seed'] for r in items],n=len(items))
        for key in columns:
            if key in ('seed','level','beta','method','geometry_weight','effect_weight','risk_weight','event_weight'):continue
            vals=[r[key] for r in items if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
            if vals:result[key]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
        summary.append(result)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for metric in ('attack_success_rate','collision_rate','grade_violation_rate','utility'):
        final_metric_figure(rows,groups,metric,output)
    learning_curve_figure(series_by_run,output)
    stability_figure([r for r in rows if main_row(r)],output)
    beta_panel(rows,output)
    ablation_panel(rows,output)
    trajectory_paths=sorted({p for root in roots for p in Path(root).rglob('trajectory.json')})
    if trajectory_paths:trajectory_figure(trajectory_paths,output)
    (output/'figures_note.json').write_text(json.dumps(dict(
        attack_weight_study='not run this round; figure 6 omitted',
        collision_direction='L1/L2 lower collision is the goal; L3 higher valid target-collision is the goal',
        drawdown_scope='selected lineage, second half of per-round selection evaluations'),indent=2),encoding='utf-8')
    return dict(runs=len(rows),groups=len(groups),series_runs=len(series_by_run),output=str(output),statistics='independent seed mean and sample SD; <3 seeds exploratory')


def matrix(runs,output):
    """Training-intent x actual-behavior consistency matrix over selected models (main runs)."""
    from .runner import load_state,configuration,evaluate_model
    from .behavior import build_matrix,classify,CLASSES
    from .figures import matrix_heatmap
    from .model import ActorCritic
    from ..training.checkpoint import source_digest
    identity=source_digest()
    entries=[];missing=[];skipped_non_main=[]
    for path in sorted(Path(runs).rglob('selected.pt')):
        state=load_state(path);cfg=configuration(state['config'])
        if state['source_sha256']!=identity:raise ValueError('Matrix checkpoint source mismatch: '+str(path))
        if not main_row(cfg.to_dict()):skipped_non_main.append(path.parent.name);continue
        model=ActorCritic(cfg.hidden);model.load_state_dict(state['model'])
        report=evaluate_model(model,cfg,'selection',cfg.eval_cases)
        entries.append((cfg.level,cfg.method,cfg.seed,report['episodes']))
    for path in sorted(Path(runs).rglob('config.json')):
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        if main_row(raw) and not (path.parent/'selected.pt').is_file():
            missing.append(path.parent.name)
    if not entries:raise ValueError('No selected checkpoints found under '+str(runs))
    result=build_matrix(entries)
    merged=[]
    for level in (1,2,3):
        episodes=[e for (l,_,_,eps) in entries if l==level for e in eps]
        counts={name:0 for name in CLASSES}
        for episode in episodes:counts[classify(episode)]+=1
        total=len(episodes)
        if total:merged.append(dict(level=level,cases=total,
            classes={name:(counts[name]/total if total else 0.) for name in CLASSES}))
    result.update(protocol='straight_consistency_v1',selection_split='selection',
        merged_rows=merged,missing=missing,skipped_non_main=skipped_non_main)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    matrix_heatmap(result,output.with_suffix(''))
    csv_path=output.with_suffix('.per_seed.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(('seed','level','method',*CLASSES))
        for level,method,seed,episodes in sorted(entries):
            counts={name:0 for name in CLASSES}
            for episode in episodes:counts[classify(episode)]+=1
            writer.writerow((seed,level,method,*(counts[name] for name in CLASSES)))
    return result


def build_pool(runs,output):
    from .runner import load_state,rank_report
    from ..training.checkpoint import source_digest
    members={};physical=None;identity=source_digest();skipped_non_main=[]
    for path in sorted(Path(runs).rglob('selected.pt')):
        state=load_state(path);cfg=state['config'];selection=state.get('selection') or {}
        if state['source_sha256']!=identity:raise ValueError('Pool checkpoint source mismatch')
        contract={k:cfg[k] for k in ('duration','policy_frequency','simulation_frequency','target_speed','target_time_gap','hidden')}
        if physical is not None and physical!=contract:raise ValueError('Pool physical/model contract mismatch')
        physical=contract
        if not main_row(cfg):skipped_non_main.append(path.parent.name);continue
        if cfg['level'] not in (1,2,3) or not selection.get('eligible'):continue
        overall=selection['overall']
        if overall.get('attack_success_rate',0)<=0:continue
        gates={1:(overall.get('blocking_time',0.)>=1.5 and overall.get('collision_rate',1.)<=.125 and overall.get('grade_violation_rate',1.)<=.125),
              2:(overall.get('cut_in_rate',0.)>0 and overall.get('collision_rate',1.)<=.125),
              3:(overall.get('collision_rate',0.)>0)}
        if not gates[cfg['level']]:continue
        score=rank_report(overall,cfg['level']);previous=members.get(cfg['level'])
        if previous is None or score>tuple(previous['rank']):
            members[cfg['level']]=dict(level=cfg['level'],checkpoint=str(path.resolve()),rank=list(score),
                seed=cfg['seed'],method=cfg['method'],config=cfg,selection=selection,source_sha256=state['source_sha256'],
                descriptors={k:overall.get(k) for k in ('blocking_time','cut_in_rate','stop_time','collision_rate',
                    'grade_violation_rate','avg_speed_ratio','max_target_brake','cut_count','cut_first_thw',
                    'contact_lateral_rate','contact_longitudinal_rate')})
    result=dict(protocol='straight_pool_v1',selection_split='selection',members=list(members.values()),
        missing_levels=[k for k in (1,2,3) if k not in members],
        skipped_non_main=skipped_non_main,
        scope='three representative grade slots; no claim of within-grade mechanism diversity')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    return result
