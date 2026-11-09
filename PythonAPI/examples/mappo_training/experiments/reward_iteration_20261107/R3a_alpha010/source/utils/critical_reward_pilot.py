"""Explicitly authorized weak-red reward pilot: prepare and serial C0/C1 screen."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/adversarial_u1000_critical_reward_20261101_v1'


def prepare(root=EXPERIMENT, case_manifest=None, c1_controller=None):
    from training.checkpoint_protocol import atomic_write_json
    from training.mut_mappo_config import load_config, validate_config
    from environment.mut_mappo_critical_reward import DEFAULTS, PROFILE
    base=load_config('experiments/adversarial_u1000_interactions_20261029_v2/base_config.json')
    base={k:v for k,v in base.items() if not k.startswith('_')}
    if case_manifest is not None: base['case_manifest']=str(case_manifest)
    if c1_controller is not None: base['task']['c1_controller']=c1_controller
    base['task']['follow_spectator']=True
    (root/'configs').mkdir(parents=True,exist_ok=True)
    results={}
    for group in ('R0','R1'):
        config=copy.deepcopy(base)
        config['task']['team_reward']['reference_dir']=str(root/'references'/group)
        config['diagnostic_authorization']={'live_execution':True,'weak_red_diagnostic_only':True,
            'source':'User approved implementation and live tests on 2026-11-01',
            'formal_red_qualification':False}
        if group=='R1':
            config['task']['reward_profile']=PROFILE
            config['task']['team_reward'].update(DEFAULTS,attack_weight=2.)
            config['mutual']['quality_metric']='critical'
            config['evaluation']['selection_metric']='safe_critical_event'
        validated=validate_config(config)
        path=root/'configs'/f'{group}_B0_s42.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config:
            raise FileExistsError('preserve existing pilot configuration; create a new version')
        atomic_write_json(config,path)
        results[group]=validated['_root_validation']
    atomic_write_json(results,root/'control/offline_compatibility.json')
    return results


def screen(root=EXPERIMENT, limit=12, repeats=2, group='R0'):
    from training.checkpoint_protocol import atomic_write_json
    from training.mut_mappo_config import load_config, validate_config, cases_for, portable_config
    from training.train_mut_mappo import create_runtime, run_case, append_record
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.critical_interaction_metrics import analyze_trace, PROTOCOL_DEFAULT
    config=validate_config(load_config(root/'configs'/f'{group}_B0_s42.json'))
    cases=cases_for(config,'gate')
    cases=([c for c in cases if c.get('direction') in ('northbound','eastbound')] if limit==12 else
           cases[:2] if limit==2 else cases)
    if limit not in (2,12,24) or len(cases)!=limit or repeats!=2:
        raise ValueError('screen is a complete 2/12/24-case mirror subset and two paired seeds')
    directory=root/f'screen_{limit}'
    directory.mkdir(parents=True,exist_ok=True)
    atomic_write_json(portable_config(config),directory/'resolved_config.json')
    import hashlib
    # C0/C1 do not require training references. Capture executable sources now,
    # without invoking run_identity's training-reference precondition.
    source_files=['utils/critical_reward_pilot.py','environment/mut_mappo_env.py',
        'environment/carla_multi_agent_env.py','environment/mut_mappo_efficiency_reward.py',
        'environment/mut_mappo_critical_reward.py','training/train_mut_mappo.py',
        'training/mut_mappo_config.py','evaluation/critical_interaction_metrics.py']
    if config['task'].get('reward_profile')=='normal_driving_critical_team_v2':
        source_files.append('environment/mut_mappo_critical_reward_v2.py')
    if config['task'].get('c1_controller')=='lane_pid_v1':
        source_files+=['environment/frozen_lane_traffic.py']
    source_identity={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}
    source_path=directory/'source_identity.json'
    if source_path.exists() and json.loads(source_path.read_text(encoding='utf-8'))!=source_identity:
        raise ValueError('screen source changed; retain old evidence and create a new experiment')
    atomic_write_json(source_identity,source_path)
    completed=[]
    records=directory/'cases.jsonl'
    if records.exists():
        completed=[json.loads(line) for line in records.read_text(encoding='utf-8').splitlines() if line]
        if any(not r['valid'] for r in completed):
            raise RuntimeError('prior screen has invalid evidence; inspect before creating a new attempt')
    done={(r['case_key'],r['seed'],r['condition']) for r in completed}
    state={'status':'RUNNING','reward_group':group,'n_planned':len(cases)*repeats*2,'n_completed':len(completed)}
    atomic_write_json(state,directory/'status.json')
    with CarlaOwner(run_dir=directory,purpose='authorized weak-red paired C0/C1 screen'):
        try:
            for index,case in enumerate(cases):
                for repetition in range(repeats):
                    seed=case['seed']+repetition*10000
                    for condition in ('C0','C1'):
                        if (case['case_key'],seed,condition) in done: continue
                        runtime=None;row=None
                        counts={'total':0,'decision':0,'setup':0}
                        def audit(event):
                            ticks=int(event.get('ticks',1)); counts['total']+=ticks
                            kind=event.get('kind','setup'); counts[kind]=counts.get(kind,0)+ticks
                        trace=directory/'traces'/f'{index:03}_{repetition}_{condition}.jsonl'
                        if trace.exists(): raise FileExistsError('uncommitted trace must be preserved and reviewed')
                        try:
                            runtime=create_runtime(config,condition=condition,audit_callback=audit)
                            row=run_case(runtime,case,seed=seed,trace_path=trace)
                        finally:
                            try:
                                if runtime: runtime.close()
                            finally:
                                append_record(directory/'tick_ledger.jsonl',dict(case_key=case['case_key'],seed=seed,
                                    condition=condition,counts=counts,episode_returned=row is not None,
                                    scope='all explicit controller ticks including construction/reset/cleanup'))
                        row['controller_tick_counts']=dict(counts)
                        if condition=='C1':
                            rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
                            row['critical_analysis']=analyze_trace(rows,PROTOCOL_DEFAULT)
                        append_record(records,row); completed.append(row)
                        state.update(n_completed=len(completed),last_case=case['case_key'],last_condition=condition)
                        atomic_write_json(state,directory/'status.json')
                        print(json.dumps({'event':'SCREEN_CASE_DONE','case':case['case_key'],'seed':seed,
                            'condition':condition,'valid':row['valid'],'red_success':row['red_success'],
                            'blue_eligible':row['blue_eligible'],'end_reason':row.get('end_reason')},ensure_ascii=False),flush=True)
                        if not row['valid']: raise RuntimeError('invalid screen execution')
            groups=[]
            for repetition in range(repeats):
                seeds={c['seed']+repetition*10000 for c in cases}
                for condition in ('C0','C1'):
                    rows=[r for r in completed if r['seed'] in seeds and r['condition']==condition]
                    groups.append(dict(repetition=repetition,condition=condition,n=len(rows),
                        red_success=sum(r['red_success'] for r in rows)/len(rows),
                        blue_eligible=sum(r['blue_eligible'] for r in rows)/len(rows)))
            ready=all(g['red_success']>=.95 and (g['condition']=='C0' or g['blue_eligible']>=.95) for g in groups)
            state.update(status='COMPLETED',ready_for_training=ready,groups=groups,
                scope='weak-red interaction screen; not base driving qualification',
                decision_ticks=sum(r.get('tick_counts',{}).get('decision',0) for r in completed))
            ledger=directory/'tick_ledger.jsonl'
            if ledger.exists():
                state['controller_world_ticks']=sum(json.loads(line)['counts']['total']
                    for line in ledger.read_text(encoding='utf-8').splitlines())
            atomic_write_json(state,directory/'summary.json')
            atomic_write_json(state,directory/'status.json')
            return state
        except BaseException as error:
            state.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            atomic_write_json(state,directory/'status.json')
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','screen'])
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--root',type=Path,default=EXPERIMENT)
    p.add_argument('--case-manifest')
    p.add_argument('--group',choices=['R0','R1','R2'],default='R0')
    p.add_argument('--c1-controller',choices=['frozen_actor','lane_pid_v1'])
    args=p.parse_args()
    root=args.root if args.root.is_absolute() else ROOT/args.root
    result=prepare(root,args.case_manifest,args.c1_controller) if args.action=='prepare' else screen(root,limit=args.limit,group=args.group)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
