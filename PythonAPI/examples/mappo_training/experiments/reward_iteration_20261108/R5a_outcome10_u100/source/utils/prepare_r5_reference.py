"""Serial frozen-blue calibration. Never invoked automatically by R5 config validation."""
import argparse
import hashlib
import statistics
from pathlib import Path
import json
from evaluation.r5_outcome_metrics import BurdenFilter, PROTOCOL
from environment.mut_mappo_efficiency_reward import reference_identity
from training.mut_mappo_config import load_config,validate_config,resolve_path,cases_for
from training.checkpoint_protocol import atomic_write_json,sha256_file

def aggregate(runs,horizon,min_samples=3):
    steps={}
    for step in range(1,horizon+1):
        reds={}
        for rid in ('0','1'):
            points=[r.get(str(step),{}).get(rid) for r in runs]
            points=[p for p in points if p is not None]
            if len(points)<min_samples: continue
            reds[rid]=dict(progress_mean=statistics.mean(p['progress'] for p in points),
                progress_std=statistics.pstdev(p['progress'] for p in points),
                burden_mean=statistics.mean(p['burden'] for p in points),
                burden_std=statistics.pstdev(p['burden'] for p in points),valid_samples=len(points))
        if reds:steps[str(step)]=reds
    return dict(protocol=PROTOCOL,repeats=len(runs),steps=steps,
        calibration='engineering scales plus repeated-reference spread; state/time confounded')

def prepare(config_path,repeats=3,splits=('train','gate','selection')):
    from training.train_mut_mappo import create_runtime,run_case
    from utils.mut_mappo_session import CarlaOwner
    config=validate_config(load_config(config_path))
    if config['task']['reward_profile']!='individual_red_outcome_v5':raise ValueError('R5 config required')
    if repeats<config['task']['team_reward'].get('min_reference_samples',3):raise ValueError('too few reference repetitions')
    config['_building_reference']=True
    folder=resolve_path(config['task']['team_reward']['reference_dir']);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():raise FileExistsError('reference generation requires a fresh directory')
    identity=reference_identity(config);header=dict(identity=identity,cases={},schema='r5_reference_v1')
    runtime=None
    with CarlaOwner(config['host'],config['port'],folder,'R5 reference'):
        try:
            runtime=create_runtime(config,condition='C2')
            for split in splits:
                for case in cases_for(config,split):
                    runs=[];results=[]
                    for rep in range(repeats):
                        filt=BurdenFilter(config['task']['timestep'],config['task']['team_reward'].get('filter_tau_s',.3))
                        curve={};completed={};contacted=set();step=0
                        def collect(info):
                            nonlocal step
                            step+=1
                            metric=info['mut_metrics'];snaps=info['post_tick_kinematics']
                            snaps={int(k):v for k,v in snaps.items()}
                            reached={int(k):v for k,v in info.get('reached_target',{}).items()}
                            for rid in (0,1):
                                if metric.get('agent_outcomes',{}).get(str(rid),metric.get('agent_outcomes',{}).get(rid))=='success':reached[rid]=True
                            for event in metric.get('contact_events',[]):
                                contacted.update(i for i in (event.get('agent_id'),event.get('other_agent_id')) if i in (0,1))
                            active={int(k):v for k,v in info['active_mask'].items()}
                            # Terminal snapshots may be inactive; never differentiate them.
                            burden=filt.step(snaps,active,reached)
                            points={}
                            for rid in (0,1):
                                if rid in contacted:continue
                                prog=metric['route_progress'].get(str(rid),metric['route_progress'].get(rid))
                                if reached.get(rid) or prog==1.:
                                    completed[str(rid)]=dict(progress=1.,burden=0.)
                                elif rid in burden:
                                    points[str(rid)]=dict(progress=prog,burden=burden[rid])
                            points.update(completed);curve[str(step)]=points
                        result=run_case(runtime,case,seed=case['seed']+rep*10000,step_callback=collect)
                        if not result['valid']:raise RuntimeError('invalid reference rollout')
                        for n in range(step+1,config['task']['max_steps']+1):curve[str(n)]=dict(completed)
                        runs.append(curve);results.append(result)
                    outcome=aggregate(runs,config['task']['max_steps'],config['task']['team_reward'].get('min_reference_samples',3))
                    outcome['filter_tau_s']=config['task']['team_reward'].get('filter_tau_s',.3)
                    if not outcome['steps']:raise RuntimeError('no calibrated reference coverage')
                    progress={rid:[0.]+[outcome['steps'].get(str(n),{}).get(rid,{}).get('progress_mean',0.) for n in range(1,config['task']['max_steps']+1)] for rid in ('0','1')}
                    record=dict(case_key=case['case_key'],identity=identity,progress=progress,outcome_reference=outcome,
                        reference_runs=results,response_reference_mode='frozen_base_blue',scope='diagnostic reference, not qualified red certification')
                    filename=hashlib.sha256(case['case_key'].encode()).hexdigest()+'.json'
                    atomic_write_json(record,folder/filename)
                    header['cases'][case['case_key']]=dict(file=filename,sha256=sha256_file(folder/filename),repeats=repeats,split=split)
                    atomic_write_json(header,folder/'manifest.json')
                    print('R5_REFERENCE_DONE',case['case_key'],flush=True)
        finally:
            if runtime is not None:runtime.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--splits',nargs='+',default=['train','gate','selection'],choices=['train','gate','selection','test'])
    a=p.parse_args();prepare(a.config,a.repeats,a.splits)
