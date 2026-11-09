"""Serial blue-policy replacement experiments; policy effects, not real-world causality."""
import argparse
import json
from pathlib import Path
from training.mut_mappo_config import load_config,validate_config,cases_for,make_actors
from training.checkpoint_protocol import atomic_write_json
from evaluation.r5_outcome_metrics import BurdenFilter

def contrast(records):
    if len(records)!=4:raise ValueError('exactly four policy replacement records required')
    modes={r['replacement']:r for r in records}
    if set(modes)!={'none','blue2','blue3','both'}:raise ValueError('four matched replacement modes required')
    if len({(r['case_key'],r['seed'],r['team']) for r in records})!=1:raise ValueError('counterfactual cases/seeds differ')
    valid=all(r['valid'] and not r['any_contact'] for r in records)
    if not valid:return dict(eligible=False,reason='invalid or contact-containing matched set',effects=None)
    # No arbitrary mixing of delay and burden: report both in their natural units.
    result={}
    for key in ('red_progress_debt_s','red_burden_integral_s'):
        value=lambda mode:modes[mode][key]
        result[key]=dict(blue2_marginal=value('none')-value('blue2'),
            blue3_marginal=value('none')-value('blue3'),
            joint_effect=value('none')-value('both'),
            interaction=value('none')-value('blue2')-value('blue3')+value('both'))
    return dict(eligible=True,effects=result,claim='matched simulator policy replacement, not action-level causal attribution')

def evaluate(config_path,checkpoint,output,split='selection',team='A'):
    from training.train_mut_mappo import create_runtime,run_case
    from utils.mut_mappo_session import CarlaOwner
    cfg=validate_config(load_config(config_path));folder=Path(output)
    folder.mkdir(parents=True,exist_ok=False)
    atomic_write_json(dict(status='RUNNING',checkpoint=str(checkpoint),split=split,team=team),folder/'manifest.json')
    results=[];effects=[]
    with CarlaOwner(cfg['host'],cfg['port'],folder,'R5 counterfactual evaluation'):
        for case in cases_for(cfg,split):
            group=[]
            for mode in ('none','blue2','blue3','both'):
                runtime=None
                try:
                    runtime=create_runtime(cfg,checkpoint,team=team,condition='C2')
                    source=cfg['_red_payload']
                    initial=make_actors({2:source['actors'][0],3:source['actors'][1]},source['training_config'],(2,3),cfg['device'])
                    for bid in (2,3):
                        if mode=='both' or mode=='blue'+str(bid):runtime.blue_actors[bid]=initial[bid]
                    filt=BurdenFilter(cfg['task']['timestep'],cfg['task']['team_reward'].get('filter_tau_s',.3))
                    debt=burden=0.;n=0;contact=False;last_progress={0:0.,1:0.}
                    def collect(info):
                        nonlocal debt,burden,n,contact,last_progress
                        n+=1;m=info['mut_metrics']
                        contact=contact or bool(m.get('contact_events')) or m.get('safety',{}).get('collision',False)
                        if contact:return
                        progress={int(k):v for k,v in m['route_progress'].items()}
                        last_progress=progress
                        active={int(k):v for k,v in info['active_mask'].items()}
                        snaps={int(k):v for k,v in info['post_tick_kinematics'].items()}
                        reached={rid:progress.get(rid)==1. for rid in (0,1)}
                        b=filt.step(snaps,active,reached)
                        dt=cfg['task']['timestep']
                        debt+=sum(1.-progress.get(rid,0.) for rid in (0,1))*dt/2.
                        burden+=sum(b.values())*dt/2.
                    record=run_case(runtime,case,seed=case['seed'],step_callback=collect)
                    if not contact:
                        debt+=sum(1.-last_progress.get(rid,0.) for rid in (0,1))*max(0,cfg['task']['max_steps']-n)*cfg['task']['timestep']/2.
                    record.update(replacement=mode,any_contact=contact,red_progress_debt_s=debt,
                        red_burden_integral_s=burden,fixed_horizon_s=cfg['task']['max_steps']*cfg['task']['timestep'])
                    group.append(record);results.append(record)
                    with (folder/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                finally:
                    if runtime is not None:runtime.close()
            effects.append(dict(case_key=case['case_key'],**contrast(group)))
    atomic_write_json(dict(status='COMPLETED',n=len(results),effects=effects,
        caveat='Report rejected matched sets and contacts separately; conditioning changes scene composition'),folder/'analysis.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--team',choices=['A','B'],default='A')
    p.add_argument('--split',choices=['selection','test'],default='selection');a=p.parse_args()
    evaluate(a.config,a.checkpoint,a.output_dir,a.split,a.team)
