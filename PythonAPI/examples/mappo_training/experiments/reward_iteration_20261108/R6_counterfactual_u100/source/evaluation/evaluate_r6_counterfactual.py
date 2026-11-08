"""Independent four-mode R6 selection/test evaluation; one world per team.

Run explicitly after training; never invoked during validation or imports.
All four deterministic modes are retained, including rejected/contact cases.
"""
import argparse
import hashlib
from pathlib import Path
from training.checkpoint_protocol import atomic_write_json
from training.mut_mappo_config import load_config,validate_config,cases_for
from evaluation.r6_counterfactual import PROFILE,paired_credit,settle_credit,summarize_records


def evaluate(config_path,checkpoint,output,split='selection'):
    from training.train_mut_mappo import create_runtime,append_record
    from training.r6_collection import base_actors,measured_rollout
    from evaluation.evaluate_r5_counterfactual import contrast
    from utils.mut_mappo_session import CarlaOwner
    cfg=validate_config(load_config(config_path))
    if cfg['task']['reward_profile']!=PROFILE:raise ValueError('R6 configuration required')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    atomic_write_json(dict(status='RUNNING',split=split,checkpoint=str(checkpoint),policy_mode='deterministic'),folder/'manifest.json')
    effects=[];total=0;actual_records=[]
    with CarlaOwner(cfg['host'],cfg['port'],folder,'R6 four-mode evaluation'):
        for team in ('A','B'):
            runtime=None
            try:
                runtime=create_runtime(cfg,checkpoint,team=team,condition='C2')
                current=dict(runtime.blue_actors);base=base_actors(runtime)
                for case in cases_for(cfg,split):
                    group={}
                    for mode,replaced in (('none',()),('blue2',(2,)),('blue3',(3,)),('both',(2,3))):
                        runtime.blue_actors={i:base[i] if i in replaced else current[i] for i in (2,3)}
                        timeline=folder/'timelines'/(team+'_'+hashlib.sha256(case['case_key'].encode()).hexdigest()[:16]+'_'+mode+'.json')
                        r=measured_rollout(runtime,case,case['seed'],timeline_path=timeline)
                        r['red_outcome_timeline']=str(timeline.relative_to(folder))
                        r.update(replacement=mode,policy_mode='deterministic',split=split)
                        group[mode]=r
                    credits={i:paired_credit(group['none'],group['blue'+str(i)],i,cfg['task']['team_reward']) for i in (2,3)}
                    settle_credit(group['none'],None,{},credits)
                    actual_records.append(group['none'])
                    for r in group.values():append_record(folder/'cases.jsonl',r);total+=1
                    checks=[paired_credit(group['none'],group[m],2,cfg['task']['team_reward']) for m in ('blue2','blue3','both')]
                    joint=contrast(list(group.values())) if all(x['eligible'] for x in checks) else dict(eligible=False,reason='; '.join(sorted({x['reason'] for x in checks if not x['eligible']})),effects=None)
                    effects.append(dict(team=team,case_key=case['case_key'],pair_id=case.get('pair_id'),
                        single_policy_effects={str(i):credits[i] for i in (2,3)},four_mode_joint=joint))
            finally:
                if runtime is not None:runtime.close()
    atomic_write_json(dict(status='COMPLETED',records=total,matched_groups=len(effects),effects=effects,summary=summarize_records(actual_records),
        caveats=['A/B are not independent training seeds','raw rejected/contact outcomes must be reported',
            'endpoint progress extension is a scoring convention','four-mode joint exclusion changes scene composition',
            'deterministic evaluation does not characterize stochastic-policy deployment']),folder/'analysis.json')
    atomic_write_json(dict(status='COMPLETED',split=split,records=total,matched_groups=len(effects)),folder/'manifest.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--split',choices=('selection','test'),default='selection')
    a=p.parse_args();evaluate(a.config,a.checkpoint,a.output_dir,a.split)
