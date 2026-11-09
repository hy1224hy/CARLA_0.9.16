"""Execute the first authorized R5 experiment, then stop for the agent's evidence review."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

PROJECT = Path(r'D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
ROOT = PROJECT/'experiments/reward_iteration_20261108'
sys.path.insert(0, str(PROJECT))

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def write(path, data):
    from training.checkpoint_protocol import atomic_write_json
    atomic_write_json(data, path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    result=[]
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            try: result.append(json.loads(line))
            except ValueError: pass
    return result

def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None

def bootstrap(values):
    """Case-pair cluster bootstrap; these are not training-seed confidence intervals."""
    import random
    if not values: return dict(n=0, mean=None, lower=None, upper=None)
    rng=random.Random(42)
    samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return dict(n=len(values), mean=statistics.mean(values),lower=samples[100],upper=samples[3899])

def cf_summary(data, records):
    cases=[r for r in data['effects'] if r.get('eligible') is True]
    groups={}
    for r in records:
        groups.setdefault(r['case_key'],[]).append(r)
    # Bootstrap the registered pair_id clusters rather than treating members as independent.
    measures={}
    for metric in ('red_progress_debt_s','red_burden_integral_s'):
        measures[metric]={}
        for name in ('blue2_marginal','blue3_marginal','joint_effect','interaction'):
            clustered={}
            for case in cases:
                rr=groups[case['case_key']]
                cluster=rr[0].get('pair_id') or case['case_key']
                clustered.setdefault(str(cluster),[]).append(case['effects'][metric][name])
            value=bootstrap([statistics.mean(v) for v in clustered.values()])
            value['case_mean']=mean([case['effects'][metric][name] for case in cases])
            measures[metric][name]=value
    raw=[r for r in records if r.get('replacement')=='none']
    coverage={}
    for aid in ('2','3'):
        metrics=[r.get('individual_outcome_metrics',{}).get(aid,{}) for r in raw if r.get('valid')]
        candidates=sum(m.get('candidate_samples',0) for m in metrics)
        coverage[aid]=sum(m.get('reference_samples',0) for m in metrics)/candidates if candidates else None
    persistent={aid:sum(r.get('valid') is True and not r.get('any_contact') and r.get('individual_outcome_metrics',{}).get(aid,{}).get('effective_outcome_longest_s',0.)>=1. for r in raw) for aid in ('2','3')}
    delay=measures['red_progress_debt_s']
    complete=len(data['effects'])==24 and len(records)==96 and len(groups)==24 and all(len(v)==4 for v in groups.values())
    passes=complete and len(cases)>=12 and all(delay[k]['lower'] is not None and delay[k]['lower']>0. for k in ('blue2_marginal','blue3_marginal','joint_effect')) and all((coverage[a] or 0.)>=.9 and persistent[a]>=3 for a in ('2','3'))
    return dict(n_groups=len(data['effects']),eligible=len(cases),excluded=len(data['effects'])-len(cases),complete=complete,
        contrasts=measures,coverage=coverage,persistent_cases=persistent,engineering_screen_passed=passes,
        contact_rate=mean([float(r.get('any_contact',False)) for r in raw]),
        note='Excludes any-contact matched sets; report rejection rate. Cluster CIs reflect scenarios only, not training seeds. Slowdown is interaction burden, not automatically red failure.')

def summarize(records):
    valid=[r for r in records if r.get('valid') is True]
    result=dict(n=len(records),valid=len(valid),teams={})
    for team in ('A','B'):
        cohort=[r for r in valid if r.get('team')==team]
        result['teams'][team]=dict(n=len(cohort),safety={k:mean([float(r.get('safety',{}).get(k,False)) for r in cohort]) for k in ('collision','offroad','road_rule')},roles={})
        for aid in ('2','3'):
            metric=[r.get('individual_outcome_metrics',{}).get(aid,{}) for r in cohort]
            parts=[r.get('individual_reward_components',{}).get(aid,{}) for r in cohort]
            result['teams'][team]['roles'][aid]=dict(
                outcome={k:mean([m[k] for m in metric if k in m]) for k in ('interaction_s','reference_coverage','delay_excess_integral_s','burden_excess_integral_s','outcome_signed_integral_s','effective_outcome_duration_s','effective_outcome_longest_s')},
                rewards={k:mean([m[k] for m in parts if k in m]) for k in ('base_soft','control_cost','collision','offroad','road_rule','task_violation','outcome_reward','severe')})
    return result

def snapshot(directory):
    hashes={}
    for folder in ('environment','training','models','evaluation','utils','tests'):
        for path in (PROJECT/folder).glob('*.py'):
            relative=path.relative_to(PROJECT)
            target=directory/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            hashes[relative.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copyfile(__file__,directory/'controller.py')
    hashes['controller.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(directory/'source_hashes.json',hashes)

def execute(directory,phase,module,args):
    log=directory/(phase+'_console.log')
    command=[sys.executable,'-X','utf8','-u','-m',module,*map(str,args)]
    state=dict(at=now(),status='RUNNING',version=directory.name,phase=phase,target_updates=100,command=command,log=str(log),controller_pid=os.getpid())
    write(ROOT/'control/state.json',state)
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        write(ROOT/'control/worker.json',dict(state,pid=child.pid,active=True))
        while child.poll() is None:
            time.sleep(5)
    write(ROOT/'control/worker.json',dict(state,pid=child.pid,active=False,returncode=child.returncode,ended_at=now()))
    if child.returncode: raise RuntimeError(f'{phase}: exit {child.returncode}; inspect {log}; no automatic restart')

def evaluate(directory, checkpoint, folder, split='selection'):
    execute(directory,folder,'evaluation.evaluate_mut_mappo',['--config',directory/'config.json','--checkpoint',checkpoint,'--team','both','--split',split,'--conditions','C2','--output-dir',directory/folder])
    manifest=read(directory/folder/'manifest.json')
    rr=rows(directory/folder/'cases.jsonl')
    if manifest.get('status')!='COMPLETED' or len(rr)!=48 or not all(r.get('valid') for r in rr):
        raise RuntimeError(f'{folder} incomplete or invalid; do not use as reward-tuning evidence')
    return summarize(rr)

def cf(directory, checkpoint, split='selection'):
    results={}
    for team in ('A','B'):
        folder=directory/f'counterfactual_{split}_{team}'
        execute(directory,folder.name,'evaluation.evaluate_r5_counterfactual',['--config',directory/'config.json','--checkpoint',checkpoint,'--team',team,'--split',split,'--output-dir',folder])
        data=read(folder/'analysis.json')
        if data.get('status')!='COMPLETED': raise RuntimeError('counterfactual evaluation incomplete')
        results[team]=cf_summary(data,rows(folder/'cases.jsonl'))
    return results

def initial_checkpoint(directory):
    from training.checkpoint_protocol import load_checkpoint_file,atomic_copy_checkpoint
    initial=list((directory/'run/boundaries').glob('round_00000000_*.pt'))
    if len(initial)!=1: raise RuntimeError('unambiguous initial checkpoint required')
    payload,_=load_checkpoint_file(initial[0])
    from training.mut_mappo_config import validate_config,load_config
    cfg=validate_config(load_config(directory/'config.json'))
    import torch
    for team in payload['teams'].values():
        for bid in (2,3):
            assert all(torch.equal(v,cfg['_red_payload']['actors'][bid-2][key]) for key,v in team['actors'][bid].items())
    atomic_copy_checkpoint(initial[0],directory/'base_u0.pt')
    write(directory/'BASELINE_VERIFIED.json',dict(at=now(),completed_round=0,both_blue_actors_equal_original=True))
    return directory/'base_u0.pt'

def main():
    from utils.mut_mappo_session import RunSessionOwner,pid_alive
    with RunSessionOwner(ROOT/'control/controller_lock'):
        adopt='--adopt-training' in sys.argv
        resume_eval='--resume-evaluation' in sys.argv
        if (ROOT/'control/controller.started').exists() and not (adopt or resume_eval): raise RuntimeError('controller already dispatched; inspect existing state')
        (ROOT/'control/controller.started').write_text(now(),encoding='utf-8')
        worker=read(ROOT/'control/worker.json')
        if resume_eval:
            if worker.get('active') is not False or read(ROOT/'control/state.json').get('status')!='FAILED_REVIEW_REQUIRED': raise RuntimeError('unsafe evaluation handoff')
        elif worker.get('phase')!=('training_u100' if adopt else 'references'): raise RuntimeError('unexpected handoff phase')
        write(ROOT/'control/controller.json',dict(pid=os.getpid(),status='ADOPTING_EXISTING_TRAINER' if adopt else 'WAITING_FOR_REFERENCES',at=now(),worker_pid=worker['pid']))
        while not (adopt or resume_eval) and pid_alive(worker['pid']): time.sleep(5)
        header=read(ROOT/'references/manifest.json')
        if len(header['cases'])!=72: raise RuntimeError('reference generation incomplete; inspect retained logs')
        directory=ROOT/'R5_original_u100'
        if not (adopt or resume_eval): snapshot(directory)
        config=directory/'config.json'
        if resume_eval:
            pass
        elif adopt:
            # Adopt supervision only; do not relaunch, stop, or modify the trainer.
            if 'training.train_mut_mappo' not in worker['command']: raise RuntimeError('unexpected trainer identity')
            while pid_alive(worker['pid']): time.sleep(5)
        else:
            execute(directory,'training_u100','training.train_mut_mappo',['--config',config,'--run-dir',directory/'run','--allow-candidate'])
        status=read(directory/'run/session/status.json')
        if status.get('completed_round')!=100: raise RuntimeError('not a complete 100-update boundary')
        if resume_eval:
            manifest=read(directory/'evaluation_base_u0/manifest.json')
            rr=rows(directory/'evaluation_base_u0/cases.jsonl')
            if manifest.get('status')!='COMPLETED' or len(rr)!=48 or not all(r.get('valid') for r in rr): raise RuntimeError('baseline evaluation is not complete')
            baseline=summarize(rr)
        else:
            base_checkpoint=initial_checkpoint(directory)
            baseline=evaluate(directory,base_checkpoint,'evaluation_base_u0')
        checkpoint=directory/'run/latest.pt'
        candidate=evaluate(directory,checkpoint,'evaluation_candidate_u100')
        effects=cf(directory,checkpoint)
        write(directory/'analysis.json',dict(at=now(),baseline=baseline,candidate=candidate,counterfactual=effects,
            selection_screen=any(r['engineering_screen_passed'] for r in effects.values()),
            purpose='Evidence for agent review. No automatic reward changes or final scientific success claim.'))
        write(ROOT/'control/state.json',dict(at=now(),status='AWAITING_AGENT_REWARD_ANALYSIS',version=directory.name,
            phase='review',completed_updates=100,automatic_parameter_changes=False,
            next='Agent examines complete feedback, writes the next reward rationale/configuration, then launches the next experiment if needed.'))

if __name__=='__main__':
    try: main()
    except BaseException as error:
        write(ROOT/'control/state.json',dict(at=now(),status='FAILED_REVIEW_REQUIRED',error=repr(error),automatic_restart=False))
        raise

