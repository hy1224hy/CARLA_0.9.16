"""Task metrics share frozen definitions; audit modes are reported separately."""
import hashlib
import io
import re
import json
from pathlib import Path
import numpy as np
import torch
from highway_eval.evaluate import case_contract
from .config import StudyConfig
from .env import StudyEnv


@torch.no_grad()
def evaluate_model(model,cfg,split='development',cases=40,condition='nominal',execution='deterministic',action_seed=0,capture=False):
    from .runtime import summarize
    contract=[dict(seed=2000000+i,family=('follow','adjacent')[i%2]) for i in range(cases)] if split=='selection' else case_contract(split,cases)
    if execution not in ('deterministic','stochastic'):raise ValueError('execution')
    env=StudyEnv(cfg,condition);episodes=[];traces=[];was_training=model.training;model.eval()
    try:
        with torch.random.fork_rng():
            for case in contract:
                torch.manual_seed(action_seed*10000019+case['seed'])
                obs,_=env.reset(seed=case['seed'],options={'family':case['family']})
                total=0.;parts={};trace=[];changes=0;stable=env._lane(env.attacker);previous_acc=None;jerks=[]
                while True:
                    state=torch.as_tensor(obs['state']);mask=torch.as_tensor(obs['lane_mask'])
                    if execution=='deterministic':acc,lane=model.deterministic(state,mask)
                    else:acc,lane,_,_=model.sample(state,mask)
                    obs,reward,term,trunc,info=env.step(dict(acceleration=acc.numpy(),lane=int(lane)))
                    if previous_acc is not None:jerks.append(abs(float(acc[0])-previous_acc)*5*cfg.policy_frequency)
                    previous_acc=float(acc[0])
                    total+=reward
                    for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
                    a,v=env.attacker,env.target;actual=env._lane(a)
                    if actual!=stable and abs(a.position[1]-4*actual)<.5 and abs(a.heading)<np.deg2rad(10):changes+=1;stable=actual
                    if capture:trace.append(dict(time=info['time'],attacker_x=float(a.position[0]),attacker_y=float(a.position[1]),
                        target_x=float(v.position[0]),target_y=float(v.position[1]),attacker_speed=float(a.speed),target_speed=float(v.speed),
                        reference_speed=float(env.reference.vehicles[1].speed),acceleration=float(acc[0]),lane_action=int(lane),cut_count=info['cut_count'],
                        thw=max(float(a.position[0]-v.position[0]-5),0)/max(float(v.speed),2.) if a.position[0]>v.position[0]+5 else 100.))
                    if term or trunc:break
                episodes.append(dict(info,case_seed=case['seed'],**{'return':total},reward_components=parts,
                    completed_lane_changes=changes,action_seed=action_seed,command_jerk_mean=float(np.mean(jerks)) if jerks else 0.,
                    collision_latency_capped=float(info['time']) if cfg.level==3 and info['attack_success'] else cfg.duration))
                if capture:traces.append(dict(case_seed=case['seed'],family=case['family'],steps=trace))
    finally:env.close();model.train(was_training)
    overall=summarize(episodes,cfg)
    return dict(overall=overall,families={f:summarize([e for e in episodes if e['family']==f],cfg) for f in ['follow','adjacent'] if any(e['family']==f for e in episodes)},episodes=episodes,trajectories=traces)


def audit(run,output,checkpoint='latest',team='A',split='development',cases=40,condition='nominal',execution='deterministic',action_seed=0):
    from .runtime import load,check,ActorCritic,write_json
    run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if (checkpoint not in ('latest','selected','init') and not re.fullmatch(r'ckpt_\d+',checkpoint)) or team not in ('A','B'):raise ValueError('checkpoint selection')
    payload=(run/(checkpoint+'.pt')).read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);check(state)
    cfg=StudyConfig.from_dict(state['config']);torch.set_num_threads(cfg.threads)
    model=ActorCritic(cfg.hidden,cfg.state_dim)
    chosen=state['selection']['team'] if checkpoint=='selected' else team
    model.load_state_dict(state['model'] if checkpoint=='selected' else state['models'][team])
    result=evaluate_model(model,cfg,split,cases,condition,execution,action_seed,capture=True)
    output.mkdir(parents=True);(output/'checkpoint.pt').write_bytes(payload)
    for key in ['episodes','trajectories']:
        (output/(key+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in result[key]),encoding='utf-8')
    report=dict(run=cfg.run_id,config=cfg.to_dict(),checkpoint=checkpoint,team=chosen,
        steps=state['selection']['train_steps'] if checkpoint=='selected' else state['train_steps'],
        split=split,condition=condition,execution=execution,action_seed=action_seed,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),overall=result['overall'],families=result['families'])
    write_json(output/'cases.json',case_contract(split,cases))
    write_json(output/'metrics.json',report)
    return report
