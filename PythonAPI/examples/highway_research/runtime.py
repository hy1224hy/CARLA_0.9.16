"""Same-network twin training with source contracts, resume and recoverable IO."""
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from highway_l1_style_v4.model import HybridActorCritic as ActorCritic
from highwaytest.training.checkpoint import runtime_versions
from highwaytest.training.locking import run_lock
from .config import PROTOCOL,StudyConfig,BACKENDS
from .env import StudyEnv
from .learning import update,beta_at


def source_files():
    paths={Path(__file__).parent/name for name in ['__init__.py','config.py','env.py','learning.py','runtime.py','evaluate.py']}
    for package in BACKENDS.values():paths.update(importlib.import_module(package+'.runner').source_files())
    return sorted(paths,key=str)


def digest():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for p in source_files():
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode())
    return h.hexdigest()


def atomic_save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    torch.save(value,temp)
    for attempt in range(10):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1*(attempt+1))


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def load(path):return torch.load(io.BytesIO(Path(path).read_bytes()),map_location='cpu',weights_only=False)


def check(state):
    if state['protocol']!=PROTOCOL or state['source_sha256']!=digest() or state['packages']!=runtime_versions():
        raise ValueError('Checkpoint source/runtime/protocol mismatch')


def summarize(rows,cfg):
    result=importlib.import_module(cfg.backend+'.runner').summarize(rows)
    if rows:result['canonical_return']=float(np.mean([e.get('canonical_return',e['return']) for e in rows]))
    for key in ('command_jerk_mean','collision_latency_capped'):
        if rows and all(key in r for r in rows):result[key]=float(np.mean([r[key] for r in rows]))
    return result


def rank(report,level):
    collision=report['collision_rate'] if level!=3 else 0.
    bad=report['invalid_rate']+collision+report['grade_violation_rate']
    ok=report['baseline_valid_rate']==1 and max(report['invalid_rate'],collision,report['grade_violation_rate'])<=.125
    return (int(ok),report['attack_success_rate'] if ok else -bad,report.get('style_success_rate',0.),report['utility'])


@torch.no_grad()
def collect(model,env,n,rng):
    """Carry incomplete episodes across rollouts; snapshot cursor with physics."""
    if '_collector' not in env.__dict__:
        obs,_=env.reset(seed=int(rng.integers(900000)))
        env._collector=dict(obs=obs,total=0.,parts={})
    cursor=env._collector;obs=cursor['obs'];rows=[];episodes=[]
    for _ in range(n):
        state=torch.as_tensor(obs['state']);mask=torch.as_tensor(obs['lane_mask'])
        acc,lane,z,logp=model.sample(state,mask)
        nxt,reward,term,trunc,info=env.step(dict(acceleration=acc.numpy(),lane=int(lane)))
        rows.append((obs['state'],obs['lane_mask'],z.numpy(),int(lane),float(logp),reward,
            float(model.value(state)),float(model.value(torch.as_tensor(nxt['state']))),float(term),float(term or trunc)))
        cursor['total']+=reward
        for k,v in info['reward_components'].items():cursor['parts'][k]=cursor['parts'].get(k,0.)+v
        obs=nxt
        if term or trunc:
            episodes.append(dict(info,**{'return':cursor['total']},reward_components=dict(cursor['parts'])))
            cursor['total']=0.;cursor['parts']={}
            obs,_=env.reset(seed=int(rng.integers(900000)))
    cursor['obs']=obs
    keys=['obs','mask','z','lane','logp','reward','value','next_value','terminated','ended']
    batch={k:torch.as_tensor(np.asarray(v)) for k,v in zip(keys,zip(*rows))}
    for k in batch:batch[k]=batch[k].bool() if k=='mask' else batch[k].long() if k=='lane' else batch[k].float()
    return batch,episodes


def train(cfg,output,resume=False,max_rounds=None):
    cfg.validate();output=Path(output).resolve()
    with run_lock(output):return _train(cfg,output,resume,max_rounds)


def _train(cfg,output,resume,max_rounds):
    torch.set_num_threads(cfg.threads);torch.manual_seed(cfg.seed)
    initial=ActorCritic(cfg.hidden,cfg.state_dim)
    models={t:copy.deepcopy(initial) for t in ['A','B']}
    teachers={t:copy.deepcopy(initial).eval().requires_grad_(False) for t in models}
    if cfg.algorithm=='a2c':
        optimizers={t:torch.optim.RMSprop(m.parameters(),lr=cfg.lr,alpha=.99,eps=1e-5) for t,m in models.items()}
    else:optimizers={t:torch.optim.Adam(m.parameters(),lr=cfg.lr,eps=1e-5) for t,m in models.items()}
    rngs={t:np.random.default_rng(cfg.seed+10007*i) for i,t in enumerate(models)}
    torch_states={}
    for i,t in enumerate(models):torch.manual_seed(cfg.seed+10007*i);torch_states[t]=torch.get_rng_state()
    identity=digest();packages=runtime_versions();steps=round_id=eval_steps=0
    best=(-1,-float('inf'),0.,0.);selection=None;best_model=None;inherited=0.
    if resume:
        state=load(output/'latest.pt');check(state)
        if state['config']!=cfg.to_dict():raise ValueError('Resume config changed')
        for t in models:
            models[t].load_state_dict(state['models'][t]);teachers[t].load_state_dict(state['teachers'][t])
            optimizers[t].load_state_dict(state['optimizers'][t]);rngs[t].bit_generator.state=state['numpy_rng'][t]
        torch_states=state['torch_rng'];steps=state['train_steps'];round_id=state['round']
        eval_steps=state['evaluation_steps'];best=tuple(state['best_rank']);selection=state['selection'];best_model=state['best_model'];inherited=state['wall_seconds']
        # Discard partial updates newer than the committed checkpoint.
        for filename in ['metrics.jsonl','episodes.jsonl','validation.jsonl']:
            p=output/filename;kept=[]
            for line in p.read_text().splitlines() if p.exists() else []:
                try:row=json.loads(line)
                except ValueError:continue
                if row['round']<=round_id:kept.append(row)
            p.write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')
    else:
        output.mkdir(parents=True,exist_ok=False);write_json(output/'config.json',cfg.to_dict())
        write_json(output/'manifest.json',dict(protocol=PROTOCOL,source_sha256=identity,packages=packages,
            budget='A+B sampled decisions; selection steps counted separately',algorithm=cfg.algorithm,mutual=cfg.mutual))
        root=Path(__file__).resolve().parents[1]
        for path in source_files():
            target=output/'source'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    def payload():return dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=packages)
    if not resume:atomic_save(output/'init.pt',dict(payload(),models={t:m.state_dict() for t,m in models.items()},train_steps=0))
    writer=SummaryWriter(str(output/'tensorboard'),purge_step=steps if resume else None)
    envs=state['envs'] if resume else {t:StudyEnv(cfg) for t in models}
    start=time.perf_counter();start_round=round_id
    def append(filename,data):
        with (output/filename).open('a',encoding='utf-8') as stream:stream.write(json.dumps(data,allow_nan=False)+'\n')
    def log(prefix,data):
        for key,value in data.items():
            if isinstance(value,(float,int)) and np.isfinite(value):writer.add_scalar(prefix+'/'+key,value,steps)
    def selected_payload():return dict(payload(),model=best_model,selection=selection)
    try:
        while steps<cfg.total_steps and (max_rounds is None or round_id-start_round<max_rounds):
            clock=steps;n=min(cfg.rollout_steps,(cfg.total_steps-steps)//2);round_id+=1
            for t in models:
                torch.set_rng_state(torch_states[t]);batch,episodes=collect(models[t],envs[t],n,rngs[t])
                ba,bl=beta_at(clock,cfg,t)
                stats=update(models[t],optimizers[t],batch,cfg,teachers['B' if t=='A' else 'A'],ba,bl,steps=clock)
                torch_states[t]=torch.get_rng_state();steps=clock+2*n
                stats['mean_step_reward']=float(batch['reward'].mean())
                append('metrics.jsonl',dict(stats,team=t,round=round_id,train_steps=steps));log('train/'+t,stats)
                for episode in episodes:append('episodes.jsonl',dict(episode,team=t,round=round_id,train_steps=steps))
                if episodes:
                    log('episode/'+t,summarize(episodes,cfg))
                    for k in episodes[0]['reward_components']:writer.add_scalar('reward/'+t+'/'+k,np.mean([e['reward_components'].get(k,0.) for e in episodes]),steps)
            if round_id%cfg.refresh_rounds==0:teachers={t:copy.deepcopy(m).eval().requires_grad_(False) for t,m in models.items()}
            if selection is None or steps==cfg.total_steps or steps//cfg.eval_interval!=clock//cfg.eval_interval:
                from .evaluate import evaluate_model
                reports={t:evaluate_model(m,cfg,'selection',cfg.eval_cases) for t,m in models.items()}
                eval_steps+=sum(e['episode_steps'] for r in reports.values() for e in r['episodes'])
                append('validation.jsonl',dict(round=round_id,train_steps=steps,reports=reports))
                for t,r in reports.items():log('validation/'+t,r['overall'])
                winner=max(reports,key=lambda t:rank(reports[t]['overall'],cfg.level));candidate=rank(reports[winner]['overall'],cfg.level)
                if candidate>=best:
                    best=candidate;best_model=copy.deepcopy(models[winner].state_dict())
                    selection=dict(team=winner,train_steps=steps,eligible=bool(best[0]),overall=reports[winner]['overall'])
                    atomic_save(output/'selected.pt',selected_payload())
            atomic_save(output/'latest.pt',dict(payload(),models={t:m.state_dict() for t,m in models.items()},
                teachers={t:m.state_dict() for t,m in teachers.items()},optimizers={t:o.state_dict() for t,o in optimizers.items()},
                numpy_rng={t:r.bit_generator.state for t,r in rngs.items()},torch_rng=torch_states,train_steps=steps,round=round_id,
                evaluation_steps=eval_steps,best_rank=best,selection=selection,best_model=best_model,envs=envs,wall_seconds=inherited+time.perf_counter()-start))
            if steps//100000!=clock//100000 or steps==cfg.total_steps:
                nominal=(steps//100000)*100000 or steps
                atomic_save(output/f'ckpt_{nominal}.pt',dict(payload(),models={t:m.state_dict() for t,m in models.items()},train_steps=steps))
            writer.flush();print(cfg.run_id,steps,'/',cfg.total_steps,flush=True)
    finally:
        writer.close()
        for env in envs.values():env.close()
    result=dict(status='complete' if steps==cfg.total_steps else 'paused',train_steps=steps,selection=selection,
        wall_seconds=inherited+time.perf_counter()-start,evaluation_steps=eval_steps)
    write_json(output/'summary.json',result);return result
