"""Three serial rollouts with frozen within-batch policies and role-keyed noise.

Only the factual rollout enters PPO. Replaced blue uses the deterministic
original base policy; the other blue retains its current stochastic policy
and the same per-step/role Gaussian random seed. No simulator world branching.
"""
import hashlib
import random
import numpy as np
import torch
from evaluation.r6_counterfactual import RedOutcomeIntegral,paired_credit,settle_credit
from training.checkpoint_protocol import capture_rng_state,restore_rng_state
from training.mut_mappo_config import make_actors
from training.train_no_rllib_mappo_robust_v2 import build_active_step_cache


def role_noise_seed(seed,step,aid):
    return int.from_bytes(hashlib.sha256(f'r6:{seed}:{step}:{aid}'.encode()).digest()[:8],'little')%(2**31)


class CoupledSampler:
    def __init__(self,team,base,replaced,seed):
        self.team=team;self.base=base;self.replaced=set(replaced);self.seed=seed;self.step=0

    @torch.no_grad()
    def sample(self,obs,state,active):
        from training.train_mut_mappo import policy_controls
        self.step+=1;controls={};cache={}
        for aid in (2,3):
            if not active.get(aid):continue
            if aid in self.replaced:
                controls.update(policy_controls({aid:self.base[aid]},obs,active,self.team.device,True))
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(role_noise_seed(self.seed,self.step,aid))
                    actions,rows=build_active_step_cache(obs,{aid:True},state,self.team.actors,
                        self.team.critic,[aid],self.team.device,context='R6 matched actual policy',
                        value_normalizer=self.team.value_normalizer)
                    controls.update(actions);cache.update(rows)
        return controls,cache


def base_actors(runtime):
    if not hasattr(runtime,'_r6_base_actors'):
        before=capture_rng_state()
        try:
            source=runtime.config['_red_payload']
            runtime._r6_base_actors=make_actors({2:source['actors'][0],3:source['actors'][1]},
                source['training_config'],(2,3),runtime.config['device'])
        finally:restore_rng_state(before)
    return runtime._r6_base_actors


def measured_rollout(runtime,case,seed,*,sampler=None,buffer=None,trace_path=None,timeline_path=None):
    from training.train_mut_mappo import _episode
    t=runtime.config['task'];collector=RedOutcomeIntegral(t['timestep'],t['max_steps'])
    before=capture_rng_state()
    try:
        random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
        r=_episode(runtime,case,learner=sampler,buffer=buffer,seed=seed,deterministic=True,
            trace_path=trace_path,step_callback=collector.step)
    finally:restore_rng_state(before)
    if timeline_path is not None:
        from training.checkpoint_protocol import atomic_write_json
        atomic_write_json(dict(case_key=case['case_key'],team=runtime.team,seed=seed,steps=collector.timeline),timeline_path)
    r.update(collector.finish())
    return r


def collect_training_case(runtime,case,team,buffer,seed):
    starts={i:len(buffer.data[i]) for i in (2,3)};base=base_actors(runtime)
    actual=measured_rollout(runtime,case,seed,sampler=CoupledSampler(team,base,(),seed),buffer=buffer)
    actual['replacement']='none'
    if actual.get('valid') is not True:raise ValueError('invalid factual R6 rollout cannot train PPO')
    aux=[];credits={}
    for i in (2,3):
        cf=measured_rollout(runtime,case,seed,sampler=CoupledSampler(team,base,(i,),seed))
        cf.update(replacement='blue'+str(i),record_kind='counterfactual_training_auxiliary',ppo_samples=False)
        aux.append(cf)
        credits[i]=paired_credit(actual,cf,i,runtime.config['task']['team_reward'])
    settle_credit(actual,buffer,starts,credits)
    actual['counterfactual_rollout_costs']={k:sum(r.get('tick_counts',{}).get(k,0) for r in [actual,*aux])
        for k in set().union(*(r.get('tick_counts',{}).keys() for r in [actual,*aux]))}
    actual['counterfactual_auxiliary_rollouts']=2
    actual['counterfactual_policy_protocol']='current-other-policy + role/step-keyed common Gaussian noise; deterministic original-base replacement'
    return actual,aux
