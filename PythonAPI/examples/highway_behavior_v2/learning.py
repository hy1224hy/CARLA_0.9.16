"""On-policy joint-action PPO with same-input, frozen-peer mutual KL."""
import numpy as np
import torch
from torch.distributions import Normal,Categorical,kl_divergence
from highwaytest.training.ppo import gae


def beta_at(steps,cfg):
    if cfg.method=='independent':return 0.,0.
    ramp=min(1.,max(0.,(steps-cfg.warmup_steps)/max(1,cfg.ramp_steps))) if cfg.ramp_steps else float(steps>=cfg.warmup_steps)
    return (cfg.beta*ramp*cfg.acceleration_kl_multiplier,
            cfg.beta*ramp*cfg.lane_kl_multiplier)


@torch.no_grad()
def collect(model,env,n,rng):
    obs,_=env.reset(seed=int(rng.integers(900000)))
    rows=[];episodes=[];total=0.;parts={}
    for _ in range(n):
        state=torch.as_tensor(obs['state'],dtype=torch.float32)
        mask=torch.as_tensor(obs['lane_mask'],dtype=torch.bool)
        acceleration,lane,z,logp=model.sample(state,mask)
        nxt,reward,term,trunc,info=env.step(dict(acceleration=acceleration.numpy(),lane=int(lane)))
        next_value=float(model.value(torch.as_tensor(nxt['state'],dtype=torch.float32)))
        rows.append((obs['state'],obs['lane_mask'],z.numpy(),int(lane),float(logp),reward,
                     float(model.value(state)),next_value,float(term),float(term or trunc)))
        total+=reward
        for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
        obs=nxt
        if term or trunc:
            episodes.append(dict(info,**{'return':total},reward_components=parts.copy()))
            total=0.;parts={};obs,_=env.reset(seed=int(rng.integers(900000)))
    keys=('obs','mask','z','lane','logp','reward','value','next_value','terminated','ended')
    data={key:torch.as_tensor(np.asarray(value)) for key,value in zip(keys,zip(*rows))}
    for key in data:
        data[key]=data[key].bool() if key=='mask' else data[key].long() if key=='lane' else data[key].float()
    return data,episodes


def update(model,optimizer,batch,cfg,teacher,beta_acceleration,beta_lane):
    with torch.no_grad():
        old=model.log_prob(batch['obs'],batch['mask'],batch['z'],batch['lane'])
        error=float((old-batch['logp']).abs().max())
        if error>1e-4:raise RuntimeError('Hybrid joint probability or action mask changed before PPO')
        advantages,returns=gae(batch['reward'],batch['value'],batch['next_value'],
            batch['terminated'],batch['ended'],cfg.gamma,cfg.gae_lambda)
        advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        peer_normal,peer_lane=teacher.distributions(batch['obs'],batch['mask'])
    records=[]
    for _ in range(cfg.epochs):
        for idx in torch.randperm(len(advantages)).split(cfg.minibatch):
            normal,category=model.distributions(batch['obs'][idx],batch['mask'][idx])
            logp=model.acceleration_log_prob(normal,batch['z'][idx])+category.log_prob(batch['lane'][idx])
            logratio=logp-batch['logp'][idx];ratio=logratio.exp()
            policy=-torch.minimum(ratio*advantages[idx],
                ratio.clamp(1-cfg.clip,1+cfg.clip)*advantages[idx]).mean()
            value=(model.value(batch['obs'][idx])-returns[idx]).square().mean()
            kl_acceleration=kl_divergence(normal,Normal(peer_normal.loc[idx],peer_normal.scale[idx])).sum(-1).mean()
            kl_lane=kl_divergence(category,Categorical(logits=peer_lane.logits[idx])).mean()
            entropy_acceleration=normal.entropy().sum(-1).mean()
            entropy_lane=category.entropy().mean()
            mutual=beta_acceleration*kl_acceleration+beta_lane*kl_lane
            loss=policy+.5*value-cfg.entropy_coef*entropy_acceleration-cfg.lane_entropy_coef*entropy_lane+mutual
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hybrid PPO loss')
            optimizer.zero_grad();loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite hybrid gradient')
            optimizer.step()
            values=(policy,value,entropy_acceleration,entropy_lane,kl_acceleration,kl_lane,mutual,
                    ((ratio-1)-logratio).mean(),((ratio-1).abs()>cfg.clip).float().mean(),gradient)
            records.append([float(x.detach()) for x in values])
    result=dict(zip(('policy_loss','value_loss','entropy_acceleration','entropy_lane',
        'mutual_kl_acceleration','mutual_kl_lane','mutual_loss','approx_kl','clip_fraction','gradient_norm'),
        np.mean(records,axis=0).tolist()))
    result.update(ratio_error=error,effective_beta_acceleration=beta_acceleration,effective_beta_lane=beta_lane,
        acceleration_std=float(model.log_std.clamp(-3,0).exp().detach()),
        lane_decision_fraction=float((batch['mask'].sum(-1)>1).float().mean()))
    return result
