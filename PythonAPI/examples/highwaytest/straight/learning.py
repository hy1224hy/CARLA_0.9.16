"""On-policy PPO; peer targets never receive student gradients."""
import numpy as np
import torch
from torch.distributions import Normal,kl_divergence
from ..training.ppo import gae


def beta_at(steps,beta,warmup,ramp):
    return beta*min(1.,max(0.,(steps-warmup)/max(1,ramp))) if ramp else (beta if steps>=warmup else 0.)


def peer_loss(student,teacher,obs):
    with torch.no_grad():
        target=teacher.distribution(obs)
    return kl_divergence(student.distribution(obs),target).sum(-1).mean()


@torch.no_grad()
def collect(model,env,n,rng):
    obs,_=env.reset(seed=int(rng.integers(900000)))
    rows=[];episodes=[];total=0.;components={}
    for _ in range(n):
        o=torch.as_tensor(obs,dtype=torch.float32)
        a,z,lp=model.sample(o)
        nxt,r,term,trunc,info=env.step(a.numpy())
        for key,value in info.get('reward_components',{}).items():components[key]=components.get(key,0.)+value
        rows.append((obs,z.numpy(),float(lp),r,float(model.value(o)),float(model.value(torch.as_tensor(nxt,dtype=torch.float32))),float(term),float(term or trunc)))
        total+=r;obs=nxt
        if term or trunc:
            episodes.append(dict(info,**{'return':total,'reward_components':components}))
            total=0.;components={};obs,_=env.reset(seed=int(rng.integers(900000)))
    keys=('obs','z','logp','reward','value','next_value','terminated','ended')
    return {k:torch.as_tensor(np.asarray(v),dtype=torch.float32) for k,v in zip(keys,zip(*rows))},episodes


def update(model,optimizer,batch,cfg,teacher,beta):
    with torch.no_grad():
        initial=model.log_prob(model.distribution(batch['obs']),batch['z'])
        error=float((initial-batch['logp']).abs().max())
        if error>1e-4:raise RuntimeError('PPO old-policy probability mismatch')
        adv,returns=gae(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['ended'],cfg.gamma,cfg.gae_lambda)
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
        peer=teacher.distribution(batch['obs'])
    records=[]
    for _ in range(cfg.epochs):
        for idx in torch.randperm(len(adv)).split(cfg.minibatch):
            d=model.distribution(batch['obs'][idx]);lp=model.log_prob(d,batch['z'][idx])
            ratio=(lp-batch['logp'][idx]).exp()
            policy=-torch.minimum(ratio*adv[idx],ratio.clamp(1-cfg.clip,1+cfg.clip)*adv[idx]).mean()
            value=(model.value(batch['obs'][idx])-returns[idx]).square().mean()
            kl=kl_divergence(d,Normal(peer.loc[idx],peer.scale[idx])).sum(-1).mean()
            entropy=d.entropy().sum(-1).mean()
            loss=policy+.5*value-cfg.entropy_coef*entropy+beta*kl
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            logratio=lp-batch['logp'][idx]
            records.append([float(x.detach()) for x in (policy,value,entropy,kl,beta*kl,((ratio-1)-logratio).mean(),((ratio-1).abs()>cfg.clip).float().mean(),grad)])
    result=dict(zip(('policy_loss','value_loss','entropy','mutual_kl','mutual_loss','approx_kl','clip_fraction','gradient_norm'),np.mean(records,axis=0).tolist()))
    result.update(effective_beta=beta,ratio_error=error,acceleration_std=float(model.log_std[0].clamp(-3,0).exp().detach()),steering_std=float(model.log_std[1].clamp(-3,0).exp().detach()))
    return result
