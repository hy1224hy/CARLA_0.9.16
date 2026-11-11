"""Hybrid PPO, synchronous A2C and TRPO with actual joint-distribution KL."""
import math
import torch
from torch.distributions import Normal,Categorical,kl_divergence
from highwaytest.training.ppo import gae


def ramp(steps,cfg):
    return min(1.,max(0.,(steps-cfg.warmup_steps)/max(1,cfg.ramp_steps))) if cfg.ramp_steps else float(steps>=cfg.warmup_steps)


def beta_at(steps,cfg,team):
    if cfg.mutual=='off' or (cfg.mutual=='one_way' and team=='A'):return 0.,0.
    fraction=ramp(steps,cfg)
    return cfg.beta*fraction*cfg.acc_multiplier,cfg.beta*fraction*cfg.lane_multiplier


def detached(normal,lane):
    return Normal(normal.loc.detach(),normal.scale.detach()),Categorical(logits=lane.logits.detach())


def joint_kl(old_n,old_l,new_n,new_l):
    return (kl_divergence(old_n,new_n).sum(-1)+kl_divergence(old_l,new_l)).mean()


def update(model,optimizer,batch,cfg,teacher,beta_acc,beta_lane,steps=0):
    obs,mask=batch['obs'],batch['mask']
    with torch.no_grad():
        old_log=model.log_prob(obs,mask,batch['z'],batch['lane'])
        error=float((old_log-batch['logp']).abs().max())
        if error>1e-4:raise RuntimeError('on-policy probability mismatch')
        advantages,returns=gae(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['ended'],cfg.gamma,cfg.gae_lambda)
        if cfg.algorithm!='a2c':advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        old_n,old_l=detached(*model.distributions(obs,mask))
        peer_n,peer_l=detached(*teacher.distributions(obs,mask))
    if cfg.algorithm=='trpo':
        result=trpo(model,optimizer,batch,advantages,returns,old_log,old_n,old_l,cfg)
        result.update(ratio_error=error,mutual_loss=0.,effective_beta_acceleration=0.,effective_beta_lane=0.)
        return result
    records=[]
    indices=[torch.arange(len(obs))] if cfg.algorithm=='a2c' else None
    for _ in range(1 if cfg.algorithm=='a2c' else cfg.epochs):
        batches=indices if indices is not None else torch.randperm(len(obs)).split(cfg.minibatch)
        for idx in batches:
            normal,lane=model.distributions(obs[idx],mask[idx])
            logp=model.acceleration_log_prob(normal,batch['z'][idx])+lane.log_prob(batch['lane'][idx])
            ratio=(logp-old_log[idx]).exp()
            if cfg.algorithm=='a2c':policy=-(logp*advantages[idx]).mean()
            else:policy=-torch.minimum(ratio*advantages[idx],ratio.clamp(1-cfg.clip,1+cfg.clip)*advantages[idx]).mean()
            value=(model.value(obs[idx])-returns[idx]).square().mean()
            ka=kl_divergence(normal,Normal(peer_n.loc[idx],peer_n.scale[idx])).sum(-1).mean()
            kl=kl_divergence(lane,Categorical(logits=peer_l.logits[idx])).mean()
            # Baselines do not accidentally receive mutual regularization.
            active=cfg.algorithm=='ppo' and cfg.mutual!='off'
            mutual=(beta_acc*ka+beta_lane*kl) if active else ka*0.
            self_penalty=joint_kl(Normal(old_n.loc[idx],old_n.scale[idx]),Categorical(logits=old_l.logits[idx]),normal,lane)
            ea=normal.entropy().sum(-1).mean();el=lane.entropy().mean()
            loss=policy+.5*value-cfg.entropy_coef*ea-cfg.lane_entropy_coef*el+mutual+cfg.self_kl*ramp(steps,cfg)*self_penalty
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite learning loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            records.append([float(x.detach()) for x in [policy,value,ea,el,ka,kl,mutual,norm]])
    means=torch.tensor(records).mean(0).tolist()
    result=dict(zip(['policy_loss','value_loss','entropy_acceleration','entropy_lane','mutual_kl_acceleration','mutual_kl_lane','mutual_loss','gradient_norm'],means))
    with torch.no_grad():result['actual_kl']=float(joint_kl(old_n,old_l,*model.distributions(obs,mask)))
    with torch.no_grad():
        decision=mask.sum(-1)>1
        _,category=model.distributions(obs,mask)
        lane_kls=kl_divergence(category,peer_l)
        result['lane_decision_fraction']=float(decision.float().mean())
        result['mutual_kl_lane_decision']=float(lane_kls[decision].mean()) if decision.any() else 0.
    result.update(ratio_error=error,optimizer_steps=float(len(records)),effective_beta_acceleration=beta_acc if cfg.mutual!='off' else 0.,effective_beta_lane=beta_lane if cfg.mutual!='off' else 0.)
    return result


def flat_grad(value,params,create_graph=False):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(value,params,create_graph=create_graph,retain_graph=True)])


def assign(params,flat):
    cursor=0
    with torch.no_grad():
        for p in params:
            count=p.numel();p.copy_(flat[cursor:cursor+count].view_as(p));cursor+=count


def conjugate_gradient(operator,b,steps):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=torch.dot(r,r)
    for _ in range(steps):
        ap=operator(p);denom=torch.dot(p,ap)
        if not torch.isfinite(denom) or denom<=1e-12:break
        alpha=rr/denom;x+=alpha*p;r-=alpha*ap;next_rr=torch.dot(r,r)
        if next_rr<1e-12:break
        p=r+(next_rr/rr)*p;rr=next_rr
    return x


def trpo(model,optimizer,batch,advantages,returns,old_log,old_n,old_l,cfg):
    params=[p for name,p in model.named_parameters() if not name.startswith('critic.')]
    original=torch.cat([p.detach().reshape(-1) for p in params]).clone()
    obs,mask=batch['obs'],batch['mask']
    def objective():
        logp=model.log_prob(obs,mask,batch['z'],batch['lane'])
        return ((logp-old_log).exp()*advantages).mean()
    before=objective();gradient=flat_grad(before,params).detach()
    def fisher(v):
        kl=joint_kl(old_n,old_l,*model.distributions(obs,mask))
        first=flat_grad(kl,params,create_graph=True)
        return flat_grad((first*v).sum(),params).detach()+cfg.trpo_damping*v
    direction=conjugate_gradient(fisher,gradient,cfg.trpo_cg_steps)
    curvature=torch.dot(direction,fisher(direction))
    accepted=False;actual=0.;surrogate=float(before.detach())
    if torch.isfinite(curvature) and curvature>1e-12:
        step=direction*math.sqrt(2*cfg.trpo_delta/float(curvature))
        for backtrack in range(cfg.trpo_backtracks):
            assign(params,original+(0.5**backtrack)*step)
            with torch.no_grad():
                actual=float(joint_kl(old_n,old_l,*model.distributions(obs,mask)))
                surrogate=float(objective())
            if math.isfinite(actual) and actual<=cfg.trpo_delta and surrogate>float(before.detach()):accepted=True;break
    if not accepted:assign(params,original);actual=0.;surrogate=float(before.detach())
    # Actor update is a constrained line search; critic alone uses Adam.
    values=[]
    for _ in range(cfg.critic_epochs):
        loss=(model.value(obs)-returns).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5);optimizer.step();values.append(float(loss.detach()))
    return dict(policy_loss=-surrogate,value_loss=sum(values)/len(values),actual_kl=actual,
        trpo_accepted=float(accepted),optimizer_steps=float(cfg.critic_epochs),gradient_norm=float(gradient.norm()))
