"""Observational diagnostics around the production PPO; never changes losses/RNG."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import torch


def independent_gae(rows, boundaries, bootstrap, critic, normalizer, gamma, lam):
    result = []
    for (start, end), state in zip(boundaries, bootstrap):
        value = 0.
        if state is not None:
            with torch.no_grad():
                value = float(normalizer.denormalize(critic(torch.tensor(state).float()[None], rows[0][0])).item())
        episode = rows[1][start:end]
        acc, values = 0., []
        for row in reversed(episode):
            delta = row.reward + gamma * value * (1-row.terminated) - row.value
            acc = delta + gamma*lam*(1-row.done)*acc
            values.append(acc)
            value = row.value
        result.extend(reversed(values))
    return np.array(result)


def gradient_vector(loss, actor):
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()), allow_unused=True)
    return torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten()
                      for g,p in zip(gradients, actor.parameters())]).detach()


@contextmanager
def instrument(directory):
    from training import mut_mappo_learning as learning
    from training import train_mut_mappo as runner
    from training.checkpoint_protocol import capture_rng_state, restore_rng_state
    from training.action_protocol import action_tanh_to_env
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original_sample, original_episode = learning.TeamLearner.sample, runner._episode
    original_prepare, original_update = learning.prepare_mappo_update_batch, learning.run_mappo_ppo_update
    current = {'index':0, 'action_checks':0, 'max_action_error':0., 'max_logp_error':0., 'episodes':[]}

    def sample(self, *args, **kw):
        controls, cache = original_sample(self, *args, **kw)
        for aid, row in cache.items():
            error = float(np.max(np.abs(controls[aid]-action_tanh_to_env(np.tanh(row['action_z'])))))
            with torch.no_grad():
                lp,_ = self.actors[aid].evaluate_actions_z(torch.tensor(row['obs']).float()[None],
                                                       torch.tensor(row['action_z']).float()[None])
            le = abs(float(lp.item())-row['logp'])
            current['action_checks'] += 1
            current['max_action_error'] = max(current['max_action_error'],error)
            current['max_logp_error'] = max(current['max_logp_error'],le)
            if error > 1e-6 or le > 1e-5:
                raise ValueError('real rollout action/logp mismatch')
        return controls, cache

    def episode(runtime, case, **kw):
        buffer = kw.get('buffer')
        if buffer is None:
            return original_episode(runtime, case, **kw)
        starts = {aid:len(buffer.data[aid]) for aid in (2,3)}
        if current['index'] < 2:
            kw['trace_path'] = directory/('trace_'+runtime.team+'_'+case['case_key']+'_'+str(kw.get('seed'))+'.jsonl')
        record = original_episode(runtime, case, **kw)
        audit = {}
        for aid in (2,3):
            rows = buffer.data[aid][starts[aid]:]
            folded = math.fsum(runtime.config['learning']['gamma']**i*r.reward for i,r in enumerate(rows))
            # Tail settlement must preserve the entire observed team return.
            error = abs(folded-record['team_discounted_return']) if rows else None
            if error is not None and error > 1e-5:
                raise ValueError('post-departure team return mismatch')
            audit[aid] = dict(rows=len(rows),folded_return_error=error,
                terminated=record['agent_terminated'][aid],truncated=record['agent_truncated'][aid],
                tail=record['team_tail_credit'].get(aid))
        current['episodes'].append(dict(case=case['case_key'],roles=audit))
        return record

    def prepare(buffer, critic, ids, device, **kw):
        prepared = original_prepare(buffer,critic,ids,device,**kw)
        errors={}
        for aid,t in prepared.per_agent_tensors.items():
            expected = independent_gae(((aid,),buffer.data[aid]),buffer.episode_boundaries[aid],
                buffer.episode_bootstrap_states[aid],critic,kw['value_normalizer'],kw['gamma'],kw['gae_lambda'])
            error = float(np.max(np.abs(expected-t['raw_adv'].cpu().numpy())))
            if error > 1e-3:
                raise ValueError('independent GAE mismatch')
            errors[aid]=error
        current['gae_errors']=errors
        if current['index']<2:
            torch.save(dict(buffer=buffer,prepared=prepared),directory/('batch_'+str(current['index'])+'.pt'))
        return prepared

    def update(prepared, actors, actor_optims, critic, critic_optim, ids, **kw):
        before, diagnostics = {}, {}
        rng = capture_rng_state()
        try:
            for aid,t in prepared.per_agent_tensors.items():
                actor=actors[aid]
                lp,_=actor.evaluate_actions_z(t['obs'],t['act'])
                before[aid]=lp.detach()
                ratio=torch.exp(lp-t['old_logp'])
                loss=-torch.minimum(ratio*t['adv'],ratio.clamp(1-kw['clip_eps'],1+kw['clip_eps'])*t['adv']).mean()
                policy=gradient_vector(loss,actor)
                entropy,_=actor.estimate_post_tanh_entropy(t['obs'])
                exploration=gradient_vector(-kw['ent_coef']*entropy.mean(),actor)
                norm=float(policy.norm()); enorm=float(exploration.norm())
                diagnostics[aid]=dict(n=len(lp),positive_advantage=int((t['adv']>0).sum()),
                    policy_grad_norm=norm,entropy_grad_norm=enorm,
                    entropy_to_policy_grad_ratio=enorm/max(norm,1e-12),
                    gradient_cosine=float(torch.dot(policy,exploration)/max(norm*enorm,1e-12)),
                    raw_adv_std=float(t['raw_adv'].std()),old_surrogate=float(-loss.detach()))
        finally:
            restore_rng_state(rng)
        result=original_update(prepared,actors,actor_optims,critic,critic_optim,ids,**kw)
        with torch.no_grad():
            for aid,t in prepared.per_agent_tensors.items():
                lp,_=actors[aid].evaluate_actions_z(t['obs'],t['act'])
                delta=lp-before[aid];ratio=torch.exp(lp-t['old_logp']);positive=t['adv']>0
                objective=torch.minimum(ratio*t['adv'],ratio.clamp(1-kw['clip_eps'],1+kw['clip_eps'])*t['adv']).mean()
                diagnostics[aid].update(new_surrogate=float(objective),
                    surrogate_improvement=float(objective)-diagnostics[aid]['old_surrogate'],
                    positive_adv_logp_delta=float(delta[positive].mean()) if positive.any() else None,
                    advantage_logp_alignment=float((t['adv']*delta).mean()),
                    actor_lr=actor_optims[aid].param_groups[0]['lr'])
        runner.append_record(directory/'updates.jsonl',dict(index=current['index'],context=kw['context'],
            roles=diagnostics,action_checks=current['action_checks'],max_action_error=current['max_action_error'],
            max_logp_error=current['max_logp_error'],gae_errors=current['gae_errors'],episodes=current['episodes']))
        current.update(index=current['index']+1,action_checks=0,max_action_error=0.,max_logp_error=0.,episodes=[])
        return result

    with patch.object(learning.TeamLearner,'sample',sample),patch.object(runner,'_episode',episode), \
         patch.object(learning,'prepare_mappo_update_batch',prepare),patch.object(learning,'run_mappo_ppo_update',update):
        yield
