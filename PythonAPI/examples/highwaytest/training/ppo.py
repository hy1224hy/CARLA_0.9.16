"""PPO over joint time rows: recompute BOTH messages for each current-policy row."""
import numpy as np
import torch
from torch.distributions import kl_divergence


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def gae(rewards, values, next_values, terminated, ended, gamma, lam):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - ended[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


@torch.no_grad()
def collect(model, env, n, rng, cursor, preview=None):
    rows = {k: [] for k in ("obs", "links", "state", "z", "logp", "reward", "value",
                            "next_value", "terminated", "ended")}
    families, episodes = [], []
    obs, info = cursor if cursor else env.reset(seed=int(rng.integers(0, 900_000)))
    episode_return = np.zeros(2, dtype=np.float64)
    for _ in range(n):
        action, z, logp = model.sample(tensor(obs), tensor(info["links"]))
        value = model.values(tensor(info["state"]))
        next_obs, reward, term, trunc, next_info = env.step(action.numpy())
        episode_return += reward
        if preview is not None:
            preview.step(env, next_info, action.numpy(), reward)
        values = (obs, info["links"], info["state"], z.numpy(), logp.numpy(), reward,
                  value.numpy(), model.values(tensor(next_info["state"])).numpy(), float(term), float(term or trunc))
        for key, item in zip(rows, values):
            rows[key].append(item)
        families.append(info["family"])
        obs, info = next_obs, next_info
        if term or trunc:
            episodes.append({k: v for k, v in info.items() if k not in ("state", "links")} |
                            {"return": float(episode_return.mean()), "return_agent_0": float(episode_return[0]),
                             "return_agent_1": float(episode_return[1]), "terminated": term, "truncated": trunc})
            episode_return[:] = 0
            obs, info = env.reset(seed=int(rng.integers(0, 900_000)))
    return {k: tensor(v) for k, v in rows.items()}, families, episodes, (obs, info)


@torch.no_grad()
def check_on_policy(model,batch,tolerance=1e-5):
    """Keep the invariant; verify rare batch-kernel roundoff in sampling's row shape."""
    initial=model.distribution(batch['obs'],batch['links'])
    error=float((TeamLogProb(initial,batch['z'])-batch['logp']).abs().max())
    result={'batch_error':error,'rowwise_error':0.,'rowwise_checked':False}
    if not np.isfinite(error):raise RuntimeError('nonfinite on-policy probability check')
    if error>tolerance:
        rows=[model.distribution(o,l) for o,l in zip(batch['obs'],batch['links'])]
        means=torch.stack([d.loc for d in rows]);scales=torch.stack([d.scale for d in rows])
        rowp=torch.stack([TeamLogProb(d,z) for d,z in zip(rows,batch['z'])])
        row_error=float((rowp-batch['logp']).abs().max())
        result.update(rowwise_checked=True,rowwise_error=row_error)
        equivalent=(torch.allclose(initial.loc,means,atol=1e-6,rtol=1e-5) and
                    torch.allclose(initial.scale,scales,atol=1e-7,rtol=1e-6))
        if not equivalent or error>10*tolerance or not np.isfinite(row_error) or row_error>tolerance:
            raise RuntimeError(f'on-policy log-prob mismatch: batch={error}, row={row_error}; policy drift or excessive numerical error')
    return initial,result


def update(model, optimizer, batch, config, teacher, gates):
    adv, returns = gae(batch["reward"], batch["value"], batch["next_value"],
                       batch["terminated"].unsqueeze(-1), batch["ended"].unsqueeze(-1),
                       config.gamma, config.gae_lambda)
    adv = (adv - adv.mean(0)) / adv.std(0, unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        initial, probability_check = check_on_policy(model,batch)
        ratio_error = probability_check['rowwise_error'] if probability_check['rowwise_checked'] else probability_check['batch_error']
        teacher_dist = teacher.distribution(batch["obs"], batch["links"]) if teacher is not None else None
    totals = []
    with torch.no_grad():
        without_messages = model.distribution(batch["obs"], batch["links"], intervention="zero")
        message_effect = (initial.loc - without_messages.loc).abs().mean().item()
    for _ in range(config.epochs):
        for idx in torch.randperm(len(adv)).split(config.minibatch):
            dist = model.distribution(batch["obs"][idx], batch["links"][idx])
            logp = TeamLogProb(dist, batch["z"][idx])
            ratio = (logp - batch["logp"][idx]).exp()
            policy = -torch.minimum(ratio * adv[idx],
                                    ratio.clamp(1 - config.clip, 1 + config.clip) * adv[idx]).mean()
            value_loss = (model.values(batch["state"][idx]) - returns[idx]).square().mean()
            mutual = torch.zeros(())
            if teacher_dist is not None:
                # Student || stopgrad(teacher), role matched on the SAME input rows.
                from torch.distributions import Normal
                peer = Normal(teacher_dist.loc[idx], teacher_dist.scale[idx])
                mutual = (kl_divergence(dist, peer).sum(-1) * gates[idx, None]).mean()
            entropy = dist.entropy().sum(-1).mean()  # pre-tanh Gaussian entropy convention
            loss = policy + 0.5 * value_loss - config.entropy_coef * entropy + config.mutual_beta * mutual
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(grad):
                raise FloatingPointError("nonfinite PPO gradient")
            optimizer.step()
            log_ratio = logp - batch["logp"][idx]
            totals.append([policy.item(), value_loss.item(), mutual.item(), entropy.item(),
                           ((ratio - 1) - log_ratio).mean().item(),
                           ((ratio - 1).abs() > config.clip).float().mean().item(), float(grad), loss.item()])
    means = np.mean(totals, axis=0)
    variance = returns.var(unbiased=False)
    ev = float(1 - (returns - batch["value"]).var(unbiased=False) / variance) if variance > 1e-8 else 0.0
    return dict(zip(("policy_loss", "value_loss", "mutual_kl", "entropy", "approx_kl", "clip_fraction",
                     "gradient_norm", "total_loss"), map(float, means)),
                ratio_error=ratio_error, gate_fraction=float(gates.mean()), explained_variance=ev,
                batched_ratio_error=probability_check['batch_error'],
                on_policy_rowwise_check=probability_check['rowwise_checked'],
                message_mean_effect=message_effect, link_fraction=float(batch["links"].mean()),
                action_saturation=float((batch["z"].tanh().abs() > 0.95).float().mean()),
                policy_std=float(model.action_std().detach().mean()),
                acceleration_std=float(model.action_std().detach()[:,0].mean()),
                steering_std=float(model.action_std().detach()[:,1].mean()),
                learning_rate=optimizer.param_groups[0]["lr"])


def TeamLogProb(dist, z):
    from ..models.team import Team
    return Team.log_prob(dist, z)
