"""Soft Actor-Critic (fixed entropy temperature) twin training with optional mutual KL.

Same protocol as the PPO twin: A/B replicas, independent streams, frozen teacher
snapshots, same-input pre-tanh Gaussian KL, beta warmup/ramp, A+B budget,
identical evaluation/selection. Off-policy replay buffers are saved for
bit-exact resume.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter
from .model import ActorCritic
from .learning import beta_at
from ..training.checkpoint import source_digest, source_files, runtime_versions, save
from ..training.locking import run_lock
from .runner import PROTOCOL, load_state, configuration, append, evaluate_model, rank_report, model_from_state


class SACActorCritic(ActorCritic):
    """Policy reuses the gaussian tanh-squashed ActorCritic; action-conditioned twin Q networks."""

    def __init__(self, hidden=128):
        super().__init__(hidden)
        def qnet():
            return nn.Sequential(nn.Linear(24, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.q1 = qnet()
        self.q2 = qnet()

    def q_values(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.cursor = 0
        self.length = 0
        self.obs = np.zeros((capacity, 22), dtype=np.float32)
        self.act = np.zeros((capacity, 2), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.nxt = np.zeros((capacity, 22), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.length

    def push(self, obs, act, reward, nxt, done):
        i = self.cursor
        self.obs[i] = obs
        self.act[i] = act
        self.reward[i] = reward
        self.nxt[i] = nxt
        self.done[i] = done
        self.cursor = (i + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.length, batch)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.act[idx]),
                torch.as_tensor(self.reward[idx]), torch.as_tensor(self.nxt[idx]),
                torch.as_tensor(self.done[idx]))

    def state(self):
        return dict(capacity=self.capacity, cursor=self.cursor, length=self.length,
                    obs=self.obs.copy(), act=self.act.copy(), reward=self.reward.copy(),
                    nxt=self.nxt.copy(), done=self.done.copy())

    @classmethod
    def from_state(cls, state):
        buffer = cls(state['capacity'])
        buffer.cursor = state['cursor']
        buffer.length = state['length']
        buffer.obs[:] = state['obs']
        buffer.act[:] = state['act']
        buffer.reward[:] = state['reward']
        buffer.nxt[:] = state['nxt']
        buffer.done[:] = state['done']
        return buffer


def sac_update(model, target, optimizer, buffer, batch, cfg, teacher, beta, rng):
    s, a, r, s2, done = buffer.sample(batch, rng)
    with torch.no_grad():
        dist2 = model.distribution(s2)
        z2 = dist2.rsample()
        a2 = z2.tanh()
        lp2 = model.log_prob(dist2, z2)
        q1t, q2t = target.q_values(s2, a2)
        y = r + cfg.gamma * (1 - done) * (torch.min(q1t, q2t) - cfg.alpha * lp2)
    q1, q2 = model.q_values(s, a)
    q_loss = ((q1 - y).square() + (q2 - y).square()).mean()
    dist = model.distribution(s)
    z = dist.rsample()
    a_new = z.tanh()
    lp = model.log_prob(dist, z)
    q1p, q2p = model.q_values(s, a_new)
    policy_loss = (cfg.alpha * lp - torch.min(q1p, q2p)).mean()
    mutual = torch.zeros(())
    if teacher is not None and beta > 0:
        with torch.no_grad():
            peer = teacher.distribution(s)
        mutual = kl_divergence(dist, peer).sum(-1).mean()
    loss = q_loss + policy_loss + beta * mutual
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite SAC loss')
    optimizer.zero_grad()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
    if not torch.isfinite(grad):
        raise FloatingPointError('Nonfinite SAC gradient')
    optimizer.step()
    with torch.no_grad():
        for p, tp in zip(model.parameters(), target.parameters()):
            tp.mul_(1 - cfg.tau).add_(p, alpha=cfg.tau)
        entropy = model.distribution(s).entropy().sum(-1).mean()
    return dict(q_loss=float(q_loss.detach()), policy_loss=float(policy_loss.detach()),
                entropy=float(entropy.detach()), mutual_kl=float(mutual.detach()),
                mutual_loss=float(beta * mutual.detach()), gradient_norm=float(grad),
                effective_beta=beta,
                acceleration_std=float(model.log_std[0].clamp(-3, 0).exp().detach()),
                steering_std=float(model.log_std[1].clamp(-3, 0).exp().detach()))


def train_sac(config, output, initialize_from=None, resume=None, max_rounds=None):
    from .environment import StraightEnv
    cfg = config.validate()
    output = Path(output).resolve()
    if max_rounds is not None and max_rounds < 1:
        raise ValueError('max_rounds must be positive')
    if resume and initialize_from:
        raise ValueError('Choose resume or initialization')
    torch.set_num_threads(cfg.threads)
    models = {}
    optimizers = {}
    targets = {}
    buffers = {}
    rngs = {}
    torch_states = {}
    for j, name in enumerate(('A', 'B')):
        torch.manual_seed(cfg.seed + 10007 * j)
        models[name] = SACActorCritic(cfg.hidden)
        optimizers[name] = torch.optim.Adam(models[name].parameters(), lr=cfg.lr, eps=1e-5)
        targets[name] = copy.deepcopy(models[name]).eval().requires_grad_(False)
        buffers[name] = ReplayBuffer()
        rngs[name] = np.random.default_rng(cfg.seed + 10007 * j)
        torch_states[name] = torch.get_rng_state()
    teachers = {k: copy.deepcopy(v).eval().requires_grad_(False) for k, v in models.items()}
    steps = round_id = validation_steps = 0
    best = (-1, -float('inf'), 0.)
    selection = None
    best_model = None
    identity = source_digest()
    versions = runtime_versions()
    if resume:
        state = load_state(resume)
        if Path(resume).resolve() != output / 'latest.pt':
            raise ValueError('Resume this run latest.pt only')
        if state['config'] != cfg.to_dict() or state['source_sha256'] != identity or state['packages'] != versions:
            raise ValueError('Resume config/source/runtime mismatch')
        for k in models:
            models[k].load_state_dict(state['models'][k])
            optimizers[k].load_state_dict(state['optimizers'][k])
            targets[k].load_state_dict(state['targets'][k])
            buffers[k] = ReplayBuffer.from_state(state['buffers'][k])
            teachers[k].load_state_dict(state['teachers'][k])
            rngs[k].bit_generator.state = state['numpy_rng'][k]
            torch_states[k] = state['torch_rng'][k]
        steps = state['train_steps']
        round_id = state['round']
        validation_steps = state['validation_steps']
        best = tuple(state['best_rank'])
        selection = state['selection']
        best_model = state['best_model']
        if best_model is not None:
            save(output / 'selected.pt', dict(protocol=PROTOCOL, learner='sac', config=cfg.to_dict(),
                source_sha256=identity, packages=versions, model=best_model, selection=selection))
        for filename in ('metrics.jsonl', 'episodes.jsonl', 'validation.jsonl'):
            path = output / filename
            if path.exists():
                committed = []
                lines = path.read_text(encoding='utf-8').splitlines()
                for index, line in enumerate(lines):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if index != len(lines) - 1:
                            raise
                        break
                    if row['round'] <= round_id:
                        committed.append(row)
                path.write_text(''.join(json.dumps(r) + '\n' for r in committed), encoding='utf-8')
    else:
        output.mkdir(parents=True, exist_ok=False)
        if initialize_from:
            initial = load_state(initialize_from)
            compatible = ((initial['config']['level'] == 0 and initial.get('attack_level') is None)
                          or initial.get('attack_level') == cfg.level)
            if not compatible or initial['config']['hidden'] != cfg.hidden:
                raise ValueError('Initialization must be a compatible normal driver or matching-level attack driver')
            if not initial.get('qualified') or initial['source_sha256'] != identity or initial['packages'] != versions:
                raise ValueError('Driver qualification/source/runtime mismatch')
            for key in ('duration', 'policy_frequency', 'simulation_frequency', 'target_speed', 'target_time_gap'):
                if initial['config'][key] != cfg.to_dict()[key]:
                    raise ValueError('Driver physical contract mismatch: ' + key)
            for model in models.values():
                model.actor.load_state_dict(initial['actor'])
            targets = {k: copy.deepcopy(v).eval().requires_grad_(False) for k, v in models.items()}
            teachers = {k: copy.deepcopy(v).eval().requires_grad_(False) for k, v in models.items()}
        (output / 'config.json').write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
        (output / 'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL, learner='sac',
            source_sha256=identity, packages=versions,
            initialization={'path': str(initialize_from),
                            'sha256': __import__('hashlib').sha256(Path(initialize_from).read_bytes()).hexdigest()} if initialize_from else None,
            budget='A+B environment decisions'), indent=2), encoding='utf-8')
        source = output / 'source'
        source.mkdir()
        for file in source_files():
            snapshot = source / file.relative_to(Path(__file__).resolve().parents[1])
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(file.read_bytes())
    writer = SummaryWriter(str(output / 'tensorboard'), purge_step=0)
    writer.add_text('configuration', json.dumps(cfg.to_dict(), indent=2), 0)

    def log(row, prefix):
        allowed = ({'q_loss', 'policy_loss', 'entropy', 'mutual_kl', 'mutual_loss', 'gradient_norm',
                    'effective_beta', 'acceleration_std', 'steering_std', 'mean_step_reward', 'decisions_per_second'}
                   if prefix.startswith('train/') else
                   {'success_rate', 'attack_success_rate', 'collision_rate', 'invalid_rate',
                    'grade_violation_rate', 'baseline_valid_rate', 'utility', 'red_delay_m', 'return',
                    'blocking_time', 'cut_in_rate', 'stop_qualified_rate'})
        for key, value in row.items():
            if key in allowed and isinstance(value, (int, float, bool)) and np.isfinite(value):
                writer.add_scalar(prefix + '/' + key, value, row['train_steps'])

    envs = {k: StraightEnv(cfg) for k in models}
    start_round = round_id
    started_at = time.perf_counter()
    inherited_wall = state.get('wall_seconds', 0.) if resume else 0.
    try:
        while steps < cfg.total_steps and (max_rounds is None or round_id - start_round < max_rounds):
            n = min(cfg.rollout_steps, (cfg.total_steps - steps) // 2)
            clock = steps
            round_id += 1
            beta = beta_at(clock, cfg.beta, cfg.warmup_steps, cfg.ramp_steps) if cfg.method == 'mutual' else 0.
            for name in models:
                torch.set_rng_state(torch_states[name])
                started = time.perf_counter()
                obs, _ = envs[name].reset(seed=int(rngs[name].integers(900000)))
                episodes = []
                total = 0.
                components = {}
                for _ in range(n):
                    o = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        a, z, _ = models[name].sample(o)
                    nxt, r, term, trunc, info = envs[name].step(a.numpy())
                    for key, value in info.get('reward_components', {}).items():
                        components[key] = components.get(key, 0.) + value
                    buffers[name].push(obs, a.numpy(), r, nxt, float(term or trunc))
                    total += r
                    obs = nxt
                    if term or trunc:
                        episodes.append(dict(info, **{'return': total, 'reward_components': components}))
                        total = 0.
                        components = {}
                        obs, _ = envs[name].reset(seed=int(rngs[name].integers(900000)))
                stats = [sac_update(models[name], targets[name], optimizers[name], buffers[name],
                                    cfg.minibatch, cfg, teachers['B' if name == 'A' else 'A'], beta, rngs[name])
                         for _ in range(n)]
                means = {key: float(np.mean([s[key] for s in stats])) for key in stats[0]}
                torch_states[name] = torch.get_rng_state()
                steps += n
                row = dict(means, team=name, round=round_id, train_steps=clock + 2 * n,
                           mean_step_reward=float(np.mean([e['return'] for e in episodes])) if episodes else 0.,
                           decisions_per_second=n / (time.perf_counter() - started))
                append(output / 'metrics.jsonl', row)
                log(row, 'train/' + name)
                for episode in episodes:
                    append(output / 'episodes.jsonl', dict(episode, team=name, round=round_id, train_steps=clock + 2 * n))
            if round_id % cfg.refresh_rounds == 0:
                teachers = {k: copy.deepcopy(v).eval().requires_grad_(False) for k, v in models.items()}
            if steps == cfg.total_steps or steps // cfg.eval_interval != clock // cfg.eval_interval or selection is None:
                reports = {k: evaluate_model(v, cfg, 'selection', cfg.eval_cases) for k, v in models.items()}
                validation_steps += sum(e['episode_steps'] for report in reports.values() for e in report['episodes'])
                append(output / 'validation.jsonl', dict(round=round_id, train_steps=steps,
                    validation_steps=validation_steps, reports=reports))
                for k, report in reports.items():
                    log(dict(report['overall'], train_steps=steps), 'validation/' + k)
                winner = max(reports, key=lambda k: rank_report(reports[k]['overall'], cfg.level))
                rank = rank_report(reports[winner]['overall'], cfg.level)
                if rank > best:
                    best = rank
                    selection = dict(team=winner, train_steps=steps, eligible=bool(rank[0]), overall=reports[winner]['overall'])
                    best_model = copy.deepcopy(models[winner].state_dict())
                    save(output / 'selected.pt', dict(protocol=PROTOCOL, learner='sac', config=cfg.to_dict(),
                        source_sha256=identity, packages=versions, model=best_model, selection=selection))
            writer.add_scalar('budget/validation_steps', validation_steps, steps)
            writer.flush()
            save(output / 'latest.pt', dict(protocol=PROTOCOL, learner='sac', config=cfg.to_dict(),
                source_sha256=identity, packages=versions,
                models={k: v.state_dict() for k, v in models.items()},
                optimizers={k: v.state_dict() for k, v in optimizers.items()},
                targets={k: v.state_dict() for k, v in targets.items()},
                buffers={k: v.state() for k, v in buffers.items()},
                teachers={k: v.state_dict() for k, v in teachers.items()},
                numpy_rng={k: v.bit_generator.state for k, v in rngs.items()},
                torch_rng=torch_states, train_steps=steps, round=round_id,
                validation_steps=validation_steps, best_rank=best, selection=selection,
                best_model=best_model, wall_seconds=inherited_wall + (time.perf_counter() - started_at)))
            print(f'level={cfg.level} {cfg.method} SAC A+B={steps}/{cfg.total_steps} beta={beta:.5f}', flush=True)
    finally:
        writer.close()
        for env in envs.values():
            env.close()
    wall_seconds = inherited_wall + (time.perf_counter() - started_at)
    summary = dict(status='complete' if steps == cfg.total_steps else 'paused', train_steps=steps,
                   validation_steps=validation_steps, selection=selection, wall_seconds=wall_seconds)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
