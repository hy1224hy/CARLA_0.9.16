import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from carla_multi_agent_env import CarlaMultiAgentEnv


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SharedActorCritic(nn.Module):
    """Shared policy + value network for all agents."""

    def __init__(self, obs_dim: int = 19, act_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std, value

    def sample_action(self, obs: torch.Tensor):
        mu, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()  # reparameterized sample
        action_tanh = torch.tanh(z)

        # tanh correction for log-prob
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)

        return action_tanh, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, action_tanh: torch.Tensor):
        mu, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # inverse tanh for density computation
        clipped = torch.clamp(action_tanh, -0.999, 0.999)
        z = 0.5 * torch.log((1 + clipped) / (1 - clipped))

        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - clipped.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


@dataclass
class Transition:
    obs: np.ndarray
    action_tanh: np.ndarray
    log_prob: float
    reward: float
    done: float
    value: float


class RolloutBuffer:
    def __init__(self):
        self.data: List[Transition] = []

    def add(self, transition: Transition):
        self.data.append(transition)

    def clear(self):
        self.data.clear()


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """
    Convert tanh action in [-1, 1] to env action format:
      throttle in [0,1], steer in [-1,1], brake in [0,1]
    """
    throttle = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake = (action_tanh[2] + 1.0) * 0.5
    return np.array([throttle, steer, brake], dtype=np.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=2,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
    )

    model = SharedActorCritic(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for update in range(1, args.updates + 1):
        buffer = RolloutBuffer()

        obs_dict, _ = env.reset()
        ep_returns = {0: 0.0, 1: 0.0}

        # Collect rollout
        for _ in range(args.rollout_steps):
            action_dict: Dict[int, np.ndarray] = {}
            step_cache = {}

            for aid, obs in obs_dict.items():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_tanh_t, logp_t, value_t = model.sample_action(obs_t)

                action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
                action_env = action_tanh_to_env(action_tanh)

                action_dict[aid] = action_env
                step_cache[aid] = {
                    "obs": obs,
                    "action_tanh": action_tanh,
                    "logp": float(logp_t.item()),
                    "value": float(value_t.item()),
                }

            next_obs, rewards, dones, info = env.step(action_dict)

            # store transitions agent-wise (shared policy)
            for aid in step_cache.keys():
                done = float(dones.get(aid, False))
                reward = float(rewards.get(aid, 0.0))
                ep_returns[aid] += reward
                buffer.add(
                    Transition(
                        obs=step_cache[aid]["obs"],
                        action_tanh=step_cache[aid]["action_tanh"],
                        log_prob=step_cache[aid]["logp"],
                        reward=reward,
                        done=done,
                        value=step_cache[aid]["value"],
                    )
                )

            obs_dict = next_obs
            if all(dones.values()):
                obs_dict, _ = env.reset()

        # Prepare tensors
        obs_arr = np.array([t.obs for t in buffer.data], dtype=np.float32)
        act_arr = np.array([t.action_tanh for t in buffer.data], dtype=np.float32)
        old_logp_arr = np.array([t.log_prob for t in buffer.data], dtype=np.float32)
        reward_arr = np.array([t.reward for t in buffer.data], dtype=np.float32)
        done_arr = np.array([t.done for t in buffer.data], dtype=np.float32)
        value_arr = np.array([t.value for t in buffer.data], dtype=np.float32)

        adv_arr, ret_arr = compute_gae(
            rewards=reward_arr,
            values=value_arr,
            dones=done_arr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t = torch.as_tensor(obs_arr, device=device)
        act_t = torch.as_tensor(act_arr, device=device)
        old_logp_t = torch.as_tensor(old_logp_arr, device=device)
        adv_t = torch.as_tensor(adv_arr, device=device)
        ret_t = torch.as_tensor(ret_arr, device=device)

        # PPO update
        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(args.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch_size):
                mb = idx[start : start + args.minibatch_size]

                new_logp, entropy, value_pred = model.evaluate_actions(obs_t[mb], act_t[mb])

                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (ret_t[mb] - value_pred).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % args.log_interval == 0:
            avg_ret = (ep_returns[0] + ep_returns[1]) / 2.0
            print(
                f"[Update {update}/{args.updates}] "
                f"avg_ep_return(last rollout)={avg_ret:.2f}, "
                f"agent0={ep_returns[0]:.2f}, agent1={ep_returns[1]:.2f}"
            )

        if update % args.save_interval == 0:
            ckpt = os.path.join(save_dir, f"minimal_ppo_update_{update}.pt")
            torch.save(
                {
                    "update": update,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"Checkpoint saved: {ckpt}")

    env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal PPO training without RLlib")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--save-dir", type=str, default="./logs/minimal_ppo_ckpts")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
