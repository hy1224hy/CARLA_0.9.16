"""
对抗训练脚本
- Agent 0, 1: 加载预训练MAPPO策略（冻结）
- Agent 2: 对抗智能体，使用PPO训练
- 对抗智能体奖励 = -(Agent 0 + Agent 1)的平均奖励
"""

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


class Actor(nn.Module):
    """Actor network for policy."""

    def __init__(self, obs_dim: int = 19, act_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
        return action_tanh, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action_tanh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        clipped = torch.clamp(action_tanh, -0.999, 0.999)
        z = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - clipped.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Critic network for value estimation."""

    def __init__(self, obs_dim: int = 19, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


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
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake]."""
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5

    # 对抗智能体可能需要不同的动作策略
    # 这里使用互斥控制，与原训练一致
    if throttle_raw >= brake_raw:
        throttle = throttle_raw
        brake = 0.0
    else:
        throttle = 0.0
        brake = brake_raw

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

    # 创建环境，3个智能体（2个预训练 + 1个对抗）
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=3,  # 2个预训练 + 1个对抗
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
    )

    # 加载预训练checkpoint
    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(f"预训练checkpoint不存在: {args.pretrained_ckpt}")

    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    pretrained_actors_state = ckpt.get("actors", {})

    # 创建冻结的预训练智能体
    pretrained_actors: Dict[int, Actor] = {}
    for aid in [0, 1]:  # Agent 0, 1
        if aid not in pretrained_actors_state:
            raise KeyError(f"Checkpoint中缺少agent {aid}的权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        # 冻结参数，不进行梯度更新
        for param in actor.parameters():
            param.requires_grad = False
        pretrained_actors[aid] = actor

    print(f"已加载预训练智能体: {list(pretrained_actors.keys())}")

    # 创建对抗智能体（Agent 2）的Actor-Critic网络
    adversarial_id = 2
    adv_actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
    adv_critic = Critic(obs_dim=19, hidden=args.hidden_dim).to(device)

    adv_actor_optim = optim.Adam(adv_actor.parameters(), lr=args.lr_actor)
    adv_critic_optim = optim.Adam(adv_critic.parameters(), lr=args.lr_critic)

    # 保存目录
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"对抗训练开始...")
    print(f"对抗智能体ID: {adversarial_id}")
    print(f"预训练智能体(冻结): {list(pretrained_actors.keys())}")

    try:
        episode_idx = 1

        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer()
            obs_dict, _ = env.reset()
            episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
            episode_step = 0
            completed_episode_returns = []

            # 收集rollout
            for _ in range(args.rollout_steps):
                episode_step += 1
                action_dict: Dict[int, np.ndarray] = {}
                step_cache = {}

                with torch.no_grad():
                    # 预训练智能体（冻结策略）
                    for aid in [0, 1]:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        mu, _ = pretrained_actors[aid](obs_t)
                        action_tanh = torch.tanh(mu).squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                    # 对抗智能体（训练策略）
                    if adversarial_id in obs_dict:
                        obs_t = torch.as_tensor(obs_dict[adversarial_id], dtype=torch.float32, device=device).unsqueeze(0)
                        action_tanh_t, logp_t = adv_actor.sample_action(obs_t)
                        value_t = adv_critic(obs_t)

                        action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
                        action_dict[adversarial_id] = action_tanh_to_env(action_tanh)

                        step_cache[adversarial_id] = {
                            "obs": obs_dict[adversarial_id],
                            "action_tanh": action_tanh,
                            "logp": float(logp_t.item()),
                            "value": float(value_t.item()),
                        }

                # 执行动作
                next_obs, rewards, dones, _ = env.step(action_dict)

                # 计算对抗智能体的奖励：负的平均奖励（零和博弈）
                if adversarial_id in obs_dict:
                    # 前两个智能体的平均奖励
                    avg_pretrained_reward = (rewards.get(0, 0.0) + rewards.get(1, 0.0)) / 2.0
                    # 对抗智能体的奖励为负值
                    adversarial_reward = -avg_pretrained_reward

                    # 存储transition
                    buffer.add(
                        Transition(
                            obs=step_cache[adversarial_id]["obs"],
                            action_tanh=step_cache[adversarial_id]["action_tanh"],
                            log_prob=step_cache[adversarial_id]["logp"],
                            reward=adversarial_reward,
                            done=float(dones.get(adversarial_id, False)),
                            value=step_cache[adversarial_id]["value"],
                        )
                    )

                    episode_rewards[adversarial_id] += adversarial_reward

                # 累积其他智能体的奖励（仅用于监控）
                for aid in [0, 1]:
                    if aid in rewards:
                        episode_rewards[aid] += rewards[aid]

                obs_dict = next_obs

                # 检查episode是否结束
                episode_done = all(dones.get(aid, False) for aid in range(3))
                if episode_done:
                    completed_episode_returns.append(episode_rewards[adversarial_id])
                    obs_dict, _ = env.reset()
                    episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
                    episode_step = 0
                    episode_idx += 1

            # PPO更新对抗智能体
            if len(buffer.data) > 0:
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

                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
                act_t = torch.as_tensor(act_arr, dtype=np.float32, device=device)
                old_logp_t = torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device)
                adv_t = torch.as_tensor(adv_arr, dtype=torch.float32, device=device)
                ret_t = torch.as_tensor(ret_arr, dtype=torch.float32, device=device)

                n = obs_t.shape[0]
                idx = np.arange(n)

                # PPO epochs
                for _ in range(args.ppo_epochs):
                    np.random.shuffle(idx)
                    for start in range(0, n, args.minibatch_size):
                        mb = idx[start : start + args.minibatch_size]

                        # 更新Actor
                        new_logp, entropy = adv_actor.evaluate_actions(obs_t[mb], act_t[mb])
                        ratio = torch.exp(new_logp - old_logp_t[mb])
                        surr1 = ratio * adv_t[mb]
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t[mb]
                        actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()

                        adv_actor_optim.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(adv_actor.parameters(), args.max_grad_norm)
                        adv_actor_optim.step()

                        # 更新Critic
                        value_pred = adv_critic(obs_t[mb])
                        critic_loss = 0.5 * (ret_t[mb] - value_pred).pow(2).mean()

                        adv_critic_optim.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(adv_critic.parameters(), args.max_grad_norm)
                        adv_critic_optim.step()

            # 日志
            if update % args.log_interval == 0:
                if completed_episode_returns:
                    avg_ret = float(np.mean(completed_episode_returns))
                    ret_text = f"avg_completed_ep_return={avg_ret:.2f}"
                else:
                    ret_text = f"current_ep_return={episode_rewards[adversarial_id]:.2f}"

                print(
                    f"[Update {update}/{args.updates}] "
                    f"episodes_finished={len(completed_episode_returns)}, "
                    f"{ret_text}, "
                    f"pretrained_avg={ (episode_rewards[0] + episode_rewards[1]) / 2.0:.2f}"
                )

            # 保存checkpoint
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"adversarial_ppo_update_{update}.pt")
                torch.save(
                    {
                        "update": update,
                        "adversarial_actor": adv_actor.state_dict(),
                        "adversarial_critic": adv_critic.state_dict(),
                        "adversarial_actor_optim": adv_actor_optim.state_dict(),
                        "adversarial_critic_optim": adv_critic_optim.state_dict(),
                        "pretrained_ckpt": args.pretrained_ckpt,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Adversarial PPO training against frozen pretrained agents")

    # 环境参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 预训练模型路径
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="预训练MAPPO checkpoint路径")

    # 训练参数
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)

    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--save-dir", type=str, default="./logs/adversarial_ppo_ckpts")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
