"""
评估对抗智能体
- Agent 0, 1: 加载预训练MAPPO策略（冻结）
- Agent 2: 加载对抗PPO策略
- 用于评估对抗效果
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.net(obs)
        mu = self.mu_head(h)
        return mu


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake]."""
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5

    if throttle_raw >= brake_raw:
        throttle = throttle_raw
        brake = 0.0
    else:
        throttle = 0.0
        brake = brake_raw

    return np.array([throttle, steer, brake], dtype=np.float32)


def evaluate(args):
    if not os.path.exists(args.adversarial_ckpt):
        raise FileNotFoundError(f"对抗checkpoint不存在: {args.adversarial_ckpt}")

    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(f"预训练checkpoint不存在: {args.pretrained_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 加载预训练checkpoint
    pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    pretrained_actors_state = pretrained_ckpt.get("actors", {})

    # 加载对抗checkpoint
    adv_ckpt = torch.load(args.adversarial_ckpt, map_location=device)
    adv_actor_state = adv_ckpt.get("adversarial_actor", {})

    # 设置seed
    seed = args.seed if args.seed is not None else 42
    set_seed(seed)

    # 创建环境
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=3,  # 2个预训练 + 1个对抗
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
    )

    # 加载预训练智能体
    pretrained_actors: Dict[int, Actor] = {}
    for aid in [0, 1]:
        if aid not in pretrained_actors_state:
            raise KeyError(f"预训练checkpoint中缺少agent {aid}的权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        pretrained_actors[aid] = actor

    # 加载对抗智能体
    adversarial_id = 2
    adv_actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
    adv_actor.load_state_dict(adv_actor_state)
    adv_actor.eval()

    print(f"评估开始...")
    print(f"预训练智能体: {list(pretrained_actors.keys())}")
    print(f"对抗智能体: {adversarial_id}")

    try:
        all_pretrained_returns = []
        all_adversarial_returns = []

        for ep in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
            step = 0
            done = False

            while not done:
                step += 1
                action_dict: Dict[int, np.ndarray] = {}

                with torch.no_grad():
                    # 预训练智能体
                    for aid in [0, 1]:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        mu = pretrained_actors[aid](obs_t)
                        action_tanh = torch.tanh(mu).squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                    # 对抗智能体
                    if adversarial_id in obs_dict:
                        obs_t = torch.as_tensor(obs_dict[adversarial_id], dtype=torch.float32, device=device).unsqueeze(0)
                        mu = adv_actor(obs_t)
                        action_tanh = torch.tanh(mu).squeeze(0).cpu().numpy()
                        action_dict[adversarial_id] = action_tanh_to_env(action_tanh)

                # 执行动作
                next_obs, rewards, dones, _ = env.step(action_dict)

                # 累积奖励
                for aid, r in rewards.items():
                    episode_rewards[aid] += float(r)

                obs_dict = next_obs
                done = all(dones.get(aid, False) for aid in range(3)) or step >= args.max_steps

            # 计算统计
            avg_pretrained = (episode_rewards[0] + episode_rewards[1]) / 2.0
            adversarial_reward = episode_rewards[2]

            all_pretrained_returns.append(avg_pretrained)
            all_adversarial_returns.append(adversarial_reward)

            print(
                f"[Episode {ep}/{args.episodes}] "
                f"pretrained_avg={avg_pretrained:.2f}, "
                f"adversarial={adversarial_reward:.2f}, "
                f"agent0={episode_rewards[0]:.2f}, agent1={episode_rewards[1]:.2f}"
            )

        # 最终统计
        print("\n" + "=" * 60)
        print("评估完成！")
        print(f"预训练智能体平均回报: {np.mean(all_pretrained_returns):.2f} (±{np.std(all_pretrained_returns):.2f})")
        print(f"对抗智能体平均回报: {np.mean(all_adversarial_returns):.2f} (±{np.std(all_adversarial_returns):.2f})")
        print(f"零和验证: {np.mean(all_pretrained_returns):.2f} ≈ {-np.mean(all_adversarial_returns):.2f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate adversarial PPO agent against frozen pretrained agents")

    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="预训练MAPPO checkpoint路径")
    parser.add_argument("--adversarial-ckpt", type=str, required=True,
                        help="对抗PPO checkpoint路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估episode数")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)
