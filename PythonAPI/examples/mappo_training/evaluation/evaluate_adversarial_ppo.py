"""
评估对抗智能体（支持单对抗和多对抗checkpoint格式）
- 合作团队: 加载预训练MAPPO策略（冻结）
- 对抗团队: 加载对抗PPO/MAPPO策略
- 兼容旧格式 (adversarial_actor) 和新格式 (adversarial_actors dict)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # 加载对抗checkpoint（兼容新旧格式）
    adv_ckpt = torch.load(args.adversarial_ckpt, map_location=device)
    adv_ckpt_args = adv_ckpt.get("args", {})

    # 检测格式: 新格式有 "adversarial_actors" dict，旧格式有 "adversarial_actor" 单个
    if "adversarial_actors" in adv_ckpt:
        # 新格式: 多对抗智能体
        adversarial_actors_state = adv_ckpt["adversarial_actors"]
        cooperative_ids = adv_ckpt.get("cooperative_ids", [0, 1])
        adversarial_ids = adv_ckpt.get("adversarial_ids", list(adversarial_actors_state.keys()))
    elif "adversarial_actor" in adv_ckpt:
        # 旧格式: 单对抗智能体
        adversarial_actors_state = {2: adv_ckpt["adversarial_actor"]}
        cooperative_ids = [0, 1]
        adversarial_ids = [2]
    else:
        raise KeyError("Checkpoint中未找到'adversarial_actor'或'adversarial_actors'")

    num_cooperative = len(cooperative_ids)
    num_adversarial = len(adversarial_ids)
    total_agents = num_cooperative + num_adversarial

    print(f"检测到 {num_adversarial} 个对抗智能体: {adversarial_ids}")

    # 设置seed
    seed = args.seed if args.seed is not None else 42
    set_seed(seed)

    # 创建环境
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=total_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
        adversarial_agent_ids=adversarial_ids,
    )

    # 加载合作团队 Actor
    pretrained_actors: Dict[int, Actor] = {}
    for aid in cooperative_ids:
        if aid not in pretrained_actors_state:
            raise KeyError(f"预训练checkpoint中缺少agent {aid}的权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        pretrained_actors[aid] = actor

    # 加载对抗团队 Actor
    adversarial_actors: Dict[int, Actor] = {}
    for aid in adversarial_ids:
        if aid not in adversarial_actors_state:
            raise KeyError(f"对抗checkpoint中缺少agent {aid}的权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(adversarial_actors_state[aid])
        actor.eval()
        adversarial_actors[aid] = actor

    print(f"评估开始...")
    print(f"合作团队 (冻结): {list(pretrained_actors.keys())}")
    print(f"对抗团队: {list(adversarial_actors.keys())}")

    try:
        all_coop_returns: List[float] = []
        all_adv_returns: Dict[int, List[float]] = {aid: [] for aid in adversarial_ids}

        for ep in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            episode_rewards = {aid: 0.0 for aid in range(total_agents)}
            step = 0
            done = False

            while not done:
                step += 1
                action_dict: Dict[int, np.ndarray] = {}

                with torch.no_grad():
                    # 合作团队
                    for aid in cooperative_ids:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        action_tanh = pretrained_actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                    # 对抗团队
                    for aid in adversarial_ids:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        action_tanh = adversarial_actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                next_obs, rewards, dones, _ = env.step(action_dict)

                for aid, r in rewards.items():
                    episode_rewards[aid] += float(r)

                obs_dict = next_obs
                done = all(dones.get(aid, False) for aid in range(total_agents)) or step >= args.max_steps

            # 统计
            avg_coop = float(np.mean([episode_rewards[cid] for cid in cooperative_ids]))
            all_coop_returns.append(avg_coop)

            adv_detail = ", ".join([f"A{aid}={episode_rewards[aid]:.2f}" for aid in adversarial_ids])
            print(
                f"[Episode {ep}/{args.episodes}] "
                f"coop_avg={avg_coop:.2f}, "
                f"adv=({adv_detail})"
            )

            for aid in adversarial_ids:
                all_adv_returns[aid].append(episode_rewards[aid])

        # 最终统计
        print("\n" + "=" * 60)
        print("评估完成！")
        print(f"合作团队平均回报: {np.mean(all_coop_returns):.2f} (±{np.std(all_coop_returns):.2f})")
        for aid in adversarial_ids:
            adv_arr = all_adv_returns[aid]
            print(f"对抗智能体 {aid} 平均回报: {np.mean(adv_arr):.2f} (±{np.std(adv_arr):.2f})")

        avg_all_adv = float(np.mean([np.mean(all_adv_returns[aid]) for aid in adversarial_ids]))
        print(f"对抗团队整体平均回报: {avg_all_adv:.2f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate adversarial agents against frozen pretrained agents")

    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="预训练MAPPO checkpoint路径")
    parser.add_argument("--adversarial-ckpt", type=str, required=True,
                        help="对抗PPO/MAPPO checkpoint路径")
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
