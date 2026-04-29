"""
脱离rilib的MAPPO训练，直接使用carla_multi_agent_env和mappo_visualizer进行训练和评估
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from visualization.mappo_visualizer import MAPPOVisualizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Actor(nn.Module):
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


def collect_positions(env: CarlaMultiAgentEnv) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    vehicle_positions: Dict[int, Tuple[float, float]] = {}
    target_positions: Dict[int, Tuple[float, float]] = {}

    for aid in range(env.num_agents):
        if aid < len(env.vehicles) and env.vehicles[aid].is_alive:
            trans = env.vehicles[aid].get_transform()
            vehicle_positions[aid] = (trans.location.x, trans.location.y)
        if aid < len(env.target_waypoints):
            wp = env.target_waypoints[aid]
            target_positions[aid] = (wp.transform.location.x, wp.transform.location.y)

    return vehicle_positions, target_positions


def evaluate(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    seed = args.seed if args.seed is not None else int(ckpt_args.get("seed", 42))
    set_seed(seed)

    num_agents = args.num_agents if args.num_agents is not None else int(ckpt_args.get("num_agents", 2))
    max_steps = args.max_steps if args.max_steps is not None else int(ckpt_args.get("max_steps", 1000))

    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=num_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=max_steps,
    )

    actors = {
        aid: Actor(obs_dim=19, act_dim=3, hidden=int(ckpt_args.get("hidden_dim", 128))).to(device)
        for aid in range(num_agents)
    }

    actors_state = ckpt.get("actors", {})
    for aid, actor in actors.items():
        if aid not in actors_state:
            raise KeyError(f"Checkpoint 中缺少 actor {aid} 的权重")
        actor.load_state_dict(actors_state[aid])
        actor.eval()

    visualizer = MAPPOVisualizer(width=1200, height=800, fps=30) if args.render else None

    try:
        for ep in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            episode_rewards = {aid: 0.0 for aid in range(num_agents)}
            step = 0

            done = False
            while not done:
                step += 1
                action_dict: Dict[int, np.ndarray] = {}

                with torch.no_grad():
                    for aid, obs in obs_dict.items():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        mu = actors[aid](obs_t).squeeze(0)
                        action_tanh = torch.tanh(mu).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                next_obs, rewards, dones, _ = env.step(action_dict)

                for aid, r in rewards.items():
                    episode_rewards[aid] += float(r)

                if visualizer is not None:
                    visualizer.handle_events()
                    if not visualizer.running:
                        raise KeyboardInterrupt
                    vehicle_positions, target_positions = collect_positions(env)
                    camera_images = env.get_camera_images()
                    visualizer.render(
                        observations=obs_dict,
                        rewards=rewards,
                        episode_rewards=episode_rewards,
                        vehicle_positions=vehicle_positions,
                        target_positions=target_positions,
                        episode=ep,
                        step=step,
                        max_episodes=args.episodes,
                        max_steps=max_steps,
                        camera_images=camera_images,
                    )

                obs_dict = next_obs
                done = all(dones.get(aid, False) for aid in range(num_agents)) or step >= max_steps

            avg_return = float(np.mean([episode_rewards[aid] for aid in range(num_agents)]))
            detail = ", ".join([f"agent{aid}={episode_rewards[aid]:.2f}" for aid in range(num_agents)])
            print(f"[Episode {ep}/{args.episodes}] avg_return={avg_return:.2f}, {detail}")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate MAPPO checkpoint (non-RLlib) with visualization")

    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估 episode 数")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=None, help="可选：覆盖 checkpoint 中 max_steps")
    parser.add_argument("--num-agents", type=int, default=None, help="可选：覆盖 checkpoint 中 num_agents")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="禁用可视化")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.render = not args.no_render
    evaluate(args)
