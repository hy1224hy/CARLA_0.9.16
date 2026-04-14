"""
脱离rilib的MAPPO训练，直接使用carla_multi_agent_env和mappo_visualizer进行训练和评估
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
from mappo_visualizer import MAPPOVisualizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Actor(nn.Module):
    """Per-agent actor network (no policy sharing)."""

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


class CentralizedCritic(nn.Module):
    """Centralized critic consumes global state (CTDE in MAPPO)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    global_state: np.ndarray
    action_tanh: np.ndarray
    log_prob: float
    reward: float
    done: float
    value: float


class RolloutBuffer:
    def __init__(self, agent_ids: List[int]):
        self.data: Dict[int, List[Transition]] = {aid: [] for aid in agent_ids}

    def add(self, agent_id: int, transition: Transition):
        self.data[agent_id].append(transition)

    def get_agent_data(self, agent_id: int) -> List[Transition]:
        return self.data[agent_id]


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake].

    Uses mutually exclusive longitudinal control to avoid throttle/brake conflict.
    """
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

    agent_ids = list(range(env.num_agents))
    state_dim = env.get_global_state().shape[0]

    actors: Dict[int, Actor] = {
        aid: Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device) for aid in agent_ids
    }
    actor_optims: Dict[int, optim.Optimizer] = {
        aid: optim.Adam(actors[aid].parameters(), lr=args.lr_actor) for aid in agent_ids
    }

    critic = CentralizedCritic(state_dim=state_dim, hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)

    visualizer = MAPPOVisualizer(width=1200, height=800, fps=30) if args.render else None

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    try:
        episode_idx = 1

        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer(agent_ids)
            obs_dict, _ = env.reset()
            episode_rewards = {aid: 0.0 for aid in agent_ids}
            episode_step = 0
            completed_episode_returns: List[float] = []

            for _ in range(args.rollout_steps):
                episode_step += 1
                action_dict: Dict[int, np.ndarray] = {}
                step_cache: Dict[int, Dict[str, np.ndarray]] = {}

                global_state_np = env.get_global_state().astype(np.float32)
                global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    global_value = float(critic(global_state_t).item())

                for aid, obs in obs_dict.items():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action_tanh_t, logp_t = actors[aid].sample_action(obs_t)

                    action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
                    action_env = action_tanh_to_env(action_tanh)

                    action_dict[aid] = action_env
                    step_cache[aid] = {
                        "obs": obs,
                        "global_state": global_state_np,
                        "action_tanh": action_tanh,
                        "logp": float(logp_t.item()),
                        "value": global_value,
                    }

                next_obs, rewards, dones, _ = env.step(action_dict)

                for aid in step_cache.keys():
                    reward = float(rewards.get(aid, 0.0))
                    done = float(dones.get(aid, False))
                    episode_rewards[aid] += reward

                    buffer.add(
                        aid,
                        Transition(
                            obs=step_cache[aid]["obs"],
                            global_state=step_cache[aid]["global_state"],
                            action_tanh=step_cache[aid]["action_tanh"],
                            log_prob=step_cache[aid]["logp"],
                            reward=reward,
                            done=done,
                            value=step_cache[aid]["value"],
                        ),
                    )

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
                        episode=episode_idx,
                        step=episode_step,
                        max_episodes=0,
                        max_steps=args.max_steps,
                        camera_images=camera_images,
                    )

                obs_dict = next_obs

                # One environment episode should terminate immediately once env marks done.
                # episode_done = all(dones.get(aid, False) for aid in agent_ids)
                episode_done = all(dones.get(aid, False) for aid in agent_ids)
                if episode_done:
                    completed_episode_returns.append(float(np.mean([episode_rewards[aid] for aid in agent_ids])))
                    obs_dict, _ = env.reset()
                    episode_rewards = {aid: 0.0 for aid in agent_ids}
                    episode_step = 0
                    episode_idx += 1

            # Build per-agent training tensors + combined critic tensors.
            per_agent_tensors = {}
            critic_states = []
            critic_returns = []

            for aid in agent_ids:
                data = buffer.get_agent_data(aid)
                if len(data) == 0:
                    continue

                obs_arr = np.array([t.obs for t in data], dtype=np.float32)
                state_arr = np.array([t.global_state for t in data], dtype=np.float32)
                act_arr = np.array([t.action_tanh for t in data], dtype=np.float32)
                old_logp_arr = np.array([t.log_prob for t in data], dtype=np.float32)
                reward_arr = np.array([t.reward for t in data], dtype=np.float32)
                done_arr = np.array([t.done for t in data], dtype=np.float32)
                value_arr = np.array([t.value for t in data], dtype=np.float32)

                adv_arr, ret_arr = compute_gae(
                    rewards=reward_arr,
                    values=value_arr,
                    dones=done_arr,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

                per_agent_tensors[aid] = {
                    "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
                    "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
                    "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
                    "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
                    "state": torch.as_tensor(state_arr, dtype=torch.float32, device=device),
                    "ret": torch.as_tensor(ret_arr, dtype=torch.float32, device=device),
                }

                critic_states.append(state_arr)
                critic_returns.append(ret_arr)

            if len(critic_states) == 0:
                continue

            critic_state_t = torch.as_tensor(np.concatenate(critic_states, axis=0), dtype=torch.float32, device=device)
            critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, axis=0), dtype=torch.float32, device=device)

            # PPO epochs: update critic once on merged data, then each actor on its own data.
            for _ in range(args.ppo_epochs):
                # Critic update
                c_n = critic_state_t.shape[0]
                c_idx = np.arange(c_n)
                np.random.shuffle(c_idx)
                for start in range(0, c_n, args.minibatch_size):
                    mb = c_idx[start : start + args.minibatch_size]
                    value_pred = critic(critic_state_t[mb])
                    critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()

                    critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                    critic_optim.step()

                # Per-agent actor update
                for aid in agent_ids:
                    if aid not in per_agent_tensors:
                        continue

                    agent_obs_t = per_agent_tensors[aid]["obs"]
                    agent_act_t = per_agent_tensors[aid]["act"]
                    agent_old_logp_t = per_agent_tensors[aid]["old_logp"]
                    agent_adv_t = per_agent_tensors[aid]["adv"]

                    a_n = agent_obs_t.shape[0]
                    a_idx = np.arange(a_n)
                    np.random.shuffle(a_idx)

                    for start in range(0, a_n, args.minibatch_size):
                        mb = a_idx[start : start + args.minibatch_size]
                        new_logp, entropy = actors[aid].evaluate_actions(agent_obs_t[mb], agent_act_t[mb])

                        ratio = torch.exp(new_logp - agent_old_logp_t[mb])
                        surr1 = ratio * agent_adv_t[mb]
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * agent_adv_t[mb]
                        actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()

                        actor_optims[aid].zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actors[aid].parameters(), args.max_grad_norm)
                        actor_optims[aid].step()

            if update % args.log_interval == 0:
                if completed_episode_returns:
                    avg_ret = float(np.mean(completed_episode_returns))
                    ret_text = f"avg_completed_ep_return(rollout)={avg_ret:.2f}"
                else:
                    avg_ret = float(np.mean([episode_rewards[aid] for aid in agent_ids]))
                    ret_text = f"avg_partial_ep_return(rollout_end)={avg_ret:.2f}"

                detail = ", ".join([f"agent{aid}={episode_rewards[aid]:.2f}" for aid in agent_ids])
                print(
                    f"[Update {update}/{args.updates}] "
                    f"episodes_finished_in_rollout={len(completed_episode_returns)}, "
                    f"{ret_text}, current_episode_running_return=({detail})"
                )

            if update % args.save_interval == 0:
                ckpt = os.path.join(save_dir, f"mappo_ctde_mappo_update_{update}.pt")
                torch.save(
                    {
                        "update": update,
                        "actors": {aid: actors[aid].state_dict() for aid in agent_ids},
                        "critic": critic.state_dict(),
                        "actor_optims": {aid: actor_optims[aid].state_dict() for aid in agent_ids},
                        "critic_optim": critic_optim.state_dict(),
                        "episode_idx": episode_idx,
                        "args": vars(args),
                    },
                    ckpt,
                )
                print(f"Checkpoint saved: {ckpt}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="MAPPO (CTDE, non-shared actor policies) without RLlib")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

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
    parser.add_argument("--render", action="store_true", help="Enable MAPPOVisualizer during rollout")

    parser.add_argument("--save-dir", type=str, default="./logs/minimal_mappo_ckpts")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
