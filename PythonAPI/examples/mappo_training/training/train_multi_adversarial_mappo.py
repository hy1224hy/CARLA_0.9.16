"""
多对抗智能体 MAPPO 训练脚本
- 合作团队 (Agents 0..N-1): 加载预训练MAPPO策略（冻结）
- 对抗团队 (Agents N..N+M-1): MAPPO (CTDE) 训练
  - 统一对抗奖励函数 = 基础驾驶 + cut_in + blocking + 零和项
  - 共享 CentralizedCritic + 各独立 Actor
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import datetime
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, CentralizedCritic

# 可视化器导入
try:
    from visualization.visualize_adversarial_training import AdversarialVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def set_seed(seed: int):
    # 修复 #6：完整播种
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Transition:
    obs: np.ndarray
    global_state: np.ndarray
    action_z: np.ndarray          # 修复 #4：存 pre-tanh z
    log_prob: float
    reward: float
    done: float                   # 回合边界（切断 GAE trace）
    terminated: float             # 场景真正终止（切断 value bootstrap）；超时截断时为 0
    value: float                  # 团队共享 V(s)（联合奖励 → 单一 value，无 #1 冲突）
    next_value: float = 0.0       # done 步：reset 前对终止状态的 bootstrap 值


class RolloutBuffer:
    def __init__(self, agent_ids: List[int]):
        self.data: Dict[int, List[Transition]] = {aid: [] for aid in agent_ids}

    def add(self, agent_id: int, transition: Transition):
        self.data[agent_id].append(transition)

    def get_agent_data(self, agent_id: int) -> List[Transition]:
        return self.data[agent_id]

    def clear(self):
        for k in self.data:
            self.data[k] = []


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


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    terminateds: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE，区分真正终止与截断，显式 bootstrap（修复 #2 + #3）。"""
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        mask_value = 1.0 - terminateds[t]
        mask_trace = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask_value - values[t]
        gae = delta + gamma * gae_lambda * mask_trace * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    total_agents = args.num_cooperative + args.num_adversarial
    cooperative_ids = list(range(args.num_cooperative))
    adversarial_ids = list(range(args.num_cooperative, total_agents))

    print(f"多对抗 MAPPO 训练配置:")
    print(f"  总智能体数: {total_agents}")
    print(f"  合作团队 (冻结): {cooperative_ids}")
    print(f"  对抗团队 (训练): {adversarial_ids}")

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
        seed=args.seed,
    )

    # TensorBoard
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('./logs/multi_adversarial_mappo', current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer_obj = torch.utils.tensorboard.SummaryWriter if hasattr(torch.utils, 'tensorboard') else None
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 可视化器
    visualizer = None
    if HAS_VISUALIZER and not args.no_viz:
        try:
            import threading
            visualizer = AdversarialVisualizer(
                host=args.host,
                port=args.port,
                width=800,
                height=750,
                fps=30,
            )
            viz_thread = threading.Thread(target=visualizer.run, daemon=True)
            viz_thread.start()
            print("可视化已启动")
        except Exception as e:
            print(f"无法启动可视化: {e}")
            visualizer = None

    # ── 加载预训练 checkpoint ──
    pretrained_path = args.pretrained_ckpt
    # 兼容路径格式修复
    while pretrained_path.startswith('\\') or pretrained_path.startswith('/'):
        pretrained_path = pretrained_path[1:]
    pretrained_path = pretrained_path.replace('\\', os.sep).replace('/', os.sep)

    print(f"正在加载预训练checkpoint: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练checkpoint不存在: {pretrained_path}")

    ckpt = torch.load(pretrained_path, map_location=device)
    pretrained_actors_state = ckpt.get("actors", {})

    # 创建冻结的合作团队 Actor
    pretrained_actors: Dict[int, Actor] = {}
    for aid in cooperative_ids:
        if aid not in pretrained_actors_state:
            raise KeyError(f"Checkpoint中缺少agent {aid}的权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False
        pretrained_actors[aid] = actor

    print(f"已加载合作团队 Actor: {list(pretrained_actors.keys())}")

    # ── 创建对抗团队网络 ──
    adversarial_state_dim = env.get_adversarial_global_state().shape[0]

    adversarial_actors: Dict[int, Actor] = {}
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}

    for aid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        # 用预训练权重初始化对抗智能体
        if aid in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[aid])
            print(f"对抗智能体 {aid} 使用预训练权重初始化")
        elif cooperative_ids and cooperative_ids[0] in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[cooperative_ids[0]])
            print(f"对抗智能体 {aid} 使用 Agent {cooperative_ids[0]} 权重初始化")
        adversarial_actors[aid] = actor
        adversarial_actor_optims[aid] = optim.Adam(actor.parameters(), lr=args.lr_actor)

    # 对抗团队集中式 Critic
    adversarial_critic = CentralizedCritic(
        state_dim=adversarial_state_dim, hidden=args.hidden_dim
    ).to(device)
    adversarial_critic_optim = optim.Adam(adversarial_critic.parameters(), lr=args.lr_critic)

    # 保存目录
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"对抗训练开始 (MAPPO CTDE)...")

    global_step = 0
    episode_idx = 1

    try:
        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer(adversarial_ids)
            obs_dict, _ = env.reset()
            episode_rewards = {aid: 0.0 for aid in range(total_agents)}
            episode_step = 0
            completed_episode_returns: List[float] = []
            last_step_was_done = True   # rollout 尾部 bootstrap 用（#3）

            # 课程学习：计算当前对抗系数
            if args.curriculum_learning:
                if update < args.warmup_updates:
                    current_adversarial_coef = 0.0
                elif update < args.warmup_updates + args.ramp_up_updates:
                    progress = (update - args.warmup_updates) / args.ramp_up_updates
                    current_adversarial_coef = args.adversarial_coef * progress
                else:
                    current_adversarial_coef = args.adversarial_coef
            else:
                current_adversarial_coef = args.adversarial_coef

            # ── Rollout ──
            for _ in range(args.rollout_steps):
                global_step += 1
                episode_step += 1
                action_dict: Dict[int, np.ndarray] = {}
                step_cache: Dict[int, Dict] = {}

                # 获取对抗团队全局状态
                adversarial_global_np = env.get_adversarial_global_state().astype(np.float32)
                adversarial_global_t = torch.as_tensor(
                    adversarial_global_np, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    adversarial_global_value = float(adversarial_critic(adversarial_global_t).item())

                    # 合作团队推理 (deterministic, frozen)
                    for aid in cooperative_ids:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        action_tanh = pretrained_actors[aid].get_deterministic_action(obs_t)
                        action_tanh_np = action_tanh.squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh_np)

                    # 对抗团队推理 (stochastic, training)
                    for aid in adversarial_ids:
                        if aid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                        # 修复 #4：取 pre-tanh z
                        action_tanh_t, logp_t, z_t = adversarial_actors[aid].sample_action_z(obs_t)

                        action_tanh_np = action_tanh_t.squeeze(0).cpu().numpy()
                        action_z_np = z_t.squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh_np)

                        step_cache[aid] = {
                            "obs": obs_dict[aid].copy(),
                            "global_state": adversarial_global_np.copy(),
                            "action_z": action_z_np,
                            "logp": float(logp_t.item()),
                            "value": adversarial_global_value,
                        }

                # 执行动作
                next_obs, rewards, dones, info = env.step(action_dict)
                terminated_info = info.get('terminated', {})

                # 更新可视化器
                if visualizer is not None:
                    visualizer.update_rewards(current_rewards=rewards, episode_rewards=episode_rewards)

                # 计算对抗智能体的组合奖励并存储 transitions
                step_transitions: Dict[int, Transition] = {}
                for aid in adversarial_ids:
                    if aid not in step_cache:
                        continue

                    # 1. 环境返回的对抗奖励（含 cut_in + blocking）
                    base_adv_reward = rewards.get(aid, 0.0)

                    # 2. 合作团队平均奖励（用于零和项）
                    coop_rewards_list = [rewards.get(cid, 0.0) for cid in cooperative_ids if cid in rewards]
                    avg_coop_reward = np.mean(coop_rewards_list) if coop_rewards_list else 0.0

                    # 3. 组合奖励 = 对抗环境奖励 + zero_sum_term * 0.01
                    combined_reward = base_adv_reward + current_adversarial_coef * (-avg_coop_reward) * 0.01

                    episode_rewards[aid] += combined_reward

                    tr = Transition(
                        obs=step_cache[aid]["obs"],
                        global_state=step_cache[aid]["global_state"],
                        action_z=step_cache[aid]["action_z"],
                        log_prob=step_cache[aid]["logp"],
                        reward=combined_reward,
                        done=float(dones.get(aid, False)),
                        terminated=0.0,   # 在 episode 结束时按场景级别回填
                        value=step_cache[aid]["value"],
                    )
                    buffer.add(aid, tr)
                    step_transitions[aid] = tr

                # 累积合作团队奖励（仅监控）
                for aid in cooperative_ids:
                    if aid in rewards:
                        episode_rewards[aid] += rewards[aid]

                # TensorBoard step 级别
                if global_step % args.log_interval == 0:
                    for aid, r in rewards.items():
                        writer.add_scalar(f'Agent_{aid}/Reward/Step', r, global_step)
                    writer.add_scalar('Adversarial/Coef', current_adversarial_coef, global_step)

                obs_dict = next_obs

                # Episode 终止
                episode_done = all(dones.get(aid, False) for aid in range(total_agents))
                last_step_was_done = episode_done
                if episode_done:
                    # 修复 #2/#3：场景级 terminated 判定 + 截断 bootstrap。
                    # 任一 agent 真正终止（碰撞/到达）→ 场景终止，不 bootstrap；
                    # 无人终止（纯 max_steps 超时）→ 截断，需 bootstrap。
                    scene_terminated = 1.0 if any(
                        terminated_info.get(aid, False) for aid in range(total_agents)
                    ) else 0.0
                    terminal_global_np = env.get_adversarial_global_state().astype(np.float32)
                    terminal_global_t = torch.as_tensor(
                        terminal_global_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        boot_val = float(adversarial_critic(terminal_global_t).item())
                    for aid, tr in step_transitions.items():
                        tr.terminated = scene_terminated
                        tr.next_value = boot_val

                    avg_adv_return = float(np.mean([episode_rewards[aid] for aid in adversarial_ids]))
                    completed_episode_returns.append(avg_adv_return)

                    for aid in range(total_agents):
                        writer.add_scalar(f'Agent_{aid}/Reward/Episode_Total', episode_rewards[aid], episode_idx)
                    writer.add_scalar('Adversarial/Avg_Episode_Return', avg_adv_return, episode_idx)
                    writer.add_scalar('Global/Episode_Length', episode_step, episode_idx)

                    obs_dict, _ = env.reset()
                    episode_rewards = {aid: 0.0 for aid in range(total_agents)}
                    episode_step = 0
                    episode_idx += 1

            # ── MAPPO 更新 (对抗团队) ──
            # 获取对齐的时序数据
            first_agent_data = buffer.get_agent_data(adversarial_ids[0])
            if len(first_agent_data) == 0:
                continue
            num_steps = len(first_agent_data)

            # 按时间步聚合：联合奖励 + 全局状态/价值（所有对抗agent共享）
            joint_rewards = np.zeros(num_steps, dtype=np.float32)
            joint_dones = np.zeros(num_steps, dtype=np.float32)
            joint_terminated = np.zeros(num_steps, dtype=np.float32)
            joint_next_value = np.zeros(num_steps, dtype=np.float32)
            global_values = np.zeros(num_steps, dtype=np.float32)
            global_states = np.zeros((num_steps, adversarial_state_dim), dtype=np.float32)

            for t in range(num_steps):
                global_values[t] = first_agent_data[t].value
                global_states[t] = first_agent_data[t].global_state
                joint_dones[t] = first_agent_data[t].done
                joint_terminated[t] = first_agent_data[t].terminated
                for aid in adversarial_ids:
                    d = buffer.get_agent_data(aid)[t]
                    joint_rewards[t] += d.reward
                    joint_dones[t] = max(joint_dones[t], d.done)

            # 修复 #3：rollout 中途结束时对最终状态 bootstrap
            final_value = 0.0
            if not last_step_was_done:
                with torch.no_grad():
                    final_gs = env.get_adversarial_global_state().astype(np.float32)
                    final_gs_t = torch.as_tensor(final_gs, dtype=torch.float32, device=device).unsqueeze(0)
                    final_value = float(adversarial_critic(final_gs_t).item())

            # 构造每步 next_value（修复 #2/#3）
            for t in range(num_steps):
                if joint_dones[t] > 0.5:
                    joint_next_value[t] = first_agent_data[t].next_value
                elif t == num_steps - 1:
                    joint_next_value[t] = final_value
                else:
                    joint_next_value[t] = global_values[t + 1]

            # 联合 GAE：一份 advantage 所有对抗 agent 共享
            joint_adv, joint_ret = compute_gae(
                rewards=joint_rewards,
                values=global_values,
                next_values=joint_next_value,
                dones=joint_dones,
                terminateds=joint_terminated,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            joint_adv = (joint_adv - joint_adv.mean()) / (joint_adv.std() + 1e-8)

            # 构建 per-agent tensor（actor 用各自 obs/act/logp，advantage 共享）
            per_agent_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
            for aid in adversarial_ids:
                data = buffer.get_agent_data(aid)
                obs_arr = np.array([t.obs for t in data], dtype=np.float32)
                act_arr = np.array([t.action_z for t in data], dtype=np.float32)
                old_logp_arr = np.array([t.log_prob for t in data], dtype=np.float32)

                per_agent_tensors[aid] = {
                    "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
                    "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
                    "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
                    "adv": torch.as_tensor(joint_adv, dtype=torch.float32, device=device),
                }

            # Critic 训练数据：每个时间步一个全局状态 → 一个联合 return（不再重复）
            critic_state_t = torch.as_tensor(global_states, dtype=torch.float32, device=device)
            critic_ret_t = torch.as_tensor(joint_ret, dtype=torch.float32, device=device)

            final_actor_losses = {aid: 0.0 for aid in adversarial_ids}
            final_critic_loss = 0.0
            final_entropies = {aid: 0.0 for aid in adversarial_ids}

            # PPO Epochs
            for epoch in range(args.ppo_epochs):
                # ── 集中式 Critic 更新 ──
                c_n = critic_state_t.shape[0]
                c_idx = np.arange(c_n)
                np.random.shuffle(c_idx)
                for start in range(0, c_n, args.minibatch_size):
                    mb = c_idx[start: start + args.minibatch_size]
                    value_pred = adversarial_critic(critic_state_t[mb])
                    critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()

                    adversarial_critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(adversarial_critic.parameters(), args.max_grad_norm)
                    adversarial_critic_optim.step()

                # ── 各 Actor 独立更新 ──
                for aid in adversarial_ids:
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
                        mb = a_idx[start: start + args.minibatch_size]
                        # 修复 #4：用存储的 pre-tanh z 重算 log_prob
                        new_logp, entropy = adversarial_actors[aid].evaluate_actions_z(
                            agent_obs_t[mb], agent_act_t[mb]
                        )

                        ratio = torch.exp(new_logp - agent_old_logp_t[mb])
                        surr1 = ratio * agent_adv_t[mb]
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * agent_adv_t[mb]
                        actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()

                        adversarial_actor_optims[aid].zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(adversarial_actors[aid].parameters(), args.max_grad_norm)
                        adversarial_actor_optims[aid].step()

                        if epoch == args.ppo_epochs - 1 and start + args.minibatch_size >= a_n:
                            final_actor_losses[aid] = actor_loss.item()
                            final_entropies[aid] = entropy.mean().item()

            # ── 日志 ──
            for aid in adversarial_ids:
                writer.add_scalar(f'Agent_{aid}/Training/Loss_Actor', final_actor_losses[aid], update)
                writer.add_scalar(f'Agent_{aid}/Training/Entropy', final_entropies[aid], update)
            avg_coop_ep_return = float(np.mean([episode_rewards[cid] for cid in cooperative_ids]))
            writer.add_scalar('Cooperative/Avg_Reward', avg_coop_ep_return, update)

            if update % args.log_interval == 0:
                if completed_episode_returns:
                    avg_ret = float(np.mean(completed_episode_returns))
                    ret_text = f"avg_adv_ep_return={avg_ret:.2f}"
                else:
                    avg_ret = float(np.mean([episode_rewards[aid] for aid in adversarial_ids]))
                    ret_text = f"current_adv_return={avg_ret:.2f}"

                adv_detail = ", ".join([f"A{aid}={episode_rewards[aid]:.2f}" for aid in adversarial_ids])
                print(
                    f"[Update {update}/{args.updates}] "
                    f"episodes_finished={len(completed_episode_returns)}, "
                    f"{ret_text}, "
                    f"coop_avg={avg_coop_ep_return:.2f}, "
                    f"adv_coef={current_adversarial_coef:.2f}, "
                    f"adv_rewards=({adv_detail})"
                )

            # ── 保存 Checkpoint ──
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"multi_adversarial_mappo_update_{update}.pt")
                torch.save(
                    {
                        "update": update,
                        "cooperative_actors": {aid: pretrained_actors[aid].state_dict() for aid in cooperative_ids},
                        "adversarial_actors": {aid: adversarial_actors[aid].state_dict() for aid in adversarial_ids},
                        "adversarial_critic": adversarial_critic.state_dict(),
                        "adversarial_actor_optims": {aid: adversarial_actor_optims[aid].state_dict() for aid in adversarial_ids},
                        "adversarial_critic_optim": adversarial_critic_optim.state_dict(),
                        "pretrained_ckpt": args.pretrained_ckpt,
                        "cooperative_ids": cooperative_ids,
                        "adversarial_ids": adversarial_ids,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        writer.close()
        print("TensorBoard 日志已保存")
        if visualizer is not None:
            visualizer.close()
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-Adversarial MAPPO Training (CTDE)")

    # 环境参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 团队配置
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="预训练MAPPO checkpoint路径")
    parser.add_argument("--num-cooperative", type=int, default=2,
                        help="合作智能体数量")
    parser.add_argument("--num-adversarial", type=int, default=2,
                        help="对抗智能体数量")

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

    parser.add_argument("--save-dir", type=str, default="./logs/multi_adversarial_ckpts")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--no-viz", action="store_true", help="不启动可视化")

    # 对抗训练参数
    parser.add_argument("--adversarial-coef", type=float, default=1.0,
                        help="对抗项的权重系数")
    parser.add_argument("--curriculum-learning", action="store_true",
                        help="启用课程学习")
    parser.add_argument("--warmup-updates", type=int, default=100,
                        help="课程学习热身轮数")
    parser.add_argument("--ramp-up-updates", type=int, default=200,
                        help="课程学习对抗系数渐增轮数")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
