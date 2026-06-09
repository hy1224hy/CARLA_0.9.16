"""
基于 MAPPO 的鲁棒训练脚本
支持：
1. 随机场景训练（从多个起始位置中随机选择）
2. 课程学习（Curriculum Learning）
3. 训练时记录各场景的成功率
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from visualization.mappo_visualizer import MAPPOVisualizer
from models.actor_critic import Actor, AgentConditionedCritic


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
    terminated: float             # 真正终止（切断 value bootstrap）
    value: float                  # 该 agent 的 V(s, aid)
    scenario_name: str
    next_value: float = 0.0       # done 步：reset 前对终止状态的 bootstrap 值


class RolloutBuffer:
    def __init__(self, agent_ids: List[int]):
        self.data: Dict[int, List[Transition]] = {aid: [] for aid in agent_ids}
        self.scenario_names: List[str] = []

    def add(self, agent_id: int, transition: Transition):
        self.data[agent_id].append(transition)

    def add_scenario(self, scenario_name: str):
        self.scenario_names.append(scenario_name)

    def get_agent_data(self, agent_id: int) -> List[Transition]:
        return self.data[agent_id]

    def clear(self):
        for k in self.data:
            self.data[k] = []
        self.scenario_names = []


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


class CurriculumLearning:
    """课程学习管理器"""

    def __init__(
        self,
        scenario_dir: str,
        curriculum_stages: List[List[str]] = None,
        warmup_updates: int = 10,
        full_scenario_updates: int = 40,
    ):
        """
        Args:
            scenario_dir: 场景文件目录
            curriculum_stages: 课程阶段配置，每个阶段是一个场景文件名列表
            warmup_updates: 只使用基础场景的更新次数
            full_scenario_updates: 达到完整场景库的更新次数
        """
        self.scenario_dir = scenario_dir
        self.warmup_updates = warmup_updates
        self.full_scenario_updates = full_scenario_updates

        # 获取所有场景文件
        all_scenario_files = []
        if os.path.exists(scenario_dir):
            all_scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])

        if curriculum_stages is None:
            # 默认课程：逐步添加场景
            self.curriculum_stages = []
            # 阶段1: 基础场景
            basic_scenarios = [f for f in all_scenario_files if 'basic' in f]
            if not basic_scenarios and all_scenario_files:
                basic_scenarios = [all_scenario_files[0]]
            self.curriculum_stages.append(basic_scenarios)

            # 阶段2: 基础 + 左右偏移
            stage2 = basic_scenarios + [f for f in all_scenario_files if 'left' in f or 'right' in f]
            self.curriculum_stages.append(list(set(stage2)))

            # 阶段3: 全部场景
            self.curriculum_stages.append(all_scenario_files)
        else:
            self.curriculum_stages = curriculum_stages

        # 记录每个场景的性能
        self.scenario_success_history: Dict[str, List[float]] = {f: [] for f in all_scenario_files}
        print(f"课程学习初始化，共 {len(self.curriculum_stages)} 个阶段，{len(all_scenario_files)} 个场景")

    def get_available_scenarios(self, current_update: int) -> List[str]:
        """根据当前更新次数获取可用场景列表"""
        if current_update < self.warmup_updates:
            stage_idx = 0
        elif current_update >= self.full_scenario_updates:
            stage_idx = len(self.curriculum_stages) - 1
        else:
            progress = (current_update - self.warmup_updates) / (self.full_scenario_updates - self.warmup_updates)
            stage_idx = min(int(progress * len(self.curriculum_stages)), len(self.curriculum_stages) - 2)

        return self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]

    def record_scenario_result(self, scenario_name: str, success: bool):
        """记录场景的训练结果"""
        success_float = 1.0 if success else 0.0
        if scenario_name not in self.scenario_success_history:
            self.scenario_success_history[scenario_name] = []
        self.scenario_success_history[scenario_name].append(success_float)

        # 只保留最近的记录
        if len(self.scenario_success_history[scenario_name]) > 20:
            self.scenario_success_history[scenario_name] = self.scenario_success_history[scenario_name][-20:]

    def get_scenario_success_rate(self, scenario_name: str) -> float:
        """获取场景的历史成功率"""
        if scenario_name not in self.scenario_success_history or len(self.scenario_success_history[scenario_name]) == 0:
            return 0.0
        return float(np.mean(self.scenario_success_history[scenario_name]))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # 初始化环境
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=2,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
        spawn_vehicles=True,  # 先生成车辆以获取正确的 state_dim
        reward_version=args.reward_version,
        disable_tensorboard=True,  # 训练脚本自行管理TB，避免重复/混乱
        seed=args.seed,
    )

    # 初始化课程学习
    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
    curriculum = CurriculumLearning(
        scenario_dir=scenario_dir,
        warmup_updates=args.warmup_updates,
        full_scenario_updates=args.full_scenario_updates,
    )

    agent_ids = list(range(env.num_agents))
    state_dim = env.get_global_state().shape[0]
    print(f"全局状态维度: {state_dim}")

    # 初始化模型
    actors: Dict[int, Actor] = {
        aid: Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device) for aid in agent_ids
    }
    actor_optims: Dict[int, optim.Optimizer] = {
        aid: optim.Adam(actors[aid].parameters(), lr=args.lr_actor) for aid in agent_ids
    }

    # 修复 #1：agent-id 条件化集中式 Critic
    critic = AgentConditionedCritic(state_dim=state_dim, num_agents=env.num_agents,
                                    hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)

    # 加载预训练模型
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"加载预训练模型: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        for aid in agent_ids:
            if aid in checkpoint.get('actors', {}):
                actors[aid].load_state_dict(checkpoint['actors'][aid])
        # 旧 checkpoint 的 critic 是非条件化 CentralizedCritic，结构不兼容 → 跳过
        if 'critic' in checkpoint:
            try:
                critic.load_state_dict(checkpoint['critic'])
            except Exception as e:
                print(f"[警告] critic 结构已变更（agent-id 条件化），跳过加载旧 critic: {e}")

    visualizer = MAPPOVisualizer(width=1200, height=800, fps=30) if args.render else None

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard
    tb_log_dir = os.path.join(save_dir, 'tb_logs')
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)

    # 训练统计（累计，用于控制台打印）
    total_episodes = 0
    total_successes = 0

    try:
        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer(agent_ids)

            # 当前 update 窗口的 episode 级统计
            window_episode_returns: List[float] = []
            window_episode_lengths: List[int] = []
            window_successes: List[float] = []
            window_per_agent_returns: Dict[int, List[float]] = {aid: [] for aid in agent_ids}
            window_reward_components: Dict[str, List[float]] = {}  # 按分量名累积

            # 获取当前阶段可用的场景
            available_scenarios = curriculum.get_available_scenarios(update)
            print(f"\n更新 {update}/{args.updates}, 可用场景数: {len(available_scenarios)}")

            # 收集 rollout 数据
            for rollout_episode in range(args.rollout_episodes):
                # 随机选择一个场景
                scenario_config = env.load_random_scenario(scenario_dir, available_scenarios)
                scenario_name = scenario_config.get('description', 'unknown') if scenario_config else 'default'

                # 重置环境
                if scenario_config:
                    env.set_scenario(scenario_config)

                obs_dict, info = env.reset()
                total_episodes += 1

                episode_rewards = {aid: 0.0 for aid in agent_ids}
                ep_length = 0
                ep_done_cleanly = False
                ep_success = False

                for step in range(args.max_steps):
                    ep_length += 1
                    action_dict: Dict[int, np.ndarray] = {}
                    step_cache: Dict[int, Dict] = {}

                    global_state_np = env.get_global_state().astype(np.float32)
                    global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32, device=device).unsqueeze(0)

                    # 修复 #1：逐 agent 计算各自 V(s, aid)
                    with torch.no_grad():
                        agent_values = {
                            aid: float(critic(global_state_t, aid).item()) for aid in agent_ids
                        }

                    for aid, obs in obs_dict.items():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            # 修复 #4：取 pre-tanh z
                            action_tanh_t, logp_t, z_t = actors[aid].sample_action_z(obs_t)

                        action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
                        action_z = z_t.squeeze(0).cpu().numpy()
                        action_env = action_tanh_to_env(action_tanh)

                        action_dict[aid] = action_env
                        step_cache[aid] = {
                            "obs": obs,
                            "global_state": global_state_np,
                            "action_z": action_z,
                            "logp": float(logp_t.item()),
                            "value": agent_values[aid],
                        }

                    next_obs, rewards, dones, info = env.step(action_dict)
                    terminated_info = info.get('terminated', {})

                    step_transitions: Dict[int, Transition] = {}
                    for aid in step_cache.keys():
                        reward = float(rewards.get(aid, 0.0))
                        done = float(dones.get(aid, False))
                        terminated = float(terminated_info.get(aid, dones.get(aid, False)))
                        episode_rewards[aid] += reward

                        transition = Transition(
                            obs=step_cache[aid]["obs"],
                            global_state=step_cache[aid]["global_state"],
                            action_z=step_cache[aid]["action_z"],
                            log_prob=step_cache[aid]["logp"],
                            reward=reward,
                            done=done,
                            terminated=terminated,
                            value=step_cache[aid]["value"],
                            scenario_name=scenario_name,
                        )
                        buffer.add(aid, transition)
                        step_transitions[aid] = transition

                    if visualizer is not None:
                        visualizer.handle_events()
                        if not visualizer.running:
                            raise KeyboardInterrupt

                    # 收集奖励分量（用最后一个 step 的 info 作为代表）
                    if 'reward_details' in info:
                        for aid, details in info['reward_details'].items():
                            for key, val in details.items():
                                comp_name = f"reward/{key}"
                                if comp_name not in window_reward_components:
                                    window_reward_components[comp_name] = []
                                window_reward_components[comp_name].append(float(val))

                    obs_dict = next_obs

                    if all(dones.values()):
                        # 修复 #2：回合结束（break）前对终止状态算 bootstrap，回填本步 transition。
                        # GAE 用 terminated 标志决定是否启用（超时截断保留，真终止忽略）。
                        terminal_state_np = env.get_global_state().astype(np.float32)
                        terminal_state_t = torch.as_tensor(
                            terminal_state_np, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            for aid, tr in step_transitions.items():
                                tr.next_value = float(critic(terminal_state_t, aid).item())

                        ep_done_cleanly = True
                        ep_success = not any(info.get('collisions', {}).values())
                        total_successes += 1 if ep_success else 0
                        break

                # 记录 episode（不丢弃！无论是否 clean done 都记录）
                avg_ret = float(np.mean([episode_rewards[aid] for aid in agent_ids]))
                window_episode_returns.append(avg_ret)
                window_episode_lengths.append(ep_length)
                window_successes.append(1.0 if ep_success else 0.0)
                for aid in agent_ids:
                    window_per_agent_returns[aid].append(episode_rewards[aid])

                # 记录场景结果到课程学习器
                curriculum.record_scenario_result(scenario_name, ep_success)

            # ===== PPO 更新（收集训练诊断指标） =====
            per_agent_tensors = {}
            critic_states = []
            critic_returns = []
            critic_agent_ids = []   # 修复 #1：每条样本对应的 agent id

            for aid in agent_ids:
                data = buffer.get_agent_data(aid)
                if len(data) == 0:
                    continue

                obs_arr = np.array([t.obs for t in data], dtype=np.float32)
                state_arr = np.array([t.global_state for t in data], dtype=np.float32)
                act_arr = np.array([t.action_z for t in data], dtype=np.float32)
                old_logp_arr = np.array([t.log_prob for t in data], dtype=np.float32)
                reward_arr = np.array([t.reward for t in data], dtype=np.float32)
                done_arr = np.array([t.done for t in data], dtype=np.float32)
                terminated_arr = np.array([t.terminated for t in data], dtype=np.float32)
                value_arr = np.array([t.value for t in data], dtype=np.float32)

                # 构造每步 next_value（修复 #2）：done 步用回填的 bootstrap；其余用下一步 value。
                # 本脚本每个 rollout 收集完整 episode，末步必为 done，故无 rollout 中途截断。
                n = len(data)
                next_value_arr = np.zeros(n, dtype=np.float32)
                for t in range(n):
                    if done_arr[t] > 0.5:
                        next_value_arr[t] = data[t].next_value
                    elif t < n - 1:
                        next_value_arr[t] = value_arr[t + 1]
                    # t==n-1 且非 done（理论上不发生）→ 保持 0

                adv_arr, ret_arr = compute_gae(
                    rewards=reward_arr,
                    values=value_arr,
                    next_values=next_value_arr,
                    dones=done_arr,
                    terminateds=terminated_arr,
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
                critic_agent_ids.append(np.full(len(ret_arr), aid, dtype=np.int64))

            # 累计 PPO 指标（跨 epoch 取均值）
            ppo_metrics = {
                "critic_loss": [],
                "explained_variance": 0.0,
            }
            for aid in agent_ids:
                ppo_metrics.update({
                    f"actor_loss_{aid}": [],
                    f"entropy_{aid}": [],
                    f"approx_kl_{aid}": [],
                    f"clip_frac_{aid}": [],
                    f"grad_norm_actor_{aid}": [],
                })
            ppo_metrics["grad_norm_critic"] = []

            if len(critic_states) > 0:
                critic_state_t = torch.as_tensor(np.concatenate(critic_states, axis=0), dtype=torch.float32, device=device)
                critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, axis=0), dtype=torch.float32, device=device)
                critic_aid_t = torch.as_tensor(np.concatenate(critic_agent_ids, axis=0), dtype=torch.long, device=device)

                for epoch in range(args.ppo_epochs):
                    # --- Critic 更新 ---
                    c_n = critic_state_t.shape[0]
                    c_idx = np.arange(c_n)
                    np.random.shuffle(c_idx)
                    epoch_critic_losses = []
                    for start in range(0, c_n, args.minibatch_size):
                        mb = c_idx[start: start + args.minibatch_size]
                        value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
                        critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()

                        critic_optim.zero_grad()
                        critic_loss.backward()
                        critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                        critic_optim.step()

                        epoch_critic_losses.append(float(critic_loss.item()))
                        ppo_metrics["grad_norm_critic"].append(float(critic_grad_norm))

                    ppo_metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

                    # --- Actor 更新 ---
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

                        epoch_actor_losses = []
                        epoch_entropies = []
                        epoch_approx_kls = []
                        epoch_clip_fracs = []

                        for start in range(0, a_n, args.minibatch_size):
                            mb = a_idx[start: start + args.minibatch_size]
                            # 修复 #4：用存储的 pre-tanh z 重算 log_prob
                            new_logp, entropy = actors[aid].evaluate_actions_z(agent_obs_t[mb], agent_act_t[mb])

                            ratio = torch.exp(new_logp - agent_old_logp_t[mb])
                            surr1 = ratio * agent_adv_t[mb]
                            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * agent_adv_t[mb]
                            actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()

                            actor_optims[aid].zero_grad()
                            actor_loss.backward()
                            actor_grad_norm = nn.utils.clip_grad_norm_(actors[aid].parameters(), args.max_grad_norm)
                            actor_optims[aid].step()

                            # 收集指标
                            epoch_actor_losses.append(float(actor_loss.item()))
                            epoch_entropies.append(float(entropy.mean().item()))
                            with torch.no_grad():
                                epoch_approx_kls.append(float((new_logp - agent_old_logp_t[mb]).pow(2).mean().item()))
                                epoch_clip_fracs.append(float(
                                    ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
                                ))

                        ppo_metrics[f"actor_loss_{aid}"].append(float(np.mean(epoch_actor_losses)))
                        ppo_metrics[f"entropy_{aid}"].append(float(np.mean(epoch_entropies)))
                        ppo_metrics[f"approx_kl_{aid}"].append(float(np.mean(epoch_approx_kls)))
                        ppo_metrics[f"clip_frac_{aid}"].append(float(np.mean(epoch_clip_fracs)))
                        ppo_metrics[f"grad_norm_actor_{aid}"].append(float(actor_grad_norm))

                # 计算 explained variance（最后更新完后用整个数据集评估一次）
                with torch.no_grad():
                    all_value_preds = critic(critic_state_t, critic_aid_t)
                    var_residual = torch.var(critic_ret_t - all_value_preds)
                    var_total = torch.var(critic_ret_t)
                    explained_var = float(1.0 - var_residual / (var_total + 1e-8))
                    ppo_metrics["explained_variance"] = explained_var

            # ===== TensorBoard 记录 =====
            if update % args.log_interval == 0 and len(window_episode_returns) > 0:
                # --- Episode 性能指标 ---
                tb_writer.add_scalar('episode/avg_return',
                    float(np.mean(window_episode_returns)), update)
                tb_writer.add_scalar('episode/std_return',
                    float(np.std(window_episode_returns)), update)
                tb_writer.add_scalar('episode/avg_length',
                    float(np.mean(window_episode_lengths)), update)
                tb_writer.add_scalar('episode/success_rate_window',
                    float(np.mean(window_successes)), update)
                tb_writer.add_scalar('episode/success_rate_cumulative',
                    total_successes / max(total_episodes, 1), update)

                for aid in agent_ids:
                    if window_per_agent_returns[aid]:
                        tb_writer.add_scalar(f'episode/return_agent_{aid}',
                            float(np.mean(window_per_agent_returns[aid])), update)

                # 控制台摘要
                avg_r = float(np.mean(window_episode_returns))
                win_succ = float(np.mean(window_successes))
                print(f"更新 {update}: avg_return={avg_r:.2f}, "
                      f"success_rate(窗口)={win_succ:.2%}, "
                      f"cum_success={total_successes/max(total_episodes,1):.2%}, "
                      f"avg_ep_len={float(np.mean(window_episode_lengths)):.0f}")

                # --- 奖励分量（按 episode 聚合后取均值） ---
                if window_reward_components:
                    for comp_name, comp_values in window_reward_components.items():
                        if comp_values:
                            tb_writer.add_scalar(comp_name,
                                float(np.mean(comp_values)), update)

                # --- PPO 训练诊断（跨 epoch 取均值） ---
                if len(critic_states) > 0:
                    tb_writer.add_scalar('ppo/critic_loss',
                        float(np.mean(ppo_metrics["critic_loss"])), update)
                    tb_writer.add_scalar('ppo/critic_grad_norm',
                        float(np.mean(ppo_metrics["grad_norm_critic"])), update)
                    tb_writer.add_scalar('ppo/explained_variance',
                        ppo_metrics["explained_variance"], update)

                    for aid in agent_ids:
                        if ppo_metrics[f"actor_loss_{aid}"]:
                            tb_writer.add_scalar(f'ppo/actor_loss_agent{aid}',
                                float(np.mean(ppo_metrics[f"actor_loss_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/entropy_agent{aid}',
                                float(np.mean(ppo_metrics[f"entropy_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/approx_kl_agent{aid}',
                                float(np.mean(ppo_metrics[f"approx_kl_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/clip_frac_agent{aid}',
                                float(np.mean(ppo_metrics[f"clip_frac_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/grad_norm_actor_agent{aid}',
                                float(np.mean(ppo_metrics[f"grad_norm_actor_{aid}"])), update)

                # --- 场景指标（滑动窗口：最近 N 次该场景的成功率） ---
                for scenario_name in curriculum.scenario_success_history:
                    history = curriculum.scenario_success_history[scenario_name]
                    if len(history) >= 3:  # 至少 3 次采样才记录
                        win_rate = float(np.mean(history))
                        safe_name = scenario_name.replace('.json', '').replace(' ', '_')[:40]
                        tb_writer.add_scalar(f'scenarios/{safe_name}/success_rate_window',
                            win_rate, update)
                        tb_writer.add_scalar(f'scenarios/{safe_name}/sample_count',
                            len(history), update)

            # 保存 checkpoint
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"mappo_robust_update_{update}.pt")
                torch.save({
                    "update": update,
                    "actors": {aid: actors[aid].state_dict() for aid in agent_ids},
                    "critic": critic.state_dict(),
                    "actor_optims": {aid: actor_optims[aid].state_dict() for aid in agent_ids},
                    "critic_optim": critic_optim.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"模型保存: {ckpt_path}")

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()
        tb_writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="鲁棒 MAPPO 训练脚本（随机场景 + 课程学习）")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 训练参数
    parser.add_argument("--updates", type=int, default=100, help="总更新次数")
    parser.add_argument("--rollout-episodes", type=int, default=4, help="每次更新收集的episode数")
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)

    # 学习率
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)

    # 课程学习
    parser.add_argument("--warmup-updates", type=int, default=10, help="基础场景预热次数")
    parser.add_argument("--full-scenario-updates", type=int, default=40, help="达到全场景的更新次数")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--pretrained-checkpoint", type=str, default="", help="预训练模型路径")

    parser.add_argument("--save-dir", type=str, default="./logs/mappo_robust")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--reward-version", type=str, default="v2",
                        choices=["v1", "v2"], help="奖励函数版本 (v1=原始, v2=优化)")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)

