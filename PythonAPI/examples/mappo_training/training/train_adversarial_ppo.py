"""
对抗训练脚本
- Agent 0, 1: 加载预训练MAPPO策略（冻结）
- Agent 2: 对抗智能体，使用PPO训练
- 对抗智能体奖励 = -(Agent 0 + Agent 1)的平均奖励
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, PerAgentCritic

# 可视化器导入
try:
    from visualization.visualize_adversarial_training import AdversarialVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # 创建 TensorBoard SummaryWriter（直接使用，不通过有问题的 MAPPOLogger）
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('./logs/adversarial_training', current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 创建可视化器（如果可用）
    visualizer = None
    if HAS_VISUALIZER and not args.no_viz:
        try:
            import threading
            visualizer = AdversarialVisualizer(
                host=args.host,
                port=args.port,
                width=800,
                height=750,
                fps=30
            )
            # 在单独线程中运行可视化
            viz_thread = threading.Thread(target=visualizer.run, daemon=True)
            viz_thread.start()
            print("可视化已启动")
        except Exception as e:
            print(f"无法启动可视化: {e}")
            visualizer = None

    # 加载预训练checkpoint（处理路径格式问题）
    pretrained_path = args.pretrained_ckpt

    # 去除开头的反斜杠
    while pretrained_path.startswith('\\') or pretrained_path.startswith('/'):
        pretrained_path = pretrained_path[1:]

    # 转换路径分隔符
    pretrained_path = pretrained_path.replace('\\', os.sep).replace('/', os.sep)

    print(f"正在加载预训练checkpoint: {pretrained_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"文件是否存在: {os.path.exists(pretrained_path)}")

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练checkpoint不存在: {pretrained_path}")

    ckpt = torch.load(pretrained_path, map_location=device)
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

    # === 新增：用预训练权重初始化对抗智能体 ===
    adv_actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
    if adversarial_id in pretrained_actors_state:
        adv_actor.load_state_dict(pretrained_actors_state[adversarial_id])
        print("对抗智能体已使用预训练权重初始化")
    else:
        # 如果没有Agent 2的预训练权重，用Agent 0或1的权重
        fallback_agent_id = 0
        if fallback_agent_id in pretrained_actors_state:
            adv_actor.load_state_dict(pretrained_actors_state[fallback_agent_id])
            print(f"对抗智能体使用 Agent {fallback_agent_id} 的预训练权重初始化")
    # ===========================================

    adv_critic = PerAgentCritic(obs_dim=19, hidden=args.hidden_dim).to(device)

    adv_actor_optim = optim.Adam(adv_actor.parameters(), lr=args.lr_actor)
    adv_critic_optim = optim.Adam(adv_critic.parameters(), lr=args.lr_critic)

    # 保存目录
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"对抗训练开始...")
    print(f"对抗智能体ID: {adversarial_id}")
    print(f"预训练智能体(冻结): {list(pretrained_actors.keys())}")

    global_step = 0
    episode_idx = 1

    try:
        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer()
            obs_dict, _ = env.reset()
            episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
            episode_step = 0
            completed_episode_returns = []

            # 计算当前对抗系数
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

            # 收集rollout
            for _ in range(args.rollout_steps):
                global_step += 1
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

                # 记录 step 级别指标到 TensorBoard（轻量级，只记录关键指标）
                if global_step % args.log_interval == 0:
                    for aid, r in rewards.items():
                        writer.add_scalar(f'Agent_{aid}/Reward/Step', r, global_step)

                    if adversarial_id in step_cache:
                        writer.add_scalar(f'Agent_{adversarial_id}/Value', step_cache[adversarial_id]["value"], global_step)

                    writer.add_scalar('Adversarial/Coef', current_adversarial_coef, global_step)

                # 更新可视化器奖励
                if visualizer is not None:
                    visualizer.update_rewards(current_rewards=rewards, episode_rewards=episode_rewards)

                # 计算对抗智能体的奖励：混合奖励
                if adversarial_id in obs_dict:
                    # 1. 获取环境返回的基础对抗奖励（含驾驶和切入奖励）
                    base_adv_reward = rewards.get(adversarial_id, 0.0)

                    # 2. 获取其他智能体的奖励（用于零和对抗项）
                    avg_pretrained_reward = (rewards.get(0, 0.0) + rewards.get(1, 0.0)) / 2.0

                    # 3. 组合奖励 = 基础对抗奖励 + 对抗系数 * (-avg_others)
                    adversarial_reward = base_adv_reward + current_adversarial_coef * (-avg_pretrained_reward) * 0.01

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

                    # 记录 episode 级别指标
                    writer.add_scalar(f'Agent_{adversarial_id}/Reward/Episode_Total', episode_rewards[adversarial_id], episode_idx)
                    writer.add_scalar(f'Agent_0/Reward/Episode_Total', episode_rewards[0], episode_idx)
                    writer.add_scalar(f'Agent_1/Reward/Episode_Total', episode_rewards[1], episode_idx)
                    writer.add_scalar('Global/Episode_Length', episode_step, episode_idx)

                    obs_dict, _ = env.reset()
                    episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
                    episode_step = 0
                    episode_idx += 1

            # PPO更新对抗智能体
            final_actor_loss = 0.0
            final_critic_loss = 0.0
            final_entropy = 0.0

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
                act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=device)
                old_logp_t = torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device)
                adv_t = torch.as_tensor(adv_arr, dtype=torch.float32, device=device)
                ret_t = torch.as_tensor(ret_arr, dtype=torch.float32, device=device)

                n = obs_t.shape[0]
                idx = np.arange(n)

                # PPO epochs
                for epoch in range(args.ppo_epochs):
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

                        # 保存最后一个batch的损失用于记录
                        if epoch == args.ppo_epochs - 1 and start + args.minibatch_size >= n:
                            final_actor_loss = actor_loss.item()
                            final_critic_loss = critic_loss.item()
                            final_entropy = entropy.mean().item()

            # 记录 update 级别指标
            writer.add_scalar(f'Agent_{adversarial_id}/Training/Loss_Actor', final_actor_loss, update)
            writer.add_scalar(f'Agent_{adversarial_id}/Training/Loss_Critic', final_critic_loss, update)
            writer.add_scalar(f'Agent_{adversarial_id}/Training/Entropy', final_entropy, update)

            pretrained_avg = (episode_rewards[0] + episode_rewards[1]) / 2.0
            writer.add_scalar('Adversarial/Pretrained_Avg_Reward', pretrained_avg, update)

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
                    f"pretrained_avg={pretrained_avg:.2f}, "
                    f"adv_coef={current_adversarial_coef:.2f}"
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
        # 关闭 TensorBoard writer
        writer.close()
        print("TensorBoard 日志已保存")

        if visualizer is not None:
            visualizer.close()
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
    parser.add_argument("--no-viz", action="store_true", help="不启动可视化")

    # 对抗训练参数
    parser.add_argument("--adversarial-coef", type=float, default=1.0,
                        help="对抗项的权重系数")
    parser.add_argument("--curriculum-learning", action="store_true",
                        help="启用课程学习")
    parser.add_argument("--warmup-updates", type=int, default=100,
                        help="课程学习热身轮数（只用基础奖励）")
    parser.add_argument("--ramp-up-updates", type=int, default=200,
                        help="课程学习对抗系数渐增轮数")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)

