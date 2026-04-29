
"""
对抗策略评估脚本
功能：
1. 加载预训练的正常智能体和对抗智能体
2. 在有/无对抗两种模式下进行评估
3. 记录对抗干扰的有效性指标
4. 支持自定义车辆初始化位置
"""

import argparse
import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_multi_agent_env import CarlaMultiAgentEnv


# =============================================================================
# 网络定义（与训练脚本保持一致）
# =============================================================================

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

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        """获取确定性动作（用于评估）"""
        mu, _ = self.forward(obs)
        action_tanh = torch.tanh(mu)
        return action_tanh.squeeze(0).cpu().numpy()


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


# =============================================================================
# 评估数据结构
# =============================================================================

@dataclass
class EpisodeMetrics:
    """单个回合的指标"""
    episode_id: int
    has_adversary: bool
    success: List[bool]  # 每个智能体是否成功到达终点
    collision: List[bool]  # 每个智能体是否发生碰撞
    total_rewards: List[float]  # 每个智能体的总奖励
    episode_length: int
    avg_speed: List[float]  # 每个智能体的平均速度
    completion_time: Optional[List[float]]  # 完成时间（步数），None表示未完成


@dataclass
class EvaluationResult:
    """完整评估结果"""
    normal_agent_ids: List[int]
    adversarial_agent_id: Optional[int]
    num_episodes: int
    baseline_metrics: List[EpisodeMetrics]  # 无对抗
    adversarial_metrics: List[EpisodeMetrics]  # 有对抗


# =============================================================================
# 评估器类
# =============================================================================

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


class AdversarialEvaluator:
    """对抗策略评估器"""

    def __init__(
        self,
        args,
        device: torch.device,
    ):
        self.args = args
        self.device = device

        # 加载正常智能体
        self.normal_actors = self._load_normal_agents(args.pretrained_ckpt)

        # 加载对抗智能体（如果提供）
        self.adversarial_actor = None
        if args.adversarial_ckpt and os.path.exists(args.adversarial_ckpt):
            self.adversarial_actor = self._load_adversarial_agent(args.adversarial_ckpt)
            print(f"已加载对抗智能体: {args.adversarial_ckpt}")

        # 自定义车辆位置（如果提供）
        self.custom_positions = None
        if args.custom_positions:
            try:
                # 首先尝试作为 JSON 字符串解析
                self.custom_positions = json.loads(args.custom_positions)
                print(f"已加载自定义位置: {self.custom_positions}")
            except json.JSONDecodeError:
                # 如果失败，尝试作为文件路径读取
                if os.path.exists(args.custom_positions):
                    try:
                        with open(args.custom_positions, 'r', encoding='utf-8') as f:
                            self.custom_positions = json.load(f)
                        print(f"已从文件加载自定义位置: {self.custom_positions}")
                    except Exception as e:
                        print(f"警告: 无法读取位置文件 - {e}")
                else:
                    print(f"警告: 无法解析自定义位置，且文件不存在: {args.custom_positions}")

    def _load_normal_agents(self, ckpt_path: str) -> Dict[int, Actor]:
        """加载正常驾驶智能体"""
        print(f"正在加载正常智能体: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        actors_state = ckpt.get("actors", {})

        actors = {}
        for aid in [0, 1]:
            if aid not in actors_state:
                raise KeyError(f"Checkpoint中缺少agent {aid}的权重")
            actor = Actor(obs_dim=19, act_dim=3, hidden=self.args.hidden_dim).to(self.device)
            actor.load_state_dict(actors_state[aid])
            actor.eval()
            actors[aid] = actor

        print(f"已加载正常智能体: {list(actors.keys())}")
        return actors

    def _load_adversarial_agent(self, ckpt_path: str) -> Actor:
        """加载对抗智能体"""
        print(f"正在加载对抗智能体: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        actor_state = ckpt.get("adversarial_actor")

        if actor_state is None:
            raise KeyError("Checkpoint中缺少对抗智能体权重")

        actor = Actor(obs_dim=19, act_dim=3, hidden=self.args.hidden_dim).to(self.device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor

    def _create_env(self, num_agents: int) -> CarlaMultiAgentEnv:
        """创建环境"""
        # 如果有自定义位置，先生成环境但不生成车辆
        if self.custom_positions:
            print(f"\n正在创建环境（延迟生成车辆以应用自定义位置）...")
            env = CarlaMultiAgentEnv(
                host=self.args.host,
                port=self.args.port,
                town=self.args.town,
                num_agents=num_agents,
                straight_road_length=self.args.straight_road_length,
                timestep=self.args.timestep,
                max_steps=self.args.max_steps,
                spawn_vehicles=False  # 延迟生成车辆
            )
            # 设置自定义位置
            env.set_custom_start_positions(self.custom_positions)
            # 现在生成车辆
            print(f"正在使用自定义位置生成车辆...")
            env._spawn_vehicles()
        else:
            # 没有自定义位置，正常创建
            env = CarlaMultiAgentEnv(
                host=self.args.host,
                port=self.args.port,
                town=self.args.town,
                num_agents=num_agents,
                straight_road_length=self.args.straight_road_length,
                timestep=self.args.timestep,
                max_steps=self.args.max_steps,
            )

        return env

    def _setup_env_with_custom_positions(self, env: CarlaMultiAgentEnv):
        """在环境创建后设置自定义位置"""
        if not self.custom_positions:
            return

        # 使用环境的方法设置自定义位置
        env.set_custom_start_positions(self.custom_positions)

    def _run_episode(
        self,
        env: CarlaMultiAgentEnv,
        has_adversary: bool,
        episode_id: int,
    ) -> EpisodeMetrics:
        """运行单个回合"""
        # 正常reset（自定义位置已在环境创建时设置）
        obs_dict, info = env.reset()

        num_agents = env.num_agents
        normal_agent_ids = [0, 1]
        adv_agent_id = 2 if has_adversary else None

        episode_rewards = {i: 0.0 for i in range(num_agents)}
        episode_speeds = {i: [] for i in range(num_agents)}
        done_agents = set()
        completion_time = [None for _ in range(num_agents)]
        step_count = 0

        try:
            while True:
                step_count += 1
                action_dict: Dict[int, np.ndarray] = {}

                with torch.no_grad():
                    # 正常智能体
                    for aid in normal_agent_ids:
                        if aid not in obs_dict or aid in done_agents:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                        action_tanh = self.normal_actors[aid].get_deterministic_action(obs_t)
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                    # 对抗智能体
                    if has_adversary and adv_agent_id is not None:
                        if adv_agent_id in obs_dict and adv_agent_id not in done_agents:
                            if self.adversarial_actor is not None:
                                obs_t = torch.as_tensor(obs_dict[adv_agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
                                action_tanh = self.adversarial_actor.get_deterministic_action(obs_t)
                                action_dict[adv_agent_id] = action_tanh_to_env(action_tanh)
                            else:
                                # 没有对抗策略时，用正常智能体策略代替
                                fallback_aid = 0
                                obs_t = torch.as_tensor(obs_dict[adv_agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
                                action_tanh = self.normal_actors[fallback_aid].get_deterministic_action(obs_t)
                                action_dict[adv_agent_id] = action_tanh_to_env(action_tanh)

                # 执行动作
                next_obs, rewards, dones, _ = env.step(action_dict)

                # 记录奖励
                for aid, r in rewards.items():
                    episode_rewards[aid] += r

                # 记录速度（从车辆状态估计）
                for aid in range(num_agents):
                    if aid < len(env.vehicles) and env.vehicles[aid].is_alive:
                        vel = env.vehicles[aid].get_velocity()
                        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                        episode_speeds[aid].append(speed)

                # 检查完成
                for aid, done in dones.items():
                    if done and aid not in done_agents:
                        done_agents.add(aid)
                        if env.reached_target.get(aid, False):
                            completion_time[aid] = step_count

                obs_dict = next_obs

                # 检查是否全部完成
                if all(dones.get(aid, False) for aid in range(num_agents)):
                    break

        finally:
            pass

        # 计算指标
        success = [env.reached_target.get(i, False) for i in range(num_agents)]
        collision = [
            (env.collision_events.get(i, False) or env.crosswalk_invasion_events.get(i, False))
            and not env.reached_target.get(i, False)
            for i in range(num_agents)
        ]
        total_rewards = [episode_rewards.get(i, 0.0) for i in range(num_agents)]
        avg_speed = [
            float(np.mean(episode_speeds[i])) if len(episode_speeds[i]) > 0 else 0.0
            for i in range(num_agents)
        ]

        return EpisodeMetrics(
            episode_id=episode_id,
            has_adversary=has_adversary,
            success=success,
            collision=collision,
            total_rewards=total_rewards,
            episode_length=step_count,
            avg_speed=avg_speed,
            completion_time=completion_time,
        )

    def evaluate(self) -> EvaluationResult:
        """执行完整评估"""
        baseline_metrics: List[EpisodeMetrics] = []
        adversarial_metrics: List[EpisodeMetrics] = []

        # 1. 评估基线（无对抗）
        print("\n" + "=" * 60)
        print("阶段 1/2: 评估基线（无对抗）")
        print("=" * 60)

        env_baseline = self._create_env(num_agents=2)  # 只有2个正常智能体
        try:
            for ep in range(self.args.num_episodes):
                print(f"\n基线回合 {ep + 1}/{self.args.num_episodes}")
                metrics = self._run_episode(env_baseline, has_adversary=False, episode_id=ep)
                baseline_metrics.append(metrics)
                self._print_episode_summary(metrics, normal_agent_ids=[0, 1])
        finally:
            env_baseline.close()

        # 2. 评估对抗（有对抗）
        print("\n" + "=" * 60)
        print("阶段 2/2: 评估对抗效果")
        print("=" * 60)

        env_adv = self._create_env(num_agents=3)  # 2个正常 + 1个对抗
        try:
            for ep in range(self.args.num_episodes):
                print(f"\n对抗回合 {ep + 1}/{self.args.num_episodes}")
                metrics = self._run_episode(env_adv, has_adversary=True, episode_id=ep)
                adversarial_metrics.append(metrics)
                self._print_episode_summary(metrics, normal_agent_ids=[0, 1], adv_agent_id=2)
        finally:
            env_adv.close()

        return EvaluationResult(
            normal_agent_ids=[0, 1],
            adversarial_agent_id=2,
            num_episodes=self.args.num_episodes,
            baseline_metrics=baseline_metrics,
            adversarial_metrics=adversarial_metrics,
        )

    def _print_episode_summary(self, metrics: EpisodeMetrics, normal_agent_ids: List[int], adv_agent_id: Optional[int] = None):
        """打印回合摘要"""
        mode = "有对抗" if metrics.has_adversary else "无对抗"
        print(f"  模式: {mode}")
        print(f"  回合长度: {metrics.episode_length} 步")

        for aid in normal_agent_ids:
            if aid < len(metrics.success):
                status = "成功" if metrics.success[aid] else ("碰撞" if metrics.collision[aid] else "未完成")
                print(f"  智能体 {aid}: 奖励={metrics.total_rewards[aid]:.2f}, 状态={status}, "
                      f"平均速度={metrics.avg_speed[aid]:.2f} m/s")

        if adv_agent_id is not None and adv_agent_id < len(metrics.success):
            print(f"  对抗智能体: 奖励={metrics.total_rewards[adv_agent_id]:.2f}")


def print_evaluation_report(result: EvaluationResult, output_dir: str):
    """打印并保存评估报告"""
    print("\n" + "=" * 80)
    print("评估报告")
    print("=" * 80)

    normal_ids = result.normal_agent_ids

    # 计算统计数据
    def get_normal_stats(metrics_list: List[EpisodeMetrics], aid: int):
        rewards = [m.total_rewards[aid] for m in metrics_list if aid < len(m.total_rewards)]
        success_rate = sum(1 for m in metrics_list if aid < len(m.success) and m.success[aid]) / len(metrics_list)
        collision_rate = sum(1 for m in metrics_list if aid < len(m.collision) and m.collision[aid]) / len(metrics_list)
        avg_speed = np.mean([m.avg_speed[aid] for m in metrics_list if aid < len(m.avg_speed)])
        completion_times = [
            m.completion_time[aid] for m in metrics_list
            if aid < len(m.completion_time) and m.completion_time[aid] is not None
        ]
        avg_completion = np.mean(completion_times) if completion_times else None

        return {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(success_rate),
            "collision_rate": float(collision_rate),
            "avg_speed": float(avg_speed),
            "avg_completion": float(avg_completion) if avg_completion else None,
        }

    baseline_stats = {aid: get_normal_stats(result.baseline_metrics, aid) for aid in normal_ids}
    adv_stats = {aid: get_normal_stats(result.adversarial_metrics, aid) for aid in normal_ids}

    # 打印表格
    print(f"\n{'指标':<20} {'基线(Agent0)':<15} {'对抗(Agent0)':<15} {'变化':<15} {'基线(Agent1)':<15} {'对抗(Agent1)':<15} {'变化':<15}")
    print("-" * 110)

    for metric_name, key in [
        ("平均奖励", "avg_reward"),
        ("成功率", "success_rate"),
        ("碰撞率", "collision_rate"),
        ("平均速度", "avg_speed"),
        ("平均完成步数", "avg_completion"),
    ]:
        row = [f"{metric_name:<20}"]
        for aid in normal_ids:
            base_val = baseline_stats[aid][key]
            adv_val = adv_stats[aid][key]

            if base_val is not None and adv_val is not None:
                if key == "avg_reward":
                    change_pct = ((adv_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
                    row.extend([
                        f"{base_val:<15.2f}",
                        f"{adv_val:<15.2f}",
                        f"{change_pct:<15.1f}%",
                    ])
                elif key in ["success_rate", "collision_rate"]:
                    change_pct = (adv_val - base_val) * 100
                    row.extend([
                        f"{base_val:<15.1%}",
                        f"{adv_val:<15.1%}",
                        f"{change_pct:<15.1f} pp",
                    ])
                elif key == "avg_speed":
                    change_pct = ((adv_val - base_val) / base_val) * 100 if base_val != 0 else 0
                    row.extend([
                        f"{base_val:<15.2f}",
                        f"{adv_val:<15.2f}",
                        f"{change_pct:<15.1f}%",
                    ])
                elif key == "avg_completion":
                    change_pct = ((adv_val - base_val) / base_val) * 100 if base_val != 0 else 0
                    row.extend([
                        f"{base_val:<15.1f}",
                        f"{adv_val:<15.1f}",
                        f"{change_pct:<15.1f}%",
                    ])
            else:
                row.extend([f"{'-':<15}", f"{'-':<15}", f"{'-':<15}"])

        print("".join(row))

    # 计算对抗干扰综合评分
    print("\n" + "-" * 80)
    print("对抗干扰效果综合评分")
    print("-" * 80)

    avg_reward_drop = np.mean([
        ((baseline_stats[aid]['avg_reward'] - adv_stats[aid]['avg_reward']) / abs(baseline_stats[aid]['avg_reward'])) * 100
        for aid in normal_ids if baseline_stats[aid]['avg_reward'] != 0
    ])
    avg_success_drop = np.mean([
        (baseline_stats[aid]['success_rate'] - adv_stats[aid]['success_rate']) * 100
        for aid in normal_ids
    ])
    avg_collision_increase = np.mean([
        (adv_stats[aid]['collision_rate'] - baseline_stats[aid]['collision_rate']) * 100
        for aid in normal_ids
    ])

    print(f"正常智能体平均奖励下降: {avg_reward_drop:.1f}%")
    print(f"正常智能体成功率下降: {avg_success_drop:.1f} pp")
    print(f"正常智能体碰撞率上升: {avg_collision_increase:.1f} pp")

    # 简单综合评分（越高表示对抗越有效）
    effectiveness_score = (
        max(0, avg_reward_drop) * 0.4 +
        max(0, avg_success_drop) * 0.4 +
        max(0, avg_collision_increase) * 0.2
    )
    print(f"\n对抗效果综合评分 (0-100): {effectiveness_score:.1f}")

    if effectiveness_score >= 50:
        print("评价: 对抗效果显著！")
    elif effectiveness_score >= 20:
        print("评价: 有一定对抗效果")
    else:
        print("评价: 对抗效果较弱")

    # 保存结果到文件
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "evaluation_result.json")

    result_dict = {
        "baseline": {
            "metrics": [asdict(m) for m in result.baseline_metrics],
            "stats": baseline_stats,
        },
        "adversarial": {
            "metrics": [asdict(m) for m in result.adversarial_metrics],
            "stats": adv_stats,
        },
        "summary": {
            "avg_reward_drop": avg_reward_drop,
            "avg_success_drop": avg_success_drop,
            "avg_collision_increase": avg_collision_increase,
            "effectiveness_score": effectiveness_score,
        }
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存至: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="评估对抗策略的干扰效果")

    # 环境参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 模型路径
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="预训练MAPPO checkpoint路径（正常智能体）")
    parser.add_argument("--adversarial-ckpt", type=str, default="",
                        help="对抗智能体checkpoint路径（留空则用正常策略代替）")

    # 评估参数
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="每个模式下评估的回合数")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./logs/evaluation",
                        help="评估结果输出目录")

    # 自定义车辆位置
    parser.add_argument("--custom-positions", type=str, default="",
                        help='自定义车辆初始化位置（JSON字符串或文件路径），例如: '
                             '{"agent0": {"x": 100, "y": -257, "z": 0.5}, '
                             '"agent1": {"x": 100, "y": -260, "z": 0.5}} '
                             '或者传入 JSON 文件路径')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建评估器
    evaluator = AdversarialEvaluator(args, device)

    # 执行评估
    result = evaluator.evaluate()

    # 生成报告
    print_evaluation_report(result, args.output_dir)


if __name__ == "__main__":
    main()

