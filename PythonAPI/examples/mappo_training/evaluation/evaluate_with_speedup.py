import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

"""
基于提速环境的对抗策略评估脚本
功能：
1. 在正常智能体策略基础上进行提速，使其更激进
2. 评估对抗智能体在激进场景下的碰撞诱导效果
3. 对比有/无对抗时的碰撞率等指标
"""

import argparse
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.speedup_env_wrapper import SpeedupEnvWrapper, AdaptiveSpeedupWrapper, create_speedup_env
from models.actor_critic import Actor


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


# =============================================================================
# 评估数据结构
# =============================================================================

@dataclass
class EpisodeMetrics:
    """单个回合的指标"""
    episode_id: int
    has_adversary: bool
    speedup_mode: str  # "none", "fixed", "adaptive"
    success: List[bool]
    collision: List[bool]
    total_rewards: List[float]
    episode_length: int
    avg_speed: List[float]
    max_speed: List[float]
    completion_time: Optional[List[float]]


@dataclass
class EvaluationResult:
    """完整评估结果"""
    normal_agent_ids: List[int]
    adversarial_agent_id: Optional[int]
    num_episodes: int
    speedup_mode: str
    baseline_metrics: List[EpisodeMetrics]  # 无对抗
    adversarial_metrics: List[EpisodeMetrics]  # 有对抗


# =============================================================================
# 评估器类
# =============================================================================

class SpeedupEvaluator:
    """提速环境评估器"""

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
            self._load_custom_positions(args.custom_positions)

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

    def _load_custom_positions(self, positions_input: str):
        """加载自定义位置"""
        try:
            # 首先尝试作为 JSON 字符串解析
            self.custom_positions = json.loads(positions_input)
            print(f"已加载自定义位置: {self.custom_positions}")
        except json.JSONDecodeError:
            # 如果失败，尝试作为文件路径读取
            if os.path.exists(positions_input):
                try:
                    with open(positions_input, 'r', encoding='utf-8') as f:
                        self.custom_positions = json.load(f)
                    print(f"已从文件加载自定义位置: {self.custom_positions}")
                except Exception as e:
                    print(f"警告: 无法读取位置文件 - {e}")
            else:
                print(f"警告: 无法解析自定义位置，且文件不存在: {positions_input}")

    def _create_base_env(self, num_agents: int) -> CarlaMultiAgentEnv:
        """创建基础环境"""
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
                spawn_vehicles=False
            )
            env.set_custom_start_positions(self.custom_positions)
            print(f"正在使用自定义位置生成车辆...")
            env._spawn_vehicles()
        else:
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

    def _create_speedup_env(self, num_agents: int, speedup_mode: str):
        """创建带提速的环境"""
        base_env = self._create_base_env(num_agents)

        if speedup_mode == "none":
            return base_env

        elif speedup_mode == "fixed":
            return SpeedupEnvWrapper(
                base_env,
                speedup_factor=self.args.speedup_factor,
                min_throttle=self.args.min_throttle,
                speedup_agent_ids=[0, 1],
            )

        elif speedup_mode == "adaptive":
            # 对于自适应模式，需要手动设置
            wrapper = AdaptiveSpeedupWrapper(
                speedup_agent_ids=[0, 1],
                target_speed=self.args.target_speed,
                max_speedup=self.args.max_speedup,
                min_speedup=self.args.min_speedup,
                min_throttle=self.args.min_throttle,
            )
            env_wrapper = SpeedupEnvWrapper(base_env, speedup_factor=1.0)
            env_wrapper.action_wrapper = wrapper
            return env_wrapper

        else:
            raise ValueError(f"Unknown speedup mode: {speedup_mode}")

    def _run_episode(
        self,
        env,
        has_adversary: bool,
        speedup_mode: str,
        episode_id: int,
    ) -> EpisodeMetrics:
        """运行单个回合"""
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

                # 如果是自适应模式，需要先更新速度信息
                if speedup_mode == "adaptive" and hasattr(env, 'action_wrapper'):
                    env.action_wrapper.update_speeds(env.vehicles)

                with torch.no_grad():
                    # 正常智能体
                    for aid in normal_agent_ids:
                        if aid not in obs_dict or aid in done_agents:
                            continue
                        obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                        action_tanh = self.normal_actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                        action_dict[aid] = action_tanh_to_env(action_tanh)

                    # 对抗智能体
                    if has_adversary and adv_agent_id is not None:
                        if adv_agent_id in obs_dict and adv_agent_id not in done_agents:
                            if self.adversarial_actor is not None:
                                obs_t = torch.as_tensor(obs_dict[adv_agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
                                action_tanh = self.adversarial_actor.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                                action_dict[adv_agent_id] = action_tanh_to_env(action_tanh)
                            else:
                                fallback_aid = 0
                                obs_t = torch.as_tensor(obs_dict[adv_agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
                                action_tanh = self.normal_actors[fallback_aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                                action_dict[adv_agent_id] = action_tanh_to_env(action_tanh)

                # 执行动作
                next_obs, rewards, dones, step_info = env.step(action_dict)

                # 记录奖励
                for aid, r in rewards.items():
                    episode_rewards[aid] += r

                # 记录速度
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
        max_speed = [
            float(np.max(episode_speeds[i])) if len(episode_speeds[i]) > 0 else 0.0
            for i in range(num_agents)
        ]

        return EpisodeMetrics(
            episode_id=episode_id,
            has_adversary=has_adversary,
            speedup_mode=speedup_mode,
            success=success,
            collision=collision,
            total_rewards=total_rewards,
            episode_length=step_count,
            avg_speed=avg_speed,
            max_speed=max_speed,
            completion_time=completion_time,
        )

    def evaluate(self) -> Dict[str, EvaluationResult]:
        """执行完整评估（对多种提速模式进行评估）"""
        results = {}

        speedup_modes = []
        if self.args.eval_no_speedup:
            speedup_modes.append("none")
        if self.args.eval_fixed_speedup:
            speedup_modes.append("fixed")
        if self.args.eval_adaptive_speedup:
            speedup_modes.append("adaptive")

        if not speedup_modes:
            speedup_modes = ["fixed"]  # 默认只评估固定提速

        for speedup_mode in speedup_modes:
            print(f"\n{'='*60}")
            print(f"评估提速模式: {speedup_mode}")
            print(f"{'='*60}")

            baseline_metrics: List[EpisodeMetrics] = []
            adversarial_metrics: List[EpisodeMetrics] = []

            # 1. 评估基线（无对抗）
            print(f"\n阶段 1/2: 评估基线（无对抗）")
            env = self._create_speedup_env(num_agents=2, speedup_mode=speedup_mode)
            try:
                for ep in range(self.args.num_episodes):
                    print(f"  基线回合 {ep + 1}/{self.args.num_episodes}")
                    metrics = self._run_episode(env, has_adversary=False, speedup_mode=speedup_mode, episode_id=ep)
                    baseline_metrics.append(metrics)
                    self._print_episode_summary(metrics, normal_agent_ids=[0, 1])
            finally:
                env.close()

            # 2. 评估对抗（有对抗）
            print(f"\n阶段 2/2: 评估对抗效果")
            env = self._create_speedup_env(num_agents=3, speedup_mode=speedup_mode)
            try:
                for ep in range(self.args.num_episodes):
                    print(f"  对抗回合 {ep + 1}/{self.args.num_episodes}")
                    metrics = self._run_episode(env, has_adversary=True, speedup_mode=speedup_mode, episode_id=ep)
                    adversarial_metrics.append(metrics)
                    self._print_episode_summary(metrics, normal_agent_ids=[0, 1], adv_agent_id=2)
            finally:
                env.close()

            results[speedup_mode] = EvaluationResult(
                normal_agent_ids=[0, 1],
                adversarial_agent_id=2,
                num_episodes=self.args.num_episodes,
                speedup_mode=speedup_mode,
                baseline_metrics=baseline_metrics,
                adversarial_metrics=adversarial_metrics,
            )

        return results

    def _print_episode_summary(self, metrics: EpisodeMetrics, normal_agent_ids: List[int], adv_agent_id: Optional[int] = None):
        """打印回合摘要"""
        mode = "有对抗" if metrics.has_adversary else "无对抗"
        print(f"    模式: {mode}, 长度: {metrics.episode_length} 步")

        for aid in normal_agent_ids:
            if aid < len(metrics.success):
                status = "成功" if metrics.success[aid] else ("碰撞" if metrics.collision[aid] else "未完成")
                print(f"    智能体 {aid}: 奖励={metrics.total_rewards[aid]:.2f}, 状态={status}, "
                      f"平均速度={metrics.avg_speed[aid]:.2f} m/s, 最大速度={metrics.max_speed[aid]:.2f} m/s")

        if adv_agent_id is not None and adv_agent_id < len(metrics.success):
            print(f"    对抗智能体: 奖励={metrics.total_rewards[adv_agent_id]:.2f}")


def print_evaluation_report(results: Dict[str, EvaluationResult], output_dir: str):
    """打印并保存评估报告"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"提速评估综合报告")
    print(f"{'='*80}")

    all_summary_data = {}

    for speedup_mode, result in results.items():
        print(f"\n{'-'*80}")
        print(f"提速模式: {speedup_mode}")
        print(f"{'-'*80}")

        normal_ids = result.normal_agent_ids

        # 计算统计数据
        def get_normal_stats(metrics_list: List[EpisodeMetrics], aid: int):
            rewards = [m.total_rewards[aid] for m in metrics_list if aid < len(m.total_rewards)]
            success_rate = sum(1 for m in metrics_list if aid < len(m.success) and m.success[aid]) / len(metrics_list)
            collision_rate = sum(1 for m in metrics_list if aid < len(m.collision) and m.collision[aid]) / len(metrics_list)
            avg_speed = np.mean([m.avg_speed[aid] for m in metrics_list if aid < len(m.avg_speed)])
            max_speed = np.max([m.max_speed[aid] for m in metrics_list if aid < len(m.max_speed)])
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
                "max_speed": float(max_speed),
                "avg_completion": float(avg_completion) if avg_completion else None,
            }

        baseline_stats = {aid: get_normal_stats(result.baseline_metrics, aid) for aid in normal_ids}
        adv_stats = {aid: get_normal_stats(result.adversarial_metrics, aid) for aid in normal_ids}

        # 打印表格
        print(f"\n{'指标':<20} {'基线(Agent0)':<15} {'对抗(Agent0)':<15} {'变化':<15} {'基线(Agent1)':<15} {'对抗(Agent1)':<15} {'变化':<15}")
        print(f"{'-'*110}")

        for metric_name, key in [
            ("平均奖励", "avg_reward"),
            ("成功率", "success_rate"),
            ("碰撞率", "collision_rate"),
            ("平均速度", "avg_speed"),
            ("最大速度", "max_speed"),
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
                    elif key in ["avg_speed", "max_speed"]:
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

            print(''.join(row))

        # 计算对抗干扰综合评分
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

        print(f"\n正常智能体平均奖励下降: {avg_reward_drop:.1f}%")
        print(f"正常智能体成功率下降: {avg_success_drop:.1f} pp")
        print(f"正常智能体碰撞率上升: {avg_collision_increase:.1f} pp")

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

        # 保存数据
        all_summary_data[speedup_mode] = {
            "baseline_stats": baseline_stats,
            "adv_stats": adv_stats,
            "summary": {
                "avg_reward_drop": float(avg_reward_drop),
                "avg_success_drop": float(avg_success_drop),
                "avg_collision_increase": float(avg_collision_increase),
                "effectiveness_score": float(effectiveness_score),
            }
        }

    # 跨模式对比
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"跨模式对比 - 碰撞率")
        print(f"{'='*80}")
        print(f"{'模式':<15} {'基线碰撞':<15} {'对抗碰撞':<15} {'碰撞提升':<15}")
        print(f"{'-'*60}")

        for speedup_mode, result in results.items():
            data = all_summary_data[speedup_mode]
            base_collision = np.mean([data["baseline_stats"][aid]["collision_rate"] for aid in [0, 1]])
            adv_collision = np.mean([data["adv_stats"][aid]["collision_rate"] for aid in [0, 1]])
            collision_increase = data["summary"]["avg_collision_increase"]
            print(f"{speedup_mode:<15} {base_collision:<15.1%} {adv_collision:<15.1%} {collision_increase:<15.1f} pp")

    # 保存详细结果
    result_file = os.path.join(output_dir, "speedup_evaluation_result.json")
    full_data = {
        "summary": all_summary_data,
        "raw": {
            mode: {
                "baseline": [asdict(m) for m in result.baseline_metrics],
                "adversarial": [asdict(m) for m in result.adversarial_metrics],
            }
            for mode, result in results.items()
        }
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存至: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="基于提速环境的对抗策略评估")

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
    parser.add_argument("--output-dir", type=str, default="./logs/speedup_evaluation",
                        help="评估结果输出目录")

    # 提速模式选择
    parser.add_argument("--eval-no-speedup", action="store_true",
                        help="评估无提速模式（作为对照）")
    parser.add_argument("--eval-fixed-speedup", action="store_true",
                        help="评估固定提速模式")
    parser.add_argument("--eval-adaptive-speedup", action="store_true",
                        help="评估自适应提速模式")

    # 固定提速参数
    parser.add_argument("--speedup-factor", type=float, default=1.5,
                        help="固定提速系数")
    parser.add_argument("--min-throttle", type=float, default=0.3,
                        help="最小油门")

    # 自适应提速参数
    parser.add_argument("--target-speed", type=float, default=15.0,
                        help="目标速度 (m/s)")
    parser.add_argument("--max-speedup", type=float, default=2.0,
                        help="最大提速系数")
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="最小提速系数")

    # 自定义车辆位置
    parser.add_argument("--custom-positions", type=str, default="",
                        help="自定义车辆初始化位置（JSON字符串或文件路径）")

    args = parser.parse_args()

    # 如果没有指定任何模式，默认评估固定提速
    if not args.eval_no_speedup and not args.eval_fixed_speedup and not args.eval_adaptive_speedup:
        args.eval_fixed_speedup = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建评估器
    evaluator = SpeedupEvaluator(args, device)

    # 执行评估
    results = evaluator.evaluate()

    # 生成报告
    print_evaluation_report(results, args.output_dir)


if __name__ == "__main__":
    main()
