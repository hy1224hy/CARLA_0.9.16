"""
A/B 对比评估脚本 — 比较不同奖励函数版本训练的 checkpoint 表现

用法:
  # 单 checkpoint 评估（用当前环境默认奖励记录指标）
  python -m evaluation.compare_reward_versions --checkpoint <path.pt> --episodes 50

  # 两个 checkpoint 的 A/B 对比
  python -m evaluation.compare_reward_versions \\
      --checkpoint-a <v1_checkpoint.pt> --label-a "v1 (原始)" \\
      --checkpoint-b <v2_checkpoint.pt> --label-b "v2 (优化)" \\
      --episodes 50 --scenarios-dir data/training_scenarios

  # 随机场景评估
  python -m evaluation.compare_reward_versions \\
      --checkpoint-a ckpt_v1.pt --checkpoint-b ckpt_v2.pt \\
      --episodes 50 --use-random-scenarios
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor


# ── 评估指标 ──

@dataclass
class EvalMetrics:
    """单次评估的聚合指标"""
    label: str = ""
    checkpoint: str = ""
    num_episodes: int = 0

    success_rate: float = 0.0       # 成功率（到达终点）
    collision_rate: float = 0.0     # 碰撞率
    timeout_rate: float = 0.0       # 超时率（未到达未碰撞）

    avg_speed_ms: float = 0.0       # 平均速度 (m/s)
    avg_speed_kmh: float = 0.0      # 平均速度 (km/h)
    avg_lateral_deviation: float = 0.0  # 平均横向偏差 (m)
    avg_lane_invasions_per_ep: float = 0.0  # 每 episode 平均车道偏离次数
    avg_episode_steps: float = 0.0  # 平均 episode 步数
    avg_episode_return_v2: float = 0.0  # 以 v2 奖励衡量的平均 return

    # 分 agent 指标
    per_agent: Dict[int, Dict] = field(default_factory=dict)


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5
    if throttle_raw >= brake_raw:
        return np.array([throttle_raw, steer, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, steer, brake_raw], dtype=np.float32)


def load_actors(checkpoint_path: str, device: torch.device) -> Tuple[Dict[int, Actor], dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    num_agents = int(ckpt_args.get("num_agents", 2))
    hidden_dim = int(ckpt_args.get("hidden_dim", 128))

    actors = {}
    actors_state = ckpt.get("actors", {})
    for aid in range(num_agents):
        actor = Actor(obs_dim=19, act_dim=3, hidden=hidden_dim).to(device)
        if aid in actors_state:
            actor.load_state_dict(actors_state[aid])
        actor.eval()
        actors[aid] = actor
    return actors, ckpt_args


def run_episode(
    env: CarlaMultiAgentEnv,
    actors: Dict[int, Actor],
    device: torch.device,
    scenario_config: Optional[Dict] = None,
    deterministic: bool = True,
) -> Dict:
    """运行单个 episode，返回详细指标"""
    if scenario_config is not None:
        env.set_scenario(scenario_config)
    obs_dict, _ = env.reset()

    num_agents = env.num_agents
    episode_returns: Dict[int, float] = {aid: 0.0 for aid in range(num_agents)}
    speeds: List[float] = []
    lateral_deviations: List[float] = []
    lane_invasions: int = 0
    collision_occurred: bool = False
    success_achieved: bool = False

    step = 0
    max_steps = env.max_steps
    done = False

    while not done:
        step += 1
        action_dict: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for aid, obs in obs_dict.items():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic:
                    action_tanh = actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                else:
                    action_tanh = actors[aid].get_action(obs_t, deterministic=False).squeeze(0).cpu().numpy()
                action_dict[aid] = action_tanh_to_env(action_tanh)

        next_obs, rewards, dones, info = env.step(action_dict)

        for aid, r in rewards.items():
            episode_returns[aid] += float(r)

        # 收集质量指标
        for aid in range(num_agents):
            if aid < len(env.vehicles) and env.vehicles[aid].is_alive:
                vel = env.vehicles[aid].get_velocity()
                spd = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                speeds.append(spd)

                # 横向偏差
                transform = env.vehicles[aid].get_transform()
                wp = env.map.get_waypoint(transform.location)
                if wp:
                    veh_loc = transform.location
                    wp_loc = wp.transform.location
                    rv = wp.transform.get_right_vector()
                    vec = __import__('carla').Vector3D(
                        veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                    lat = abs(vec.x * rv.x + vec.y * rv.y)
                    lateral_deviations.append(lat)

            if env.lane_invasion_events.get(aid, False):
                lane_invasions += 1
            if env.collision_events.get(aid, False) or env.crosswalk_invasion_events.get(aid, False):
                collision_occurred = True
            if env.reached_target.get(aid, False):
                success_achieved = True

        obs_dict = next_obs
        done = all(dones.get(aid, False) for aid in range(num_agents)) or step >= max_steps

    return {
        "returns": episode_returns,
        "avg_speed_ms": float(np.mean(speeds)) if speeds else 0.0,
        "avg_lateral_deviation": float(np.mean(lateral_deviations)) if lateral_deviations else 0.0,
        "lane_invasions": lane_invasions,
        "collision": collision_occurred,
        "success": success_achieved,
        "steps": step,
        "timed_out": step >= max_steps and not collision_occurred and not success_achieved,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    label: str,
    num_episodes: int,
    scenario_configs: Optional[List[Dict]],
    env_kwargs: dict,
    device: torch.device,
    deterministic: bool = True,
) -> EvalMetrics:
    """评估单个 checkpoint"""
    actors, ckpt_args = load_actors(checkpoint_path, device)
    num_agents = int(ckpt_args.get("num_agents", 2))

    env = CarlaMultiAgentEnv(**env_kwargs)

    metrics = EvalMetrics(label=label, checkpoint=checkpoint_path, num_episodes=num_episodes)

    successes = 0
    collisions = 0
    timeouts = 0
    all_speeds = []
    all_ldevs = []
    total_lane_invasions = 0
    total_steps = 0
    total_return_v2 = 0.0

    try:
        for ep in range(1, num_episodes + 1):
            scenario = scenario_configs[ep - 1] if scenario_configs else None
            result = run_episode(env, actors, device, scenario_config=scenario, deterministic=deterministic)

            if result["success"]:
                successes += 1
            if result["collision"]:
                collisions += 1
            if result["timed_out"]:
                timeouts += 1

            all_speeds.append(result["avg_speed_ms"])
            all_ldevs.append(result["avg_lateral_deviation"])
            total_lane_invasions += result["lane_invasions"]
            total_steps += result["steps"]
            total_return_v2 += float(np.mean(list(result["returns"].values())))

            if ep % 10 == 0 or ep == 1:
                avg_ret = float(np.mean(list(result["returns"].values())))
                print(f"  [{label}] Episode {ep}/{num_episodes} | "
                      f"return={avg_ret:.1f} | speed={result['avg_speed_ms']:.1f}m/s | "
                      f"ldev={result['avg_lateral_deviation']:.3f}m | "
                      f"{'SUCCESS' if result['success'] else 'COLLISION' if result['collision'] else 'TIMEOUT'}")

    except KeyboardInterrupt:
        print(f"\n  [{label}] 评估被用户中断")
    finally:
        env.close()

    n = num_episodes
    metrics.success_rate = successes / n if n > 0 else 0.0
    metrics.collision_rate = collisions / n if n > 0 else 0.0
    metrics.timeout_rate = timeouts / n if n > 0 else 0.0
    metrics.avg_speed_ms = float(np.mean(all_speeds)) if all_speeds else 0.0
    metrics.avg_speed_kmh = metrics.avg_speed_ms * 3.6
    metrics.avg_lateral_deviation = float(np.mean(all_ldevs)) if all_ldevs else 0.0
    metrics.avg_lane_invasions_per_ep = total_lane_invasions / n if n > 0 else 0.0
    metrics.avg_episode_steps = total_steps / n if n > 0 else 0.0
    metrics.avg_episode_return_v2 = total_return_v2 / n if n > 0 else 0.0

    return metrics


def print_comparison(metrics_a: EvalMetrics, metrics_b: Optional[EvalMetrics] = None):
    """打印对比报告"""
    def fmt_val(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, float) else str(v)

    def delta_str(a, b, higher_better=True):
        if b is None:
            return ""
        if a == 0:
            return ""
        d = (b - a) / abs(a) * 100
        direction = "↑" if (d > 0) == higher_better else "↓"
        return f"{d:+.1f}% {direction}"

    print("\n" + "=" * 80)
    print("A/B 对比评估报告")
    print("=" * 80)

    if metrics_b is None:
        print(f"\n  模型: {metrics_a.label}")
        print(f"  Checkpoint: {metrics_a.checkpoint}")
        print(f"  Episodes: {metrics_a.num_episodes}")
    else:
        print(f"\n  {'指标':<25} {'A: ' + metrics_a.label:<30} {'B: ' + metrics_b.label:<30} {'变化'}")
        print(f"  {'-'*25} {'-'*30} {'-'*30} {'-'*12}")
        rows = [
            ("成功率", metrics_a.success_rate, metrics_b.success_rate, True),
            ("碰撞率", metrics_a.collision_rate, metrics_b.collision_rate, False),
            ("超时率", metrics_a.timeout_rate, metrics_b.timeout_rate, False),
            ("平均速度 (km/h)", metrics_a.avg_speed_kmh, metrics_b.avg_speed_kmh, True),
            ("平均横向偏差 (m)", metrics_a.avg_lateral_deviation, metrics_b.avg_lateral_deviation, False),
            ("每ep车道偏离次数", metrics_a.avg_lane_invasions_per_ep, metrics_b.avg_lane_invasions_per_ep, False),
            ("平均episode步数", metrics_a.avg_episode_steps, metrics_b.avg_episode_steps, False),
            ("平均episode return", metrics_a.avg_episode_return_v2, metrics_b.avg_episode_return_v2, True),
        ]
        for name, a, b, higher_better in rows:
            print(f"  {name:<25} {fmt_val(a):<30} {fmt_val(b):<30} {delta_str(a, b, higher_better)}")

    print("\n" + "=" * 80)

    # 单模型详细报告
    def print_single(m: EvalMetrics):
        print(f"\n  [{m.label}] 详细指标:")
        print(f"    Checkpoint: {m.checkpoint}")
        print(f"    成功率:    {m.success_rate*100:.1f}%")
        print(f"    碰撞率:    {m.collision_rate*100:.1f}%")
        print(f"    超时率:    {m.timeout_rate*100:.1f}%")
        print(f"    平均速度:  {m.avg_speed_ms:.2f} m/s ({m.avg_speed_kmh:.1f} km/h)")
        print(f"    平均横向偏差: {m.avg_lateral_deviation:.3f} m")
        print(f"    每ep车道偏离: {m.avg_lane_invasions_per_ep:.1f} 次")
        print(f"    平均ep步数:  {m.avg_episode_steps:.0f}")
        print(f"    平均ep return: {m.avg_episode_return_v2:.1f}")

    print_single(metrics_a)
    if metrics_b is not None:
        print_single(metrics_b)

    print("\n" + "=" * 80)


def load_scenarios(scenario_dir: Optional[str], num_episodes: int) -> Optional[List[Dict]]:
    """加载场景列表，如果指定了目录则从中循环采样"""
    if scenario_dir is None:
        return None
    if not os.path.exists(scenario_dir):
        print(f"场景目录不存在: {scenario_dir}，使用默认直道")
        return None

    files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])
    if not files:
        return None

    configs = []
    for f in files:
        path = os.path.join(scenario_dir, f)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                configs.append(json.load(fh))
        except Exception as e:
            print(f"警告: 无法加载场景 {f}: {e}")

    if not configs:
        return None

    # 循环采样到需要的 episode 数
    repeated = []
    for i in range(num_episodes):
        repeated.append(configs[i % len(configs)])
    return repeated


def build_parser():
    p = argparse.ArgumentParser(
        description="A/B 对比评估 — 比较不同奖励函数版本的 checkpoint 表现"
    )

    # Checkpoint
    p.add_argument("--checkpoint", type=str, default=None,
                   help="单 checkpoint 评估")
    p.add_argument("--checkpoint-a", type=str, default=None,
                   help="A/B 对比: checkpoint A (如 v1 训练)")
    p.add_argument("--checkpoint-b", type=str, default=None,
                   help="A/B 对比: checkpoint B (如 v2 训练)")
    p.add_argument("--label-a", type=str, default="v1 (原始奖励)",
                   help="checkpoint A 的标签")
    p.add_argument("--label-b", type=str, default="v2 (优化奖励)",
                   help="checkpoint B 的标签")

    # 评估参数
    p.add_argument("--episodes", type=int, default=50, help="评估 episode 数 (默认50)")
    p.add_argument("--scenarios-dir", type=str,
                   default=None,
                   help="场景 JSON 目录（用于多样化评估，为空则使用默认直道）")
    p.add_argument("--use-random-scenarios", action="store_true",
                   help="使用 data/training_scenarios 目录的场景")
    p.add_argument("--stochastic", action="store_true",
                   help="使用随机策略（默认确定性）")

    # 环境参数
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town02")
    p.add_argument("--straight-road-length", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-json", type=str, default=None,
                   help="保存结果到 JSON 文件")

    return p


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 场景
    if args.use_random_scenarios and args.scenarios_dir is None:
        args.scenarios_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'training_scenarios'
        )
    scenario_configs = load_scenarios(args.scenarios_dir, args.episodes)

    env_kwargs = dict(
        host=args.host, port=args.port, town=args.town,
        num_agents=2, straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps,
    )

    deterministic = not args.stochastic

    # 判断运行模式
    if args.checkpoint_a and args.checkpoint_b:
        # A/B 对比模式
        print(f"\n{'='*60}")
        print(f"A/B 对比评估: {args.label_a} vs {args.label_b}")
        print(f"{'='*60}")

        # 使用相同的 scenario_configs 保证公平对比
        print(f"\n[1/2] 评估 A: {args.label_a}")
        print(f"  Checkpoint: {args.checkpoint_a}")
        metrics_a = evaluate_checkpoint(
            args.checkpoint_a, args.label_a,
            args.episodes, scenario_configs, env_kwargs, device, deterministic,
        )

        print(f"\n[2/2] 评估 B: {args.label_b}")
        print(f"  Checkpoint: {args.checkpoint_b}")
        metrics_b = evaluate_checkpoint(
            args.checkpoint_b, args.label_b,
            args.episodes, scenario_configs, env_kwargs, device, deterministic,
        )

        print_comparison(metrics_a, metrics_b)
        results = {"a": vars(metrics_a), "b": vars(metrics_b)}

    elif args.checkpoint:
        # 单 checkpoint 评估模式
        print(f"\n{'='*60}")
        print(f"单 Checkpoint 评估")
        print(f"{'='*60}")
        metrics = evaluate_checkpoint(
            args.checkpoint, "checkpoint",
            args.episodes, scenario_configs, env_kwargs, device, deterministic,
        )
        print_comparison(metrics)
        results = vars(metrics)

    else:
        print("请指定 --checkpoint (单评估) 或 --checkpoint-a + --checkpoint-b (A/B对比)")
        return

    # 保存 JSON
    if args.output_json:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return super().default(obj)

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, cls=NpEncoder, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
