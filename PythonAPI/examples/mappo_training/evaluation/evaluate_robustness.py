"""
鲁棒性评估脚本
在多个测试场景下评估策略性能，输出 TensorBoard 日志和丰富的评估指标。

用法:
  python -m evaluation.evaluate_robustness --checkpoint <path.pt>
  python -m evaluation.evaluate_robustness --checkpoint <path.pt> --render --num-episodes-per-scenario 3
  python -m evaluation.evaluate_robustness --checkpoint <path.pt> --output-file results.json

查看 TensorBoard:
  tensorboard --logdir logs/eval_robustness/
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import json
import math
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import carla

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor


# ── 工具函数 ──

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake]."""
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5
    if throttle_raw >= brake_raw:
        throttle, brake = throttle_raw, 0.0
    else:
        throttle, brake = 0.0, brake_raw
    return np.array([throttle, steer, brake], dtype=np.float32)


def safe_tag(name: str) -> str:
    """将场景名/类别名转为 TensorBoard 安全标签."""
    return name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("/", "_")


# ── 评估指标数据类 ──

@dataclass
class EpisodeMetrics:
    """单个评估 episode 的详细指标"""
    episode_id: int = 0
    scenario_name: str = ""
    category: str = ""

    # 基础结果
    returns: Dict[int, float] = field(default_factory=dict)
    total_return: float = 0.0
    success: bool = False
    collision: bool = False
    timed_out: bool = False
    steps: int = 0

    # 速度
    avg_speed_ms: float = 0.0
    max_speed_ms: float = 0.0

    # 车道保持
    avg_lateral_deviation: float = 0.0
    max_lateral_deviation: float = 0.0
    lane_invasions: int = 0

    # 驾驶平顺性
    jerk_total: float = 0.0
    steer_rate_total: float = 0.0

    # 安全
    ttc_min: float = float('inf')       # 最小 TTC（越低越危险）
    lidar_min: float = 1.0              # 最小 LiDAR 距离（归一化值）

    # 效率
    completion_time_s: float = 0.0      # 到达目标所用时间（秒）

    # 奖励分解（按组件名累积）
    reward_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioSummary:
    """单个场景的汇总指标"""
    scenario_name: str = ""
    category: str = ""
    num_episodes: int = 0
    success_rate: float = 0.0
    collision_rate: float = 0.0
    timeout_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    avg_steps: float = 0.0
    avg_speed_kmh: float = 0.0
    avg_lateral_deviation: float = 0.0
    lane_invasions_per_ep: float = 0.0
    avg_jerk: float = 0.0
    avg_steer_rate: float = 0.0
    avg_ttc_min: float = 0.0
    avg_lidar_min: float = 0.0


# ── 单场景评估 ──

def evaluate_scenario(
    env: CarlaMultiAgentEnv,
    actors: Dict[int, Actor],
    scenario_config: Dict,
    device: torch.device,
    num_episodes: int = 5,
    timestep: float = 0.1,
) -> Tuple[List[EpisodeMetrics], ScenarioSummary]:
    """在指定场景下评估策略，返回逐 episode 的详细指标列表和汇总."""
    scenario_name = scenario_config.get('description', scenario_config.get('filename', 'unknown'))
    category = scenario_config.get('category', 'unknown')
    print(f"\n=== 评估场景: {scenario_name} ===")

    episodes: List[EpisodeMetrics] = []

    for ep in range(num_episodes):
        env.set_scenario(scenario_config)
        if len(env.vehicles) == 0:
            env._spawn_vehicles()

        obs_dict, info = env.reset()
        num_agents = env.num_agents

        m = EpisodeMetrics(
            episode_id=ep,
            scenario_name=scenario_name,
            category=category,
            returns={aid: 0.0 for aid in range(num_agents)},
        )

        all_speeds: List[float] = []
        all_lateral_devs: List[float] = []
        episode_step = 0
        collision_occurred = False
        success_achieved = False
        reward_components_accum: Dict[str, float] = {}

        for step in range(env.max_steps):
            episode_step += 1
            action_dict: Dict[int, np.ndarray] = {}

            with torch.no_grad():
                for aid, obs in obs_dict.items():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tanh = actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                    action_dict[aid] = action_tanh_to_env(action_tanh)

            next_obs, rewards, dones, info = env.step(action_dict)

            for aid, r in rewards.items():
                m.returns[aid] += float(r)

            # 每步采集车辆运动数据
            for aid in range(num_agents):
                if aid < len(env.vehicles) and env.vehicles[aid].is_alive:
                    vel = env.vehicles[aid].get_velocity()
                    spd = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    all_speeds.append(spd)
                    m.max_speed_ms = max(m.max_speed_ms, spd)

                    transform = env.vehicles[aid].get_transform()
                    wp = env.map.get_waypoint(transform.location)
                    if wp:
                        veh_loc = transform.location
                        wp_loc = wp.transform.location
                        rv = wp.transform.get_right_vector()
                        vec = carla.Vector3D(veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                        lat_dev = abs(vec.x * rv.x + vec.y * rv.y)
                        all_lateral_devs.append(lat_dev)
                        m.max_lateral_deviation = max(m.max_lateral_deviation, lat_dev)

            # 车道偏离
            for aid in range(num_agents):
                if env.lane_invasion_events.get(aid, False):
                    m.lane_invasions += 1

            # 碰撞
            if info.get('collisions'):
                if any(info['collisions'].values()):
                    collision_occurred = True

            # 提取奖励分解
            reward_details = info.get('reward_details', {})
            for aid, details in reward_details.items():
                for comp_name, comp_val in details.items():
                    key = str(comp_name)
                    reward_components_accum[key] = reward_components_accum.get(key, 0.0) + float(comp_val)

            obs_dict = next_obs
            if all(dones.values()):
                break

        # 汇总 episode 指标
        m.total_return = sum(m.returns.values())
        m.success = not collision_occurred and all(env.reached_target.values())
        m.collision = collision_occurred
        m.timed_out = not m.success and not m.collision
        m.steps = episode_step
        m.avg_speed_ms = float(np.mean(all_speeds)) if all_speeds else 0.0
        m.avg_lateral_deviation = float(np.mean(all_lateral_devs)) if all_lateral_devs else 0.0
        m.completion_time_s = episode_step * timestep

        # 从 reward_details 提取细分指标
        m.jerk_total = reward_components_accum.get('jerk', 0.0)
        m.steer_rate_total = reward_components_accum.get('steer_rate', 0.0)
        ttc_val = reward_components_accum.get('ttc', 0.0)
        m.ttc_min = ttc_val if ttc_val < 0 else float('inf')
        m.lidar_min = reward_components_accum.get('safety_lidar', 1.0)
        m.reward_components = reward_components_accum

        episodes.append(m)

        # 控制台输出
        status = "成功" if m.success else ("碰撞" if m.collision else "超时")
        print(f"  第 {ep} 轮: {status} | 回报 {m.total_return:.1f} | "
              f"步数 {m.steps} | 均速 {m.avg_speed_ms * 3.6:.1f} km/h | "
              f"横向偏差 {m.avg_lateral_deviation:.3f}m | 车道偏离 {m.lane_invasions}次")

    # 构建场景汇总
    n = len(episodes)
    successes = sum(1 for e in episodes if e.success)
    collisions = sum(1 for e in episodes if e.collision)
    timeouts = sum(1 for e in episodes if e.timed_out)

    returns_list = [e.total_return for e in episodes]
    speeds_list = [e.avg_speed_ms * 3.6 for e in episodes if e.avg_speed_ms > 0]
    ldevs_list = [e.avg_lateral_deviation for e in episodes]
    steps_list = [e.steps for e in episodes]
    jerks_list = [e.jerk_total for e in episodes]
    steer_rates_list = [e.steer_rate_total for e in episodes]
    ttc_list = [e.ttc_min for e in episodes if e.ttc_min != float('inf')]
    lidar_list = [e.lidar_min for e in episodes]

    summary = ScenarioSummary(
        scenario_name=scenario_name,
        category=category,
        num_episodes=n,
        success_rate=successes / n,
        collision_rate=collisions / n,
        timeout_rate=timeouts / n,
        avg_return=float(np.mean(returns_list)),
        std_return=float(np.std(returns_list)),
        avg_steps=float(np.mean(steps_list)),
        avg_speed_kmh=float(np.mean(speeds_list)) if speeds_list else 0.0,
        avg_lateral_deviation=float(np.mean(ldevs_list)),
        lane_invasions_per_ep=sum(e.lane_invasions for e in episodes) / n,
        avg_jerk=float(np.mean(jerks_list)),
        avg_steer_rate=float(np.mean(steer_rates_list)),
        avg_ttc_min=float(np.mean(ttc_list)) if ttc_list else 0.0,
        avg_lidar_min=float(np.mean(lidar_list)) if lidar_list else 1.0,
    )

    print(f"\n场景结果: 成功率 {summary.success_rate:.2%} | "
          f"平均回报 {summary.avg_return:.1f} | "
          f"均速 {summary.avg_speed_kmh:.1f} km/h | "
          f"横向偏差 {summary.avg_lateral_deviation:.3f}m")

    return episodes, summary


# ── 场景加载 ──

def load_test_scenarios(test_scenario_dir: str) -> List[Dict]:
    """加载所有测试场景"""
    scenarios = []
    if not os.path.exists(test_scenario_dir):
        print(f"测试场景目录不存在: {test_scenario_dir}")
        return scenarios
    for filename in sorted(os.listdir(test_scenario_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(test_scenario_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    scenario_config = json.load(f)
                scenario_config['filename'] = filename
                scenarios.append(scenario_config)
            except Exception as e:
                print(f"加载场景失败 {filename}: {e}")
    return scenarios


def generate_boundary_scenarios() -> List[Dict]:
    """生成边界测试场景"""
    scenarios = []
    for offset in [0.0, 1.0, 2.0]:
        scenarios.append({
            "description": f"边界: 极小间距 (offset={offset}m)",
            "agent0": {"x": -7.45, "y": 176.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 188.0 + offset, "z": 0.5, "yaw": 90.0},
        })
    for distance in [50.0, 60.0, 70.0]:
        scenarios.append({
            "description": f"边界: 极大间距 ({distance}m)",
            "agent0": {"x": -7.45, "y": 160.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 160.0 + distance, "z": 0.5, "yaw": 90.0},
        })
    for lane_offset in [3.0, 4.0, -3.0, -4.0]:
        scenarios.append({
            "description": f"边界: 横向偏移 ({lane_offset}m)",
            "agent0": {"x": -7.45 + lane_offset, "y": 176.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 210.0, "z": 0.5, "yaw": 90.0},
        })
    for angle in [-10.0, -5.0, 5.0, 10.0]:
        scenarios.append({
            "description": f"边界: 初始角度偏差 ({angle}°)",
            "agent0": {"x": -7.45, "y": 176.0, "z": 0.5, "yaw": 90.0 + angle},
            "agent1": {"x": -7.45, "y": 210.0, "z": 0.5, "yaw": 90.0},
        })
    return scenarios


# ── TensorBoard 日志 ──

def log_scenarios_to_tb(writer: SummaryWriter, summaries: List[ScenarioSummary], global_step: int):
    """将场景摘要写入 TensorBoard."""
    for s in summaries:
        tag = safe_tag(s.scenario_name)
        writer.add_scalar(f"scenario/{tag}/success_rate", s.success_rate, global_step)
        writer.add_scalar(f"scenario/{tag}/collision_rate", s.collision_rate, global_step)
        writer.add_scalar(f"scenario/{tag}/avg_return", s.avg_return, global_step)
        writer.add_scalar(f"scenario/{tag}/avg_speed_kmh", s.avg_speed_kmh, global_step)
        writer.add_scalar(f"scenario/{tag}/avg_lateral_deviation", s.avg_lateral_deviation, global_step)
        writer.add_scalar(f"scenario/{tag}/avg_steps", s.avg_steps, global_step)
        writer.add_scalar(f"scenario/{tag}/lane_invasions_per_ep", s.lane_invasions_per_ep, global_step)


def log_category_to_tb(writer: SummaryWriter, category_summaries: Dict[str, List[ScenarioSummary]], global_step: int):
    """将分类汇总写入 TensorBoard."""
    for cat, summaries in category_summaries.items():
        tag = safe_tag(cat)
        n_total = sum(s.num_episodes for s in summaries)
        successes = sum(s.success_rate * s.num_episodes for s in summaries)
        collisions = sum(s.collision_rate * s.num_episodes for s in summaries)
        timeouts = sum(s.timeout_rate * s.num_episodes for s in summaries)

        writer.add_scalar(f"category/{tag}/success_rate", successes / n_total if n_total else 0, global_step)
        writer.add_scalar(f"category/{tag}/collision_rate", collisions / n_total if n_total else 0, global_step)
        writer.add_scalar(f"category/{tag}/timeout_rate", timeouts / n_total if n_total else 0, global_step)
        writer.add_scalar(f"category/{tag}/avg_return",
                          float(np.mean([s.avg_return for s in summaries])), global_step)
        writer.add_scalar(f"category/{tag}/avg_speed_kmh",
                          float(np.mean([s.avg_speed_kmh for s in summaries])), global_step)
        writer.add_scalar(f"category/{tag}/avg_lateral_deviation",
                          float(np.mean([s.avg_lateral_deviation for s in summaries])), global_step)
        writer.add_scalar(f"category/{tag}/lane_invasions_per_ep",
                          float(np.mean([s.lane_invasions_per_ep for s in summaries])), global_step)


def log_agent_to_tb(writer: SummaryWriter, all_episodes: List[EpisodeMetrics], num_agents: int, global_step: int):
    """将分 agent 的指标写入 TensorBoard."""
    for aid in range(num_agents):
        returns = [e.returns.get(aid, 0.0) for e in all_episodes]
        successes = sum(1 for e in all_episodes if e.success)
        n = len(all_episodes)
        speeds = [e.avg_speed_ms * 3.6 for e in all_episodes if e.avg_speed_ms > 0]
        ldevs = [e.avg_lateral_deviation for e in all_episodes]

        writer.add_scalar(f"agent/{aid}/avg_return", float(np.mean(returns)) if returns else 0, global_step)
        writer.add_scalar(f"agent/{aid}/success_rate", successes / n if n else 0, global_step)
        writer.add_scalar(f"agent/{aid}/avg_speed_kmh", float(np.mean(speeds)) if speeds else 0, global_step)
        writer.add_scalar(f"agent/{aid}/avg_lateral_deviation", float(np.mean(ldevs)) if ldevs else 0, global_step)


def log_reward_breakdown_to_tb(writer: SummaryWriter, all_episodes: List[EpisodeMetrics], global_step: int):
    """将奖励分解写入 TensorBoard."""
    comp_accum: Dict[str, List[float]] = {}
    for ep in all_episodes:
        for comp, val in ep.reward_components.items():
            comp_accum.setdefault(comp, []).append(val)
    for comp, vals in comp_accum.items():
        tag = safe_tag(comp)
        writer.add_scalar(f"reward_breakdown/{tag}", float(np.mean(vals)), global_step)


def log_safety_to_tb(writer: SummaryWriter, all_episodes: List[EpisodeMetrics], global_step: int):
    """将安全指标写入 TensorBoard."""
    ttc_vals = [e.ttc_min for e in all_episodes if e.ttc_min != float('inf')]
    lidar_vals = [e.lidar_min for e in all_episodes]
    collisions = sum(1 for e in all_episodes if e.collision)
    n = len(all_episodes)

    writer.add_scalar("safety/avg_ttc_min", float(np.mean(ttc_vals)) if ttc_vals else 0, global_step)
    writer.add_scalar("safety/avg_lidar_min", float(np.mean(lidar_vals)) if lidar_vals else 1.0, global_step)
    writer.add_scalar("safety/collision_rate", collisions / n if n else 0, global_step)


def log_quality_to_tb(writer: SummaryWriter, all_episodes: List[EpisodeMetrics], global_step: int):
    """将驾驶质量指标写入 TensorBoard."""
    jerk_vals = [e.jerk_total for e in all_episodes]
    steer_rate_vals = [e.steer_rate_total for e in all_episodes]
    max_speeds = [e.max_speed_ms * 3.6 for e in all_episodes if e.max_speed_ms > 0]

    writer.add_scalar("quality/avg_jerk", float(np.mean(jerk_vals)) if jerk_vals else 0, global_step)
    writer.add_scalar("quality/avg_steer_rate", float(np.mean(steer_rate_vals)) if steer_rate_vals else 0, global_step)
    writer.add_scalar("quality/avg_max_speed_kmh", float(np.mean(max_speeds)) if max_speeds else 0, global_step)


def log_histograms_to_tb(writer: SummaryWriter, all_episodes: List[EpisodeMetrics], global_step: int):
    """将分布直方图写入 TensorBoard."""
    returns = np.array([e.total_return for e in all_episodes])
    steps = np.array([e.steps for e in all_episodes])
    speeds = np.array([e.avg_speed_ms * 3.6 for e in all_episodes if e.avg_speed_ms > 0])

    writer.add_histogram("distribution/episode_return", returns, global_step)
    writer.add_histogram("distribution/episode_steps", steps, global_step)
    writer.add_histogram("distribution/avg_speed_kmh", speeds, global_step)


# ── 主评估函数 ──

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"使用设备: {device}")

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        print(f"随机种子: {args.seed}")

    # 加载模型
    if not os.path.exists(args.checkpoint):
        print(f"checkpoint不存在: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cooperative_ids: List[int] = []
    adversarial_ids: List[int] = []
    actors: Dict[int, Actor] = {}

    if "adversarial_actors" in checkpoint:
        cooperative_ids = checkpoint.get("cooperative_ids", [0, 1])
        adversarial_ids = checkpoint.get("adversarial_ids",
                                         list(checkpoint["adversarial_actors"].keys()))
        total_agents = len(cooperative_ids) + len(adversarial_ids)
        print(f"检测到多对抗checkpoint: {len(cooperative_ids)} 合作 + {len(adversarial_ids)} 对抗")

        if "cooperative_actors" in checkpoint:
            for aid in cooperative_ids:
                actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                actor.load_state_dict(checkpoint["cooperative_actors"][aid])
                actor.eval()
                actors[aid] = actor
        else:
            for aid in cooperative_ids:
                if aid in checkpoint.get("actors", {}):
                    actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                    actor.load_state_dict(checkpoint["actors"][aid])
                    actor.eval()
                    actors[aid] = actor

        for aid in adversarial_ids:
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
            actor.load_state_dict(checkpoint["adversarial_actors"][aid])
            actor.eval()
            actors[aid] = actor

    elif "adversarial_actor" in checkpoint:
        cooperative_ids = [0, 1]
        adversarial_ids = [2]
        total_agents = 3
        print("检测到旧格式单对抗checkpoint")

        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(checkpoint["adversarial_actor"])
        actor.eval()
        actors[2] = actor

        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
            for aid in cooperative_ids:
                if aid in pretrained_ckpt.get("actors", {}):
                    coop_actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                    coop_actor.load_state_dict(pretrained_ckpt["actors"][aid])
                    coop_actor.eval()
                    actors[aid] = coop_actor
    else:
        total_agents = checkpoint.get("args", {}).get("num_agents", 2)
        agent_ids = list(range(total_agents))
        print(f"检测到标准MAPPO checkpoint: {total_agents} agents")

        for aid in agent_ids:
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
            if aid in checkpoint.get('actors', {}):
                actor.load_state_dict(checkpoint['actors'][aid])
            else:
                print(f"警告: agent {aid} 未找到模型参数")
            actor.eval()
            actors[aid] = actor

    if not actors:
        print("错误: 未加载任何模型")
        return

    # 初始化环境
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=total_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
        spawn_vehicles=False,
        adversarial_agent_ids=adversarial_ids,
    )

    print(f"模型已加载: {args.checkpoint}")
    print(f"合作智能体: {cooperative_ids if cooperative_ids else list(actors.keys())}")
    if adversarial_ids:
        print(f"对抗智能体: {adversarial_ids}")

    # 初始化 TensorBoard
    writer = None
    if not args.no_tensorboard:
        tb_log_dir = os.path.join(args.tb_log_dir,
                                   datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard 日志目录: {tb_log_dir}")

    # 收集测试场景
    all_scenarios: List[Dict] = []

    if args.include_training_scenarios:
        training_scenario_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
        training_scenarios = load_test_scenarios(training_scenario_dir)
        for s in training_scenarios:
            s['category'] = 'training'
        all_scenarios.extend(training_scenarios)
        print(f"已加载 {len(training_scenarios)} 个训练场景")

    if args.test_scenario_dir and os.path.exists(args.test_scenario_dir):
        test_scenarios = load_test_scenarios(args.test_scenario_dir)
        for s in test_scenarios:
            s['category'] = 'test'
        all_scenarios.extend(test_scenarios)
        print(f"已加载 {len(test_scenarios)} 个测试场景")

    if args.include_boundary_scenarios:
        boundary_scenarios = generate_boundary_scenarios()
        for s in boundary_scenarios:
            s['category'] = 'boundary'
        all_scenarios.extend(boundary_scenarios)
        print(f"已生成 {len(boundary_scenarios)} 个边界场景")

    if not all_scenarios:
        print("未找到任何测试场景")
        if writer:
            writer.close()
        return

    # 开始评估
    all_summaries: List[ScenarioSummary] = []
    all_episodes: List[EpisodeMetrics] = []
    category_summaries: Dict[str, List[ScenarioSummary]] = {}

    for idx, scenario_config in enumerate(all_scenarios):
        episodes, summary = evaluate_scenario(
            env=env,
            actors=actors,
            scenario_config=scenario_config,
            device=device,
            num_episodes=args.num_episodes_per_scenario,
            timestep=args.timestep,
        )

        all_summaries.append(summary)
        all_episodes.extend(episodes)
        category_summaries.setdefault(summary.category, []).append(summary)

        # 每个场景评估完就写入 TensorBoard（用场景索引作为 step）
        if writer:
            log_scenarios_to_tb(writer, [summary], idx)

    # 全局汇总日志（step=0，方便横向对比）
    global_step = 0
    if writer:
        log_category_to_tb(writer, category_summaries, global_step)
        log_agent_to_tb(writer, all_episodes, total_agents, global_step)
        log_reward_breakdown_to_tb(writer, all_episodes, global_step)
        log_safety_to_tb(writer, all_episodes, global_step)
        log_quality_to_tb(writer, all_episodes, global_step)
        log_histograms_to_tb(writer, all_episodes, global_step)

    # ── 控制台最终汇总 ──
    print("\n" + "=" * 70)
    print("总体评估总结")
    print("=" * 70)

    for category, summaries in category_summaries.items():
        n_total = sum(s.num_episodes for s in summaries)
        successes = sum(s.success_rate * s.num_episodes for s in summaries)
        collisions = sum(s.collision_rate * s.num_episodes for s in summaries)
        timeouts = sum(s.timeout_rate * s.num_episodes for s in summaries)

        returns_all = [s.avg_return for s in summaries]
        speeds_all = [s.avg_speed_kmh for s in summaries]
        ldevs_all = [s.avg_lateral_deviation for s in summaries]
        lanes_all = [s.lane_invasions_per_ep for s in summaries]

        print(f"\n[{category}]")
        print(f"  场景数: {len(summaries)}, 总测试数: {n_total}")
        print(f"  成功率: {successes / n_total:.2%}  "
              f"碰撞率: {collisions / n_total:.2%}  "
              f"超时率: {timeouts / n_total:.2%}")
        print(f"  平均回报: {np.mean(returns_all):.1f} ± {np.std(returns_all):.1f}")
        print(f"  均速: {np.mean(speeds_all):.1f} km/h  "
              f"横向偏差: {np.mean(ldevs_all):.3f}m  "
              f"车道偏离/ep: {np.mean(lanes_all):.2f}")

    # 全部分 agent 汇总
    print(f"\n[分 Agent 指标]")
    for aid in range(total_agents):
        returns = [e.returns.get(aid, 0.0) for e in all_episodes]
        successes = sum(1 for e in all_episodes if e.success)
        n = len(all_episodes)
        print(f"  Agent {aid}: 平均回报 {np.mean(returns):.1f} ± {np.std(returns):.1f}, "
              f"成功率 {successes / n:.2%}")

    # 奖励分解汇总
    if all_episodes and all_episodes[0].reward_components:
        print(f"\n[奖励分解]")
        comp_accum: Dict[str, List[float]] = {}
        for ep in all_episodes:
            for comp, val in ep.reward_components.items():
                comp_accum.setdefault(comp, []).append(val)
        for comp, vals in sorted(comp_accum.items()):
            print(f"  {comp}: {np.mean(vals):.2f}")

    # 保存结果
    if args.output_file:
        output_data = {
            'checkpoint': args.checkpoint,
            'scenario_summaries': [
                {k: v for k, v in vars(s).items()} for s in all_summaries
            ],
            'global_metrics': {
                'total_episodes': len(all_episodes),
                'num_scenarios': len(all_summaries),
                'success_rate': sum(1 for e in all_episodes if e.success) / len(all_episodes),
                'collision_rate': sum(1 for e in all_episodes if e.collision) / len(all_episodes),
                'timeout_rate': sum(1 for e in all_episodes if e.timed_out) / len(all_episodes),
                'avg_return': float(np.mean([e.total_return for e in all_episodes])),
                'avg_speed_kmh': float(np.mean([e.avg_speed_ms * 3.6 for e in all_episodes if e.avg_speed_ms > 0])),
                'avg_lateral_deviation': float(np.mean([e.avg_lateral_deviation for e in all_episodes])),
                'avg_lane_invasions_per_ep': float(np.mean([e.lane_invasions for e in all_episodes])),
            },
        }
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output_file}")

    if writer:
        writer.close()
    env.close()


# ── 命令行参数 ──

def build_parser():
    parser = argparse.ArgumentParser(description="鲁棒性评估脚本（TensorBoard + 丰富指标）")

    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--pretrained-ckpt", type=str, default="",
                        help="预训练合作团队checkpoint（旧格式对抗评估时使用）")
    parser.add_argument("--output-file", type=str, default="", help="结果保存路径（JSON）")

    # 环境参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 评估参数
    parser.add_argument("--num-episodes-per-scenario", type=int, default=5,
                        help="每个场景测试的episode数")
    parser.add_argument("--test-scenario-dir", type=str, default="", help="测试场景目录")
    parser.add_argument("--include-training-scenarios", action="store_true", default=True,
                        help="包含训练场景")
    parser.add_argument("--include-boundary-scenarios", action="store_true", default=True,
                        help="包含边界场景")

    # TensorBoard
    parser.add_argument("--tb-log-dir", type=str, default="./logs/eval_robustness",
                        help="TensorBoard 日志目录 (default: ./logs/eval_robustness)")
    parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard 日志")

    # 其他
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)
