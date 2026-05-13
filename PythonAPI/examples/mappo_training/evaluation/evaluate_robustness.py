"""
鲁棒性评估脚本
在多个测试场景下评估策略性能
测试场景包括：
1. 训练见过的场景
2. 全新的未见过的场景
3. 边界情况（极近/极远等）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import json
from typing import Dict, List, Tuple
import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
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


def evaluate_scenario(
    env: CarlaMultiAgentEnv,
    actors: Dict[int, Actor],
    scenario_config: Dict,
    device: torch.device,
    num_episodes: int = 5,
    render: bool = False,
) -> Dict:
    """在指定场景下评估策略"""
    results = {
        'total_episodes': num_episodes,
        'successful_episodes': 0,
        'collision_episodes': 0,
        'timeout_episodes': 0,
        'rewards': [],
        'steps': [],
    }

    scenario_name = scenario_config.get('description', 'unknown')
    print(f"\n=== 评估场景: {scenario_name} ===")

    for ep in range(num_episodes):
        env.set_scenario(scenario_config)

        # 重置环境
        if len(env.vehicles) == 0:
            env._spawn_vehicles()

        obs_dict, info = env.reset()
        episode_reward = 0.0
        episode_step = 0
        collided = False

        for step in range(env.max_steps):
            episode_step += 1
            action_dict: Dict[int, np.ndarray] = {}

            for aid, obs in obs_dict.items():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_tanh = actors[aid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                action_env = action_tanh_to_env(action_tanh)
                action_dict[aid] = action_env

            next_obs, rewards, dones, info = env.step(action_dict)

            for aid in rewards:
                episode_reward += rewards[aid]

            if info.get('collisions'):
                if any(info['collisions'].values()):
                    collided = True

            obs_dict = next_obs

            if all(dones.values()):
                break

        # 记录结果
        success = not collided and all(env.reached_target.values())
        results['rewards'].append(episode_reward)
        results['steps'].append(episode_step)

        if success:
            results['successful_episodes'] += 1
            print(f"  第 {ep} 轮: 成功! 回报 {episode_reward:.1f}, 步数 {episode_step}")
        elif collided:
            results['collision_episodes'] += 1
            print(f"  第 {ep} 轮: 碰撞! 回报 {episode_reward:.1f}")
        else:
            results['timeout_episodes'] += 1
            print(f"  第 {ep} 轮: 超时/未完成! 回报 {episode_reward:.1f}")

    success_rate = results['successful_episodes'] / num_episodes
    avg_reward = float(np.mean(results['rewards'])) if results['rewards'] else 0.0
    print(f"\n场景结果: 成功率 {success_rate:.2%}, 平均回报 {avg_reward:.1f}")

    return results


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

    # 极小间距场景
    for offset in [0.0, 1.0, 2.0]:
        scenarios.append({
            "description": f"边界: 极小间距 (offset={offset}m)",
            "agent0": {"x": -7.45, "y": 176.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 188.0 + offset, "z": 0.5, "yaw": 90.0},
        })

    # 极大间距场景
    for distance in [50.0, 60.0, 70.0]:
        scenarios.append({
            "description": f"边界: 极大间距 ({distance}m)",
            "agent0": {"x": -7.45, "y": 160.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 160.0 + distance, "z": 0.5, "yaw": 90.0},
        })

    # 极限偏移场景
    for lane_offset in [3.0, 4.0, -3.0, -4.0]:
        scenarios.append({
            "description": f"边界: 横向偏移 ({lane_offset}m)",
            "agent0": {"x": -7.45 + lane_offset, "y": 176.0, "z": 0.5, "yaw": 90.0},
            "agent1": {"x": -7.45, "y": 210.0, "z": 0.5, "yaw": 90.0},
        })

    # 角度偏差场景
    for angle in [-10.0, -5.0, 5.0, 10.0]:
        scenarios.append({
            "description": f"边界: 初始角度偏差 ({angle}°)",
            "agent0": {"x": -7.45, "y": 176.0, "z": 0.5, "yaw": 90.0 + angle},
            "agent1": {"x": -7.45, "y": 210.0, "z": 0.5, "yaw": 90.0},
        })

    return scenarios


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    if not os.path.exists(args.checkpoint):
        print(f"checkpoint不存在: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 检测checkpoint类型
    cooperative_ids: List[int] = []
    adversarial_ids: List[int] = []
    actors: Dict[int, Actor] = {}

    if "adversarial_actors" in checkpoint:
        # 多对抗MAPPO checkpoint
        cooperative_ids = checkpoint.get("cooperative_ids", [0, 1])
        adversarial_ids = checkpoint.get("adversarial_ids",
                                         list(checkpoint["adversarial_actors"].keys()))
        total_agents = len(cooperative_ids) + len(adversarial_ids)
        print(f"检测到多对抗checkpoint: {len(cooperative_ids)} 合作 + {len(adversarial_ids)} 对抗")

        # 加载合作团队
        if "cooperative_actors" in checkpoint:
            for aid in cooperative_ids:
                actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                actor.load_state_dict(checkpoint["cooperative_actors"][aid])
                actor.eval()
                actors[aid] = actor
        else:
            # 兼容：从 pretrained_ckpt 加载的格式
            for aid in cooperative_ids:
                if aid in checkpoint.get("actors", {}):
                    actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                    actor.load_state_dict(checkpoint["actors"][aid])
                    actor.eval()
                    actors[aid] = actor

        # 加载对抗团队
        for aid in adversarial_ids:
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
            actor.load_state_dict(checkpoint["adversarial_actors"][aid])
            actor.eval()
            actors[aid] = actor

    elif "adversarial_actor" in checkpoint:
        # 旧格式单对抗PPO checkpoint
        cooperative_ids = [0, 1]
        adversarial_ids = [2]
        total_agents = 3
        print("检测到旧格式单对抗checkpoint")

        # 对抗智能体
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(checkpoint["adversarial_actor"])
        actor.eval()
        actors[2] = actor

        # 合作智能体需要另外加载
        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            pretrained_ckpt = torch.load(args.pretrained_ckpt, map_location=device)
            for aid in cooperative_ids:
                if aid in pretrained_ckpt.get("actors", {}):
                    coop_actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
                    coop_actor.load_state_dict(pretrained_ckpt["actors"][aid])
                    coop_actor.eval()
                    actors[aid] = coop_actor
    else:
        # 标准MAPPO checkpoint（无对抗）
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

    # 收集测试场景
    all_scenarios = []

    # 1. 加载训练场景（如果可用）
    if args.include_training_scenarios:
        training_scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
        training_scenarios = load_test_scenarios(training_scenario_dir)
        for s in training_scenarios:
            s['category'] = 'training'
        all_scenarios.extend(training_scenarios)
        print(f"已加载 {len(training_scenarios)} 个训练场景")

    # 2. 加载测试场景（如果有）
    if args.test_scenario_dir and os.path.exists(args.test_scenario_dir):
        test_scenarios = load_test_scenarios(args.test_scenario_dir)
        for s in test_scenarios:
            s['category'] = 'test'
        all_scenarios.extend(test_scenarios)
        print(f"已加载 {len(test_scenarios)} 个测试场景")

    # 3. 生成边界场景
    if args.include_boundary_scenarios:
        boundary_scenarios = generate_boundary_scenarios()
        for s in boundary_scenarios:
            s['category'] = 'boundary'
        all_scenarios.extend(boundary_scenarios)
        print(f"已生成 {len(boundary_scenarios)} 个边界场景")

    if not all_scenarios:
        print("未找到任何测试场景")
        return

    # 开始评估
    all_results = {}
    category_summary = {}

    for scenario_config in all_scenarios:
        scenario_name = scenario_config.get('description', scenario_config.get('filename', 'unknown'))
        category = scenario_config.get('category', 'unknown')

        results = evaluate_scenario(
            env=env,
            actors=actors,
            scenario_config=scenario_config,
            device=device,
            num_episodes=args.num_episodes_per_scenario,
            render=args.render,
        )

        all_results[scenario_name] = results
        results['category'] = category

        if category not in category_summary:
            category_summary[category] = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'collision_episodes': 0,
                'timeout_episodes': 0,
                'rewards': [],
            }

        category_summary[category]['total_episodes'] += results['total_episodes']
        category_summary[category]['successful_episodes'] += results['successful_episodes']
        category_summary[category]['collision_episodes'] += results['collision_episodes']
        category_summary[category]['timeout_episodes'] += results['timeout_episodes']
        category_summary[category]['rewards'].extend(results['rewards'])

    # 打印总结
    print("\n" + "=" * 60)
    print("总体评估总结")
    print("=" * 60)

    for category, summary in category_summary.items():
        total = summary['total_episodes']
        success = summary['successful_episodes']
        collision = summary['collision_episodes']
        timeout = summary['timeout_episodes']
        avg_reward = float(np.mean(summary['rewards'])) if summary['rewards'] else 0.0

        print(f"\n[{category}]")
        print(f"  总测试数: {total}")
        print(f"  成功率: {success / total:.2%}")
        print(f"  碰撞率: {collision / total:.2%}")
        print(f"  超时率: {timeout / total:.2%}")
        print(f"  平均回报: {avg_reward:.1f}")

    # 保存结果
    if args.output_file:
        output_data = {
            'checkpoint': args.checkpoint,
            'category_summary': category_summary,
            'scenario_results': all_results,
        }
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output_file}")

    env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="鲁棒性评估脚本")

    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--pretrained-ckpt", type=str, default="", help="预训练合作团队checkpoint（旧格式对抗评估时使用）")
    parser.add_argument("--output-file", type=str, default="", help="结果保存路径（JSON）")

    # 环境参数
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 评估参数
    parser.add_argument("--num-episodes-per-scenario", type=int, default=5, help="每个场景测试的episode数")
    parser.add_argument("--test-scenario-dir", type=str, default="", help="测试场景目录")
    parser.add_argument("--include-training-scenarios", action="store_true", default=True, help="包含训练场景")
    parser.add_argument("--include-boundary-scenarios", action="store_true", default=True, help="包含边界场景")

    # 其他
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)

