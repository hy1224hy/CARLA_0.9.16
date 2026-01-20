"""
MAPPO模型评估脚本
用于测试训练好的模型
"""

import os
import sys
import argparse
import numpy as np
import torch

# 添加CARLA路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from carla_multi_agent_env import CarlaMultiAgentEnv
from mappo_networks import MAPPOAgent


def evaluate(
    model_dir: str,
    num_episodes: int = 10,
    max_steps: int = 1000,
    device: str = 'cpu',
    host: str = '127.0.0.1',
    port: int = 2000
):
    """
    评估训练好的模型
    
    Args:
        model_dir: 模型目录
        num_episodes: 评估episode数量
        max_steps: 每个episode最大步数
        device: 设备
        host: CARLA服务器地址
        port: CARLA服务器端口
    """
    print("=" * 50)
    print("MAPPO模型评估")
    print("=" * 50)
    
    # 创建环境
    print("正在初始化CARLA环境...")
    env = CarlaMultiAgentEnv(
        host=host,
        port=port,
        town='Town02',
        num_agents=2,
        straight_road_length=100.0,
        timestep=0.1,
        max_steps=max_steps
    )
    
    # 观察和动作维度
    obs_dim = 8
    action_dim = 3
    global_state_dim = 2 * 6
    
    # 加载智能体
    print("正在加载模型...")
    agents = {}
    for agent_id in range(env.num_agents):
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.startswith(f'agent_{agent_id}_') and f.endswith('.pth')]
        if not model_files:
            print(f"警告: 未找到智能体 {agent_id} 的模型文件")
            continue
        
        # 使用final模型，如果没有则使用最新的
        final_model = f'agent_{agent_id}_final.pth'
        if final_model in model_files:
            model_path = os.path.join(model_dir, final_model)
        else:
            # 按episode编号排序，使用最新的
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
            model_path = os.path.join(model_dir, model_files[-1])
        
        agents[agent_id] = MAPPOAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            hidden_dim=64,
            device=device
        )
        agents[agent_id].load(model_path)
        print(f"智能体 {agent_id} 模型加载完成: {model_path}")
    
    if not agents:
        print("错误: 没有可用的模型")
        return
    
    # 评估统计
    episode_rewards = {i: [] for i in agents.keys()}
    episode_lengths = []
    success_count = 0
    
    print("\n开始评估...")
    print("-" * 50)
    
    try:
        for episode in range(num_episodes):
            # 重置环境
            obs, info = env.reset()
            
            episode_reward = {i: 0.0 for i in agents.keys()}
            episode_length = 0
            success = False
            
            # Episode循环
            for step in range(max_steps):
                # 选择动作（使用确定性策略）
                actions = {}
                for agent_id in agents.keys():
                    obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action = agents[agent_id].get_action(obs_tensor, deterministic=True)
                    actions[agent_id] = action
                
                # 执行动作
                next_obs, rewards, dones, info = env.step(actions)
                
                # 更新奖励
                for agent_id in agents.keys():
                    episode_reward[agent_id] += rewards[agent_id]
                
                episode_length += 1
                
                # 检查是否完成
                if any(dones.values()):
                    # 检查是否成功到达目标
                    for agent_id in agents.keys():
                        if not env.vehicles[agent_id].is_alive:
                            continue
                        transform = env.vehicles[agent_id].get_transform()
                        target_wp = env.target_waypoints[agent_id]
                        distance = transform.location.distance(target_wp.transform.location)
                        if distance < 5.0:
                            success = True
                    break
                
                obs = next_obs
            
            # 记录统计信息
            for agent_id in agents.keys():
                episode_rewards[agent_id].append(episode_reward[agent_id])
            episode_lengths.append(episode_length)
            if success:
                success_count += 1
            
            # 打印episode结果
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            for agent_id in agents.keys():
                print(f"  智能体 {agent_id}: 奖励={episode_reward[agent_id]:.2f}")
            print(f"  长度: {episode_length}, 成功: {'是' if success else '否'}")
        
        # 打印总体统计
        print("\n" + "=" * 50)
        print("评估结果")
        print("=" * 50)
        for agent_id in agents.keys():
            avg_reward = np.mean(episode_rewards[agent_id])
            std_reward = np.std(episode_rewards[agent_id])
            print(f"智能体 {agent_id}:")
            print(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  最大奖励: {np.max(episode_rewards[agent_id]):.2f}")
            print(f"  最小奖励: {np.min(episode_rewards[agent_id]):.2f}")
        
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / num_episodes * 100
        print(f"\n总体统计:")
        print(f"  平均episode长度: {avg_length:.1f}")
        print(f"  成功率: {success_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    
    finally:
        # 关闭环境
        env.close()
        print("评估完成！")


def main():
    parser = argparse.ArgumentParser(description='MAPPO模型评估脚本')
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录')
    parser.add_argument('--num-episodes', type=int, default=10, help='评估episode数量')
    parser.add_argument('--max-steps', type=int, default=1000, help='每个episode最大步数')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='设备')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    evaluate(
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
        host=args.host,
        port=args.port
    )


if __name__ == '__main__':
    main()


