"""
MAPPO训练主脚本
在Town02直道上训练两个自动驾驶车辆智能体
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from typing import Dict
import datetime



# 添加CARLA路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入本地模块
from carla_multi_agent_env import CarlaMultiAgentEnv
from mappo_networks import MAPPOAgent
from mappo_buffer import MAPPOBuffer
from mappo_trainer import MAPPOTrainer
from mappo_visualizer import MAPPOVisualizer
from mappo_logger import MAPPOLogger


def train_mappo(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    update_frequency: int = 100,
    save_frequency: int = 100,
    save_dir: str = './mappo_models',
    device: str = 'cpu',
    host: str = '127.0.0.1',
    port: int = 2000,
    visualize: bool = True,
    log_dir: str = './logs/mappo_training/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
):
    """
    训练MAPPO算法
    
    Args:
        num_episodes: 训练episode数量
        max_steps_per_episode: 每个episode最大步数
        update_frequency: 更新频率（每N步更新一次）
        save_frequency: 保存频率（每N个episode保存一次）
        save_dir: 模型保存目录
        device: 设备（'cpu'或'cuda'）
        host: CARLA服务器地址
        port: CARLA服务器端口
        visualize: 是否启用可视化
        log_dir: TensorBoard日志保存目录
    """
    print("=" * 50)
    print("MAPPO训练开始")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    print("正在初始化CARLA环境...")
    env = CarlaMultiAgentEnv(
        host=host,
        port=port,
        town='Town02',
        num_agents=2,
        straight_road_length=100.0,
        timestep=0.1,
        max_steps=max_steps_per_episode
    )
    
    # 观察和动作维度
    obs_dim = 8  # 根据环境定义
    action_dim = 3  # [throttle, steer, brake]
    global_state_dim = 2 * 6  # 2个智能体，每个6维特征
    
    # 创建智能体
    print("正在创建MAPPO智能体...")
    agents = {}
    for agent_id in range(env.num_agents):
        agents[agent_id] = MAPPOAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            hidden_dim=64,
            lr_actor=3e-4,
            lr_critic=3e-4,
            device=device
        )
        print(f"智能体 {agent_id} 创建完成")
    
    # 创建经验缓冲区
    buffer = MAPPOBuffer(
        num_agents=env.num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
        buffer_size=update_frequency,
        gamma=0.99,
        gae_lambda=0.95,
        device=device
    )
    
    # 创建训练器
    trainer = MAPPOTrainer(
        agents=agents,
        buffer=buffer,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        device=device
    )
    
    # 创建可视化器（如果启用）
    visualizer = None
    if visualize:
        try:
            visualizer = MAPPOVisualizer(width=1200, height=800, fps=30)
            print("可视化已启用")
        except Exception as e:
            print(f"警告: 无法初始化可视化器: {e}")
            print("将继续训练，但不显示可视化")
            visualize = False
    
    # 创建TensorBoard日志记录器
    logger = MAPPOLogger(log_dir=log_dir, comment=f'_episodes{num_episodes}_agents{env.num_agents}')
    print(f"TensorBoard日志记录已启用，日志目录: {log_dir}")
    print(f"查看日志: tensorboard --logdir {log_dir}")
    
    # 记录超参数
    hyperparameters = {
        'num_episodes': num_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'update_frequency': update_frequency,
        'num_agents': env.num_agents,
        'obs_dim': 8,
        'action_dim': 3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_param': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 10,
        'batch_size': 64,
        'device': device
    }
    logger.log_hyperparameters(hyperparameters)
    
    # 训练统计
    episode_rewards = {i: [] for i in range(env.num_agents)}
    episode_lengths = []
    
    print("\n开始训练...")
    print("-" * 50)
    
    try:
        for episode in range(num_episodes):
            # 重置环境
            obs, info = env.reset()
            
            episode_reward = {i: 0.0 for i in range(env.num_agents)}
            episode_length = 0
            
            # 存储初始状态的价值
            global_state = env.get_global_state()
            next_values = {}
            for agent_id in range(env.num_agents):
                obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    # 使用全局状态计算价值
                    value = agents[agent_id].get_value(global_state_tensor)
                    next_values[agent_id] = value.item()
            
            # Episode循环
            for step in range(max_steps_per_episode):
                # 选择动作
                actions = {}
                log_probs_dict = {}
                values_dict = {}
                
                for agent_id in range(env.num_agents):
                    obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # 获取动作
                        action_tensor = agents[agent_id].actor.get_action(obs_tensor, deterministic=False)
                        action = action_tensor.cpu().numpy()[0]
                        
                        # 计算对数概率和价值
                        mean, std = agents[agent_id].actor(obs_tensor)
                        dist = torch.distributions.Normal(mean, std)
                        log_prob = dist.log_prob(action_tensor).sum()
                        
                        # 使用全局状态计算价值
                        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                        value = agents[agent_id].get_value(global_state_tensor)
                    
                    actions[agent_id] = action
                    log_probs_dict[agent_id] = log_prob.item()
                    values_dict[agent_id] = value.item()
                
                # 执行动作
                next_obs, rewards, dones, info = env.step(actions)
                
                # 更新奖励
                for agent_id in range(env.num_agents):
                    episode_reward[agent_id] += rewards[agent_id]
                
                episode_length += 1
                
                # 记录TensorBoard指标（每步）
                logger.log_step(
                    rewards=rewards,
                    observations=obs,
                    actions=actions,
                    values=values_dict
                )
                
                # 可视化（如果启用）
                if visualize and visualizer is not None:
                    # 处理事件
                    visualizer.handle_events()
                    if not visualizer.running:
                        print("可视化窗口已关闭，继续训练...")
                        visualize = False
                        visualizer.close()
                        visualizer = None
                    else:
                        # 获取车辆位置
                        vehicle_positions = {}
                        target_positions = {}
                        for agent_id in range(env.num_agents):
                            if env.vehicles[agent_id].is_alive:
                                transform = env.vehicles[agent_id].get_transform()
                                vehicle_positions[agent_id] = (
                                    transform.location.x,
                                    transform.location.y
                                )
                                target_wp = env.target_waypoints[agent_id]
                                target_positions[agent_id] = (
                                    target_wp.transform.location.x,
                                    target_wp.transform.location.y
                                )
                        
                        # 获取摄像头图像
                        camera_images = env.get_camera_images()
                        
                        # 渲染
                        visualizer.render(
                            observations=obs,
                            rewards=rewards,
                            episode_rewards=episode_reward,
                            vehicle_positions=vehicle_positions,
                            target_positions=target_positions,
                            episode=episode,
                            step=step,
                            max_episodes=num_episodes,
                            max_steps=max_steps_per_episode,
                            camera_images=camera_images
                        )
                
                # 存储经验
                buffer.store(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    values=values_dict,
                    log_probs=log_probs_dict,
                    dones=dones,
                    global_state=global_state
                )
                
                # 更新观察和全局状态
                obs = next_obs
                global_state = env.get_global_state()
                
                # 更新next_values
                for agent_id in range(env.num_agents):
                    if not dones[agent_id]:
                        obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            value = agents[agent_id].get_value(global_state_tensor)
                            next_values[agent_id] = value.item()
                    else:
                        next_values[agent_id] = 0.0
                
                # 定期更新或在episode结束时更新
                if buffer.size >= update_frequency or any(dones.values()):
                    if buffer.size > 0:
                        # 完成路径并计算GAE
                        advantages, returns = buffer.finish_path(next_values)
                        
                        # 检查是否有有效数据
                        if len(advantages) > 0 and all(len(advantages[aid]) > 0 for aid in advantages):
                            # 更新策略
                            stats = trainer.update(advantages, returns)
                            
                            # 记录损失到TensorBoard
                            for agent_id in range(env.num_agents):
                                if agent_id in stats:
                                    logger.log_scalar(
                                        f'Agent_{agent_id}/Loss/Actor_Loss_Step',
                                        stats[agent_id]['actor_loss'],
                                        logger.step_count
                                    )
                                    logger.log_scalar(
                                        f'Agent_{agent_id}/Loss/Critic_Loss_Step',
                                        stats[agent_id]['critic_loss'],
                                        logger.step_count
                                    )
                                    logger.log_scalar(
                                        f'Agent_{agent_id}/Loss/Entropy_Step',
                                        stats[agent_id]['entropy'],
                                        logger.step_count
                                    )
                            
                            # 打印统计信息
                            if episode % 10 == 0:
                                print(f"\nEpisode {episode}, Step {step}")
                                for agent_id in range(env.num_agents):
                                    if agent_id in stats:
                                        print(f"  智能体 {agent_id}: "
                                              f"奖励={episode_reward[agent_id]:.2f}, "
                                              f"Actor损失={stats[agent_id]['actor_loss']:.4f}, "
                                              f"Critic损失={stats[agent_id]['critic_loss']:.4f}")
                        
                        # 清空缓冲区
                        buffer.clear()
                
                # 检查是否完成
                if any(dones.values()):
                    break
            
            # 记录统计信息
            for agent_id in range(env.num_agents):
                episode_rewards[agent_id].append(episode_reward[agent_id])
            episode_lengths.append(episode_length)
            
            # 检查是否成功（到达目标）
            success = {}
            for agent_id in range(env.num_agents):
                if env.vehicles[agent_id].is_alive:
                    transform = env.vehicles[agent_id].get_transform()
                    target_wp = env.target_waypoints[agent_id]
                    distance = transform.location.distance(target_wp.transform.location)
                    success[agent_id] = distance < 5.0
                else:
                    success[agent_id] = False
            
            # 记录episode级别的指标到TensorBoard
            # 注意：损失会在策略更新时记录，这里只记录episode级别的奖励和长度
            logger.log_episode(
                episode_rewards=episode_reward,
                episode_length=episode_length,
                losses=None,  # 损失在策略更新时已记录
                success=success
            )
            
            # 定期刷新日志
            if (episode + 1) % 10 == 0:
                logger.flush()
            
            # 打印episode统计
            if (episode + 1) % 10 == 0:
                avg_rewards = {i: np.mean(episode_rewards[i][-10:]) for i in range(env.num_agents)}
                avg_length = np.mean(episode_lengths[-10:])
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  平均奖励: {avg_rewards}")
                print(f"  平均长度: {avg_length:.1f}")
                print(f"  成功率: {sum(success.values())/len(success)*100:.1f}%")
            
            # 保存模型
            if (episode + 1) % save_frequency == 0:
                for agent_id in range(env.num_agents):
                    model_path = os.path.join(save_dir, f'agent_{agent_id}_episode_{episode + 1}.pth')
                    agents[agent_id].save(model_path)
                print(f"模型已保存到 {save_dir}")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    finally:
        # 保存最终模型
        print("\n保存最终模型...")
        for agent_id in range(env.num_agents):
            final_model_path = os.path.join(save_dir, f'agent_{agent_id}_final.pth')
            agents[agent_id].save(final_model_path)
        
        # 打印最终统计
        print("\n训练统计:")
        stats = logger.get_summary_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # 关闭日志记录器
        logger.close()
        print(f"\nTensorBoard日志已保存到: {log_dir}")
        print(f"查看日志: tensorboard --logdir {log_dir}")
        
        # 关闭可视化器
        if visualizer is not None:
            visualizer.close()
        
        # 关闭环境
        env.close()
        print("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='MAPPO训练脚本')
    parser.add_argument('--num-episodes', type=int, default=1000, help='训练episode数量')
    parser.add_argument('--max-steps', type=int, default=1000, help='每个episode最大步数')
    parser.add_argument('--update-frequency', type=int, default=100, help='更新频率')
    parser.add_argument('--save-frequency', type=int, default=100, help='保存频率')
    parser.add_argument('--save-dir', type=str, default='./mappo_models', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='设备')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--no-visualize', action='store_true', help='禁用可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/mappo_training/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), help='TensorBoard日志目录')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    train_mappo(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        update_frequency=args.update_frequency,
        save_frequency=args.save_frequency,
        save_dir=args.save_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        visualize=not args.no_visualize,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()

