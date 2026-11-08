"""
使用手动驾驶数据进行预训练（行为克隆）
在正式训练前运行此脚本，将手动驾驶数据用于初始化策略网络
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import os
import argparse
import torch
import numpy as np
from typing import Dict
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from environment.carla_rllib_wrapper import RLlibCarlaEnv

try:
    from utils.manual_data_loader import ManualDataLoader
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.manual_data_loader import ManualDataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def pretrain_policy(
    manual_data_dir: str,
    output_dir: str,
    pretrain_steps: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 1e-4
):
    """
    使用手动驾驶数据预训练策略网络
    
    Args:
        manual_data_dir: 手动驾驶数据目录
        output_dir: 预训练模型保存目录
        pretrain_steps: 预训练步数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print(f"\n开始预训练（使用手动驾驶数据）...")
    print(f"数据目录: {manual_data_dir}")
    print(f"预训练步数: {pretrain_steps}")
    print(f"批次大小: {batch_size}")
    
    # 初始化Ray
    ray.init(local_mode=True, ignore_reinit_error=True)
    
    # 注册环境
    register_env("carla_env", lambda config: RLlibCarlaEnv(config))
    
    # 创建环境配置（用于获取观察和动作空间）
    env_config = {
        "host": "127.0.0.1",
        "port": 2000,
        "town": "Town02",
        "num_agents": 2,
        "straight_road_length": 100.0,
        "max_steps": 1000,
        "render": False,
    }
    
    # 创建临时环境以获取空间信息
    temp_env = RLlibCarlaEnv(env_config)
    obs_space = temp_env.observation_space[0]
    action_space = temp_env.action_space[0]
    temp_env.close()
    
    # 创建PPO配置
    config = (
        PPOConfig()
        .environment(
            env="carla_env",
            env_config=env_config,
            disable_env_checking=True
        )
        .framework("torch")
        .training(
            lr=learning_rate,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            }
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )
    
    # 创建算法实例
    algorithm = config.build()
    
    # 加载手动驾驶数据
    loader = ManualDataLoader(manual_data_dir)
    stats = loader.get_statistics()
    print("\n手动驾驶数据统计:")
    for agent_id, stat in stats.items():
        print(f"  Agent {agent_id}: {stat['num_episodes']} episodes, {stat['total_steps']} steps")
    
    # 获取所有智能体的数据
    all_observations = []
    all_actions = []
    
    for agent_id in loader.get_all_agent_ids():
        data = loader.convert_to_rllib_format(agent_id)
        all_observations.append(data["observations"])
        all_actions.append(data["actions"])
    
    # 合并所有智能体的数据
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"\n总数据量: {len(observations)} 步")
    print(f"观察形状: {observations.shape}")
    print(f"动作形状: {actions.shape}")
    
    # 转换为torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取策略
    policy = algorithm.get_policy("shared_policy")
    policy.model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)
    
    num_batches = len(observations) // batch_size
    if num_batches == 0:
        print("警告: 数据量太少，无法进行预训练")
        ray.shutdown()
        return
    
    print(f"\n开始预训练，共 {num_batches} 个批次...")
    
    losses = []
    for step in range(pretrain_steps):
        # 随机采样一个批次
        indices = np.random.choice(len(observations), size=batch_size, replace=True)
        batch_obs = torch.FloatTensor(observations[indices]).to(device)
        batch_actions = torch.FloatTensor(actions[indices]).to(device)
        
        # 前向传播
        try:
            # 获取动作分布
            action_dist_inputs = policy.model.get_action_dist_inputs(batch_obs)
            action_dist_class = policy.dist_class
            dist = action_dist_class.from_logits(action_dist_inputs)
            
            # 计算损失（负对数似然）
            log_probs = dist.logp(batch_actions)
            loss = -log_probs.mean()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
            optimizer.step()
            
            losses.append(loss.item())
            
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"预训练步数: {step + 1}/{pretrain_steps}, 平均损失: {avg_loss:.4f}")
        except Exception as e:
            print(f"预训练步骤 {step + 1} 出错: {e}")
            continue
    
    print(f"\n预训练完成！最终平均损失: {np.mean(losses[-100:]):.4f}")
    
    # 保存预训练模型
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "pretrained_policy")
    algorithm.save(checkpoint_path)
    print(f"预训练模型已保存到: {checkpoint_path}")
    
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="使用手动驾驶数据预训练策略")
    parser.add_argument("--manual-data-dir", type=str, required=True,
                       help="手动驾驶数据目录")
    parser.add_argument("--output-dir", type=str, default="./pretrained_models",
                       help="预训练模型保存目录")
    parser.add_argument("--pretrain-steps", type=int, default=1000,
                       help="预训练步数")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="学习率")
    
    args = parser.parse_args()
    
    pretrain_policy(
        manual_data_dir=args.manual_data_dir,
        output_dir=args.output_dir,
        pretrain_steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
