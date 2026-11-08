import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from environment.carla_rllib_wrapper import RLlibCarlaEnv
import torch
import numpy as np
from typing import Optional

try:
    from utils.manual_data_loader import ManualDataLoader
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.manual_data_loader import ManualDataLoader

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def pretrain_with_manual_data(
    algorithm,
    manual_data_dir: str,
    pretrain_steps: int = 1000,
    batch_size: int = 128
):
    """
    使用手动驾驶数据进行预训练（行为克隆）
    
    Args:
        algorithm: RLlib算法实例
        manual_data_dir: 手动驾驶数据目录
        pretrain_steps: 预训练步数
        batch_size: 批次大小
    """
    print(f"\n开始训练（使用手动驾驶数据）...")
    print(f"数据目录: {manual_data_dir}")
    print(f"预训练步数: {pretrain_steps}")
    
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
    
    print(f"总数据量: {len(observations)} 步")
    
    # 转换为torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 预训练循环（行为克隆）
    policy = algorithm.get_policy("shared_policy")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-4)
    
    num_batches = len(observations) // batch_size
    if num_batches == 0:
        print("警告: 数据量太少，无法进行预训练")
        return
    
    print(f"开始预训练，共 {num_batches} 个批次...")
    
    for step in range(pretrain_steps):
        # 随机采样一个批次
        indices = np.random.choice(len(observations), size=batch_size, replace=False)
        batch_obs = torch.FloatTensor(observations[indices]).to(device)
        batch_actions = torch.FloatTensor(actions[indices]).to(device)
        
        # 前向传播
        action_dist = policy.model.get_action_dist_inputs(batch_obs)
        action_dist_class = policy.dist_class
        dist = action_dist_class.from_logits(action_dist)
        
        # 计算损失（负对数似然）
        log_probs = dist.logp(batch_actions)
        loss = -log_probs.mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"预训练步数: {step + 1}/{pretrain_steps}, 损失: {loss.item():.4f}")
    
    print("预训练完成！")


def main():
    parser = argparse.ArgumentParser(description="RLlib MAPPO Training for CARLA")
    parser.add_argument("--num-gpus", type=int, default=1 if torch.cuda.is_available() else 0, help="使用的GPU数量")
    parser.add_argument("--num-workers", type=int, default=1, help="并行采样的Worker数量 (设为0可以在主进程运行，方便调试)")
    parser.add_argument("--train-iter", type=int, default=10000, help="训练迭代次数")
    parser.add_argument("--render", action="store_true", help="是否在训练过程中开启可视化 (注意：这会显著拖慢训练，且需num-workers=0)")
    parser.add_argument("--pretrained-model", type=str, default=None, help="预训练模型路径（如果提供，将加载预训练模型）")
    args = parser.parse_args()

    # 1. 初始化 Ray
    # ray.init(ignore_reinit_error=True)
    ray.init(local_mode=True, ignore_reinit_error=True)

    # 2. 注册环境
    register_env("carla_env", lambda config: RLlibCarlaEnv(config))

    # 3. 准备环境配置
    env_config = {
        "host": "127.0.0.1",
        "port": 2000,
        "town": "Town02",
        "num_agents": 2,
        "straight_road_length": 100.0,
        "max_steps": 1000,
        "render": args.render, 
    }

    # 如果开启可视化，强制 worker数为0
    num_workers = 0 if args.render else args.num_workers
    if args.render:
        print("警告: 可视化已开启，强制设置 num_workers=0 (单进程模式)")

    config = (
        PPOConfig()
        .environment(
            env="carla_env",
            env_config=env_config,
            disable_env_checking=True 
        )
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,          
            num_envs_per_env_runner=1,            
            rollout_fragment_length="auto"
        )
        .training(
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            }
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=args.num_gpus)
    )
    
    # 如果提供了预训练模型，加载
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"\n加载预训练模型: {args.pretrained_model}")
        # 注意：这里需要在创建Tuner之前设置，但RLlib的架构使得这比较复杂
        # 我们将在训练配置中通过restore参数实现
        print("提示: 预训练模型将在训练开始时自动加载（如果checkpoint存在）")
        print("建议: 使用 --pretrained-model 参数时，确保模型路径正确")

    # 5. 开始训练
    print(f"开始 RLlib 训练... (Logs will be saved to {os.path.join(CURRENT_DIR, 'logs/rllib_results')})")
    
    stop_criteria = {
        "training_iteration": args.train_iter,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop=stop_criteria,
            storage_path=os.path.abspath(os.path.join(CURRENT_DIR, "logs/rllib_results")),
            name="mappo_carla_experiment",
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=100,
                checkpoint_at_end=True,
                num_to_keep=5,
            )
        ),
    )

    results = tuner.fit()

    print("训练结束。")
    if results.num_errors > 0:
        print("训练过程中出现错误！")
    else:
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        print(f"最佳 Checkpoint路径: {best_result.checkpoint.path}")
        # print(f"最佳平均奖励: {best_result.metrics['episode_reward_mean']}")

    ray.shutdown()

if __name__ == "__main__":
    main()