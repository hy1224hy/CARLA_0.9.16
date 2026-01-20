import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from carla_rllib_wrapper import RLlibCarlaEnv
import torch

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="RLlib MAPPO Training for CARLA")
    parser.add_argument("--num-gpus", type=int, default=1 if torch.cuda.is_available() else 0, help="使用的GPU数量")
    parser.add_argument("--num-workers", type=int, default=1, help="并行采样的Worker数量 (设为0可以在主进程运行，方便调试)")
    parser.add_argument("--train-iter", type=int, default=10000, help="训练迭代次数")
    parser.add_argument("--render", action="store_true", help="是否在训练过程中开启可视化 (注意：这会显著拖慢训练，且需num-workers=0)")
    args = parser.parse_args()

    # 1. 初始化 Ray
    # ray.init(ignore_reinit_error=True)
    ray.init(local_mode=True, ignore_reinit_error=True)

    # 2. 注册自定义环境
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
                checkpoint_frequency=10,
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