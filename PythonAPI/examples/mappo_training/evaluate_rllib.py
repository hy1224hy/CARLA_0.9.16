import os
import argparse
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from carla_rllib_wrapper import RLlibCarlaEnv
import time

def evaluate(checkpoint_path, num_episodes=5):
    """
    加载模型并评估
    """
    # 1. 初始化 Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 2. 注册环境 (加载模型时需要知道环境名称对应的类)
    register_env("carla_env", lambda config: RLlibCarlaEnv(config))

    print(f"正在加载模型: {checkpoint_path} ...")
    
    # 3. 从 Checkpoint 恢复算法
    # 注意：这会自动加载训练时的配置
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保 Checkpoint 路径正确 (指向包含 params.pkl 的文件夹)")
        return

    # 4. 创建评估专用的环境实例
    # 我们手动创建一个环境实例，以便完全控制 step 循环和渲染
    env_config = {
        "host": "127.0.0.1",
        "port": 2000,
        "town": "Town02",
        "num_agents": 2,
        "render": True,  # 强制开启渲染
        "max_steps": 1000
    }
    
    env = RLlibCarlaEnv(env_config)
    
    print("\n开始评估...")
    print("按 ESC 键退出可视化窗口")
    print("-" * 50)

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            # RLlib 的 reset 返回的是 (obs, info)
            
            terminateds = {"__all__": False}
            truncateds = {"__all__": False}
            episode_reward = {i: 0.0 for i in range(env.env.num_agents)}
            step_count = 0
            
            while not terminateds["__all__"] and not truncateds["__all__"]:
                # 计算动作
                actions = {}
                for agent_id, agent_obs in obs.items():
                    # 使用 compute_single_action 获取动作
                    # explore=False 表示使用确定性策略 (ArgMax)，如果你想看随机表现设为 True
                    actions[agent_id] = algo.compute_single_action(
                        observation=agent_obs,
                        policy_id="shared_policy", # 必须指定策略ID
                        explore=False 
                    )
                
                # 环境步进
                obs, rewards, terminateds, truncateds, info = env.step(actions)
                
                # 统计奖励
                for agent_id, r in rewards.items():
                    episode_reward[agent_id] += r
                
                step_count += 1
                
                # 稍微加一点延时，防止回放太快看不清 (如果FPS很高的话)
                # time.sleep(0.02) 

            print(f"Episode {episode + 1}/{num_episodes} 完成:")
            print(f"  步数: {step_count}")
            print(f"  总奖励: {episode_reward}")
            
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"\n评估出错: {e}")
    finally:
        env.close()
        ray.shutdown()
        print("评估结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLlib MAPPO Evaluation for CARLA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 文件夹路径 (例如: .../checkpoint_000100)")
    parser.add_argument("--episodes", type=int, default=5, help="评估的 episode 数量")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 路径不存在 {args.checkpoint}")
    else:
        evaluate(args.checkpoint, args.episodes)