import gymnasium as gym
import numpy as np
from datetime import datetime
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple, Any
import os


try:
    from environment.carla_multi_agent_env import CarlaMultiAgentEnv
    from visualization.mappo_visualizer import MAPPOVisualizer
    from utils.mappo_logger import MAPPOLogger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from environment.carla_multi_agent_env import CarlaMultiAgentEnv
    from visualization.mappo_visualizer import MAPPOVisualizer

class RLlibCarlaEnv(MultiAgentEnv):
    def __init__(self, env_config: Dict):
        super().__init__()
        
        self.env = CarlaMultiAgentEnv(
            host=env_config.get("host", "127.0.0.1"),
            port=env_config.get("port", 2000),
            town=env_config.get("town", "Town02"),
            num_agents=env_config.get("num_agents", 2),
            straight_road_length=env_config.get("straight_road_length", 100.0),
            timestep=env_config.get("timestep", 0.1),
            max_steps=env_config.get("max_steps", 1000)
        )
        
        self._agent_ids = {i for i in range(self.env.num_agents)}

        # 记录已经完成的智能体
        self._done_agents = set()

        # 定义空间 
        single_action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )

        self.action_space = gym.spaces.Dict({
            i: single_action_space for i in self._agent_ids
        })
        self.observation_space = gym.spaces.Dict({
            i: single_observation_space for i in self._agent_ids
        })

        # 可视化配置
        self.enable_render = env_config.get("render", False)
        self.visualizer = None
        self.episode_rewards = {i: 0.0 for i in self._agent_ids}
        self.episode_count = 0
        
        if self.enable_render:
            self._init_visualizer()

        self.enable_logging = env_config.get("enable_logging", True)
        self.logger = None
        if self.enable_logging:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs/mappo_logs'+'/'+datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.logger = MAPPOLogger(
                log_dir=log_dir, 
                comment=f"rllib_env_{os.getpid()}"
            )
            print(f"tensorboard启动，日志路径: {self.logger.log_dir}")

    def _init_visualizer(self):
        try:
            self.visualizer = MAPPOVisualizer(width=1200, height=800, fps=30)
            print("[RLlibWrapper] 可视化已启用")
        except Exception as e:
            print(f"[RLlibWrapper] 警告: 无法初始化可视化器: {e}")
            self.enable_render = False

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset()
        
        # 重置时清空死亡名单
        self._done_agents = set()
        
        self.episode_rewards = {i: 0.0 for i in self._agent_ids}
        self.episode_count += 1
        
        if self.enable_render and self.visualizer:
            self._render_frame(obs, {i: 0.0 for i in self._agent_ids})
            
        return obs, info

    def step(self, action_dict):
        # 1. 执行动作
        obs, rewards, dones, info = self.env.step(action_dict)
        
        if self.logger:
            # 这里的 action_dict 和 obs 结构正好符合 mappo_logger 的要求
            self.logger.log_step(
                rewards=rewards,
                observations=obs,
                actions=action_dict,
                values=None  # 环境中拿不到 Critic Value，传 None 即可
            )

        # 检查是否有任何车辆发生严重碰撞
        any_crash = False
        for r in rewards.values():
            if r < -50: 
                any_crash = True
                break
        
        # 3. 更新可视化奖励
        for agent_id, reward in rewards.items():
            if agent_id in self.episode_rewards:
                self.episode_rewards[agent_id] += reward
        
        if self.enable_render:
            self._render_frame(obs, rewards)

    
        should_terminate_all = any_crash or all(dones.values())

        filtered_obs = {}
        filtered_rewards = {}
        filtered_terminateds = {}
        filtered_truncateds = {}

        for agent_id in self._agent_ids:
            if agent_id in self._done_agents:
                continue
            
            if agent_id in obs:
                filtered_obs[agent_id] = obs[agent_id]
            if agent_id in rewards:
                filtered_rewards[agent_id] = rewards[agent_id]
            
            # 单个智能体的结束状态
            is_done = dones.get(agent_id, False)
            filtered_terminateds[agent_id] = is_done
            filtered_truncateds[agent_id] = False
            
            if is_done:
                self._done_agents.add(agent_id)

        filtered_terminateds["__all__"] = should_terminate_all
        filtered_truncateds["__all__"] = False
        
        if should_terminate_all:
             for aid in self._agent_ids:
                 filtered_terminateds[aid] = True

        return filtered_obs, filtered_rewards, filtered_terminateds, filtered_truncateds, info

    def _render_frame(self, obs, rewards):
        if not self.visualizer: return
        self.visualizer.handle_events()
        if not self.visualizer.running:
            self.enable_render = False
            self.visualizer.close()
            return

        vehicle_positions = {}
        target_positions = {}
        
        for agent_id in self._agent_ids:
            # 只有活着的车才画
            if agent_id < len(self.env.vehicles) and self.env.vehicles[agent_id].is_alive:
                trans = self.env.vehicles[agent_id].get_transform()
                vehicle_positions[agent_id] = (trans.location.x, trans.location.y)
                if agent_id < len(self.env.target_waypoints):
                    target_wp = self.env.target_waypoints[agent_id]
                    target_positions[agent_id] = (target_wp.transform.location.x, target_wp.transform.location.y)

        camera_images = self.env.get_camera_images()

        self.visualizer.render(
            observations=obs,
            rewards=rewards,
            episode_rewards=self.episode_rewards,
            vehicle_positions=vehicle_positions,
            target_positions=target_positions,
            episode=self.episode_count,
            step=self.env.current_step,
            max_episodes=0,
            max_steps=self.env.max_steps,
            camera_images=camera_images
        )


    def evaluate(self, policy, episodes: int = 5, max_steps: int = None, explore: bool = False) -> Dict[str, Any]:
        """使用当前包装环境执行评估，并返回统计指标。"""
        episode_returns = []
        episode_lengths = []
        per_agent_episode_returns = []

        rollout_max_steps = max_steps if max_steps is not None else self.env.max_steps

        for ep in range(1, episodes + 1):
            obs, _ = self.reset()
            done_all = False
            step_count = 0
            ep_return = {agent_id: 0.0 for agent_id in self._agent_ids}

            while not done_all and step_count < rollout_max_steps:
                action_dict = {}
                for agent_id, agent_obs in obs.items():
                    action = policy.compute_single_action(agent_obs, explore=explore)
                    action_dict[agent_id] = np.asarray(action, dtype=np.float32)

                next_obs, rewards, terminateds, truncateds, _ = self.step(action_dict)

                for agent_id, reward in rewards.items():
                    if agent_id in ep_return:
                        ep_return[agent_id] += float(reward)

                done_all = bool(terminateds.get("__all__", False) or truncateds.get("__all__", False))
                obs = next_obs
                step_count += 1

            ep_avg = float(np.mean(list(ep_return.values()))) if ep_return else 0.0
            episode_returns.append(ep_avg)
            episode_lengths.append(step_count)
            per_agent_episode_returns.append(ep_return)

            detail = ", ".join([f"agent_{aid}: {ret:.2f}" for aid, ret in ep_return.items()])
            print(f"[Episode {ep}/{episodes}] steps={step_count}, avg_return={ep_avg:.2f}, {detail}")

        per_agent_return_mean = {}
        for aid in self._agent_ids:
            vals = [ret.get(aid, 0.0) for ret in per_agent_episode_returns]
            per_agent_return_mean[aid] = float(np.mean(vals)) if vals else 0.0

        return {
            "episodes": episodes,
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "episode_return_std": float(np.std(episode_returns)) if episode_returns else 0.0,
            "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "per_agent_return_mean": per_agent_return_mean,
        }

    def close(self):
        if self.visualizer:
            self.visualizer.close()
        self.env.close()