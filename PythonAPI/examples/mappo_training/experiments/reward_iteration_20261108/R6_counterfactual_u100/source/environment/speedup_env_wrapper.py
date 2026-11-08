"""
提速环境包装器
功能：在正常智能体已有策略基础上增加油门，使其更激进
用途：评估对抗智能体在更激进场景下的效果
"""

import numpy as np
from typing import Dict, Optional, Callable
import carla


class SpeedupActionWrapper:
    """
    动作包装器：对指定智能体的动作进行提速处理
    """

    def __init__(
        self,
        speedup_factor: float = 1.5,
        min_throttle: float = 0.3,
        speedup_agent_ids: Optional[list] = None,
    ):
        """
        初始化动作包装器

        Args:
            speedup_factor: 油门放大系数（>1表示提速）
            min_throttle: 最小油门值（确保车辆不会太慢）
            speedup_agent_ids: 需要提速的智能体ID列表，None表示所有正常智能体(0,1)
        """
        self.speedup_factor = speedup_factor
        self.min_throttle = min_throttle
        self.speedup_agent_ids = speedup_agent_ids if speedup_agent_ids is not None else [0, 1]

    def process_action(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        """
        处理单个智能体的动作

        Args:
            agent_id: 智能体ID
            action: 原始动作 [throttle, steer, brake]

        Returns:
            处理后的动作
        """
        if agent_id not in self.speedup_agent_ids:
            return action.copy()

        processed = action.copy()

        # 增加油门
        original_throttle = processed[0]
        # 放大油门，并确保不小于最小值
        processed[0] = max(
            self.min_throttle,
            min(1.0, original_throttle * self.speedup_factor)
        )

        # 减少刹车（如果有刹车的话）
        if processed[2] > 0:
            processed[2] = processed[2] * 0.5  # 刹车减半

        return processed

    def process_action_dict(self, action_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        处理整个动作字典

        Args:
            action_dict: 原始动作字典 {agent_id: action}

        Returns:
            处理后的动作字典
        """
        return {
            aid: self.process_action(aid, act)
            for aid, act in action_dict.items()
        }


class SpeedupEnvWrapper:
    """
    环境包装器：在CARLA多智能体环境基础上，对正常智能体进行提速
    """

    def __init__(
        self,
        env,
        speedup_factor: float = 3,
        min_throttle: float = 0.3,
        speedup_agent_ids: Optional[list] = None,
    ):
        """
        初始化提速环境包装器

        Args:
            env: 原始CARLA多智能体环境
            speedup_factor: 油门放大系数
            min_throttle: 最小油门值
            speedup_agent_ids: 需要提速的智能体ID列表
        """
        self.env = env
        self.action_wrapper = SpeedupActionWrapper(
            speedup_factor=speedup_factor,
            min_throttle=min_throttle,
            speedup_agent_ids=speedup_agent_ids,
        )

        # 代理环境的属性
        self.num_agents = env.num_agents
        self.vehicles = env.vehicles
        self.reached_target = env.reached_target
        self.collision_events = env.collision_events
        self.crosswalk_invasion_events = env.crosswalk_invasion_events

    def reset(self, *args, **kwargs):
        """重置环境"""
        result = self.env.reset(*args, **kwargs)
        # 更新引用
        self.vehicles = self.env.vehicles
        self.reached_target = self.env.reached_target
        self.collision_events = self.env.collision_events
        self.crosswalk_invasion_events = self.env.crosswalk_invasion_events
        return result

    def step(self, action_dict: Dict[int, np.ndarray]):
        """
        执行一步（先处理动作再执行）

        Args:
            action_dict: 原始动作字典

        Returns:
            (obs, rewards, dones, info)
        """
        # 处理动作：对正常智能体提速
        processed_action_dict = self.action_wrapper.process_action_dict(action_dict)

        # 执行处理后的动作
        obs, rewards, dones, info = self.env.step(processed_action_dict)

        # 更新引用
        self.vehicles = self.env.vehicles
        self.reached_target = self.env.reached_target
        self.collision_events = self.env.collision_events
        self.crosswalk_invasion_events = self.env.crosswalk_invasion_events

        # 记录原始动作和处理后的动作（用于调试）
        info['original_actions'] = action_dict
        info['processed_actions'] = processed_action_dict

        return obs, rewards, dones, info

    def close(self):
        """关闭环境"""
        return self.env.close()

    def __getattr__(self, name):
        """代理其他属性到原始环境"""
        return getattr(self.env, name)


class AdaptiveSpeedupWrapper(SpeedupActionWrapper):
    """
    自适应提速包装器：根据车辆速度动态调整提速程度
    速度低时多提速，速度高时少提速
    """

    def __init__(
        self,
        speedup_agent_ids: Optional[list] = None,
        target_speed: float = 15.0,  # 目标速度 m/s
        max_speedup: float = 2.0,
        min_speedup: float = 1.0,
        min_throttle: float = 0.2,
    ):
        super().__init__(
            speedup_factor=1.0,  # 基础系数会动态调整
            min_throttle=min_throttle,
            speedup_agent_ids=speedup_agent_ids,
        )
        self.target_speed = target_speed
        self.max_speedup = max_speedup
        self.min_speedup = min_speedup
        self.current_speeds = {}  # 记录当前速度

    def update_speeds(self, vehicles):
        """更新车辆速度信息"""
        for i, vehicle in enumerate(vehicles):
            if vehicle.is_alive:
                vel = vehicle.get_velocity()
                speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                self.current_speeds[i] = speed

    def process_action(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        if agent_id not in self.speedup_agent_ids:
            return action.copy()

        # 根据当前速度动态调整提速系数
        current_speed = self.current_speeds.get(agent_id, 0.0)

        # 计算自适应提速系数
        if current_speed < self.target_speed * 0.5:
            # 速度远低于目标，大幅提速
            speedup = self.max_speedup
        elif current_speed < self.target_speed:
            # 速度接近目标，线性减少提速
            progress = (current_speed - self.target_speed * 0.5) / (self.target_speed * 0.5)
            speedup = self.max_speedup - progress * (self.max_speedup - self.min_speedup)
        else:
            # 达到或超过目标速度，不再提速
            speedup = self.min_speedup

        processed = action.copy()

        # 应用自适应提速
        original_throttle = processed[0]
        processed[0] = max(
            self.min_throttle,
            min(1.0, original_throttle * speedup)
        )

        # 高速时减少刹车
        if processed[2] > 0 and current_speed > self.target_speed * 0.8:
            processed[2] = processed[2] * 0.3

        return processed


def create_speedup_env(
    env,
    mode: str = "fixed",
    **kwargs
):
    """
    工厂函数：创建提速环境

    Args:
        env: 原始环境
        mode: "fixed" (固定提速) 或 "adaptive" (自适应提速)
        **kwargs: 传递给包装器的参数

    Returns:
        包装后的环境
    """
    if mode == "fixed":
        return SpeedupEnvWrapper(env, **kwargs)
    elif mode == "adaptive":
        # 自适应模式需要特殊处理
        wrapper = AdaptiveSpeedupWrapper(**kwargs)
        env_wrapper = SpeedupEnvWrapper(env, speedup_factor=1.0)
        env_wrapper.action_wrapper = wrapper
        return env_wrapper
    else:
        raise ValueError(f"Unknown mode: {mode}")
