"""
MAPPO经验缓冲区
用于存储和采样训练数据
"""

import numpy as np
from typing import Dict, List, Tuple
import torch


class MAPPOBuffer:
    """MAPPO经验缓冲区"""
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        buffer_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        初始化缓冲区
        
        Args:
            num_agents: 智能体数量
            obs_dim: 观察维度
            action_dim: 动作维度
            global_state_dim: 全局状态维度
            buffer_size: 缓冲区大小
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            device: 设备
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # 为每个智能体创建缓冲区
        self.obs_buffers = [np.zeros((buffer_size, obs_dim), dtype=np.float32) for _ in range(num_agents)]
        self.action_buffers = [np.zeros((buffer_size, action_dim), dtype=np.float32) for _ in range(num_agents)]
        self.reward_buffers = [np.zeros(buffer_size, dtype=np.float32) for _ in range(num_agents)]
        self.value_buffers = [np.zeros(buffer_size, dtype=np.float32) for _ in range(num_agents)]
        self.log_prob_buffers = [np.zeros(buffer_size, dtype=np.float32) for _ in range(num_agents)]
        self.done_buffers = [np.zeros(buffer_size, dtype=np.bool_) for _ in range(num_agents)]
        
        # 全局状态缓冲区
        self.global_state_buffer = np.zeros((buffer_size, global_state_dim), dtype=np.float32)
        
        # 指针
        self.ptr = 0
        self.size = 0
    
    def store(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, np.ndarray],
        rewards: Dict[int, float],
        values: Dict[int, float],
        log_probs: Dict[int, float],
        dones: Dict[int, bool],
        global_state: np.ndarray
    ):
        """
        存储经验
        
        Args:
            obs: 观察
            actions: 动作
            rewards: 奖励
            values: 价值估计
            log_probs: 对数概率
            dones: 是否完成
            global_state: 全局状态
        """
        for agent_id in range(self.num_agents):
            self.obs_buffers[agent_id][self.ptr] = obs[agent_id]
            self.action_buffers[agent_id][self.ptr] = actions[agent_id]
            self.reward_buffers[agent_id][self.ptr] = rewards[agent_id]
            self.value_buffers[agent_id][self.ptr] = values[agent_id]
            self.log_prob_buffers[agent_id][self.ptr] = log_probs[agent_id]
            self.done_buffers[agent_id][self.ptr] = dones[agent_id]
        
        self.global_state_buffer[self.ptr] = global_state
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(
        self,
        next_value: float,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE（Generalized Advantage Estimation）
        
        Args:
            next_value: 下一个状态的价值
            rewards: 奖励序列
            values: 价值序列
            dones: 完成标志序列
        
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
            next_value = values[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def finish_path(self, next_values: Dict[int, float]):
        """
        完成一个episode，计算GAE
        
        Args:
            next_values: 下一个状态的价值
        
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        all_advantages = {}
        all_returns = {}
        
        for agent_id in range(self.num_agents):
            if self.size == 0:
                all_advantages[agent_id] = np.array([], dtype=np.float32)
                all_returns[agent_id] = np.array([], dtype=np.float32)
                continue
                
            # 获取当前episode的数据
            episode_rewards = self.reward_buffers[agent_id][:self.size]
            episode_values = self.value_buffers[agent_id][:self.size]
            episode_dones = self.done_buffers[agent_id][:self.size]
            
            # 计算GAE
            next_value = next_values.get(agent_id, 0.0)
            advantages, returns = self.compute_gae(
                next_value, episode_rewards, episode_values, episode_dones
            )
            
            all_advantages[agent_id] = advantages
            all_returns[agent_id] = returns
        
        return all_advantages, all_returns
    
    def get_batch(self, agent_id: int, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """
        获取批次数据
        
        Args:
            agent_id: 智能体ID
            batch_size: 批次大小（如果为None，返回所有数据）
        
        Returns:
            batch: 批次数据字典
        """
        if batch_size is None:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'obs': torch.FloatTensor(self.obs_buffers[agent_id][indices]).to(self.device),
            'actions': torch.FloatTensor(self.action_buffers[agent_id][indices]).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_prob_buffers[agent_id][indices]).to(self.device),
            'global_states': torch.FloatTensor(self.global_state_buffer[indices]).to(self.device),
        }
        
        return batch
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

