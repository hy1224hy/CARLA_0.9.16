"""
MAPPO训练器
实现CTDE架构的训练逻辑
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

try:
    from mappo_networks import MAPPOAgent
    from mappo_buffer import MAPPOBuffer
except ImportError:
    from mappo_training.mappo_networks import MAPPOAgent
    from mappo_training.mappo_buffer import MAPPOBuffer


class MAPPOTrainer:
    """MAPPO训练器"""
    
    def __init__(
        self,
        agents: Dict[int, MAPPOAgent],
        buffer: MAPPOBuffer,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0001,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            agents: 智能体字典
            buffer: 经验缓冲区
            clip_param: PPO裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数
            max_grad_norm: 梯度裁剪阈值
            ppo_epochs: PPO更新轮数
            batch_size: 批次大小
            device: 设备
        """
        self.agents = agents
        self.buffer = buffer
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
    
    def update(
        self,
        advantages: Dict[int, np.ndarray],
        returns: Dict[int, np.ndarray]
    ) -> Dict[int, Dict[str, float]]:
        """
        更新所有智能体
        
        Args:
            advantages: 优势函数
            returns: 回报
        
        Returns:
            stats: 训练统计信息
        """
        # 将advantages和returns添加到缓冲区
        for agent_id in range(self.buffer.num_agents):
            if agent_id in advantages:
                # 标准化优势函数
                adv = advantages[agent_id]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                # 存储到缓冲区（这里简化处理，实际应该扩展缓冲区）
                # 为了简化，我们在训练时直接使用这些值
        
        stats = {}
        
        # 更新每个智能体
        for agent_id, agent in self.agents.items():
            agent_stats = self._update_agent(
                agent_id, agent, advantages.get(agent_id), returns.get(agent_id)
            )
            stats[agent_id] = agent_stats
        
        return stats
    
    def _update_agent(
        self,
        agent_id: int,
        agent: MAPPOAgent,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        更新单个智能体
        
        Args:
            agent_id: 智能体ID
            agent: 智能体
            advantages: 优势函数
            returns: 回报
        
        Returns:
            stats: 统计信息
        """
        # 更新旧Actor
        agent.update_old_actor()
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 获取批次数据
        batch = self.buffer.get_batch(agent_id, batch_size=None)
        
        # 获取所有数据
        obs = batch['obs']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        global_states = batch['global_states']
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        # PPO更新多轮
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(obs))
            
            # 分批训练
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_global_states = global_states[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 计算新的对数概率和熵
                new_log_probs, entropy = agent.evaluate_actions(batch_obs, batch_actions)
                
                # 计算重要性采样比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                values = agent.get_value(batch_global_states)
                critic_loss = F.mse_loss(values, batch_returns.view(-1, 1))
                
                # 总损失
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy.mean()
                
                # 更新Actor
                agent.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
                agent.actor_optimizer.step()
                
                # 更新Critic
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.max_grad_norm)
                agent.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # 计算平均损失
        num_updates = self.ppo_epochs * (len(obs) // self.batch_size + 1)
        
        stats = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
        
        return stats

