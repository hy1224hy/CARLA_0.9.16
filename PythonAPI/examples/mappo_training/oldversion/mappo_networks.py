"""
MAPPO算法的神经网络定义
包括Actor网络和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ActorNetwork(nn.Module):
    """Actor网络：输出动作分布"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        activation: str = 'tanh'
    ):
        """
        初始化Actor网络
        
        Args:
            obs_dim: 观察维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            activation: 激活函数
        """
        super(ActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = torch.tanh
        
        # 共享特征提取层
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层：均值
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        
        # 输出层：标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观察 [batch_size, obs_dim]
        
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        
        mean = torch.tanh(self.mean_head(x))  # 限制在[-1, 1]
        
        # 标准差（确保为正）
        std = torch.exp(self.log_std.clamp(-20, 2))
        std = std.expand_as(mean)
        
        return mean, std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        采样动作
        
        Args:
            obs: 观察
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作
        """
        mean, std = self.forward(obs)
        
        if deterministic:
            return mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            # 限制动作范围
            action = torch.clamp(action, -1.0, 1.0)
            return action
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率和熵
        
        Args:
            obs: 观察
            actions: 动作
        
        Returns:
            log_probs: 对数概率
            entropy: 熵
        """
        mean, std = self.forward(obs)
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic网络：评估状态值（集中式训练）"""
    
    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 64,
        activation: str = 'tanh'
    ):
        """
        初始化Critic网络
        
        Args:
            global_state_dim: 全局状态维度（所有智能体的状态拼接）
            hidden_dim: 隐藏层维度
            activation: 激活函数
        """
        super(CriticNetwork, self).__init__()
        
        self.global_state_dim = global_state_dim
        
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = torch.tanh
        
        # 网络结构
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            global_state: 全局状态 [batch_size, global_state_dim]
        
        Returns:
            value: 状态值 [batch_size, 1]
        """
        x = self.activation(self.fc1(global_state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.value_head(x)
        
        return value


class MAPPOAgent:
    """MAPPO智能体（包含Actor和Critic）"""
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: str = 'cpu'
    ):
        """
        初始化MAPPO智能体
        
        Args:
            agent_id: 智能体ID
            obs_dim: 观察维度
            action_dim: 动作维度
            global_state_dim: 全局状态维度
            hidden_dim: 隐藏层维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            device: 设备（'cpu'或'cuda'）
        """
        self.agent_id = agent_id
        self.device = device
        
        # 创建网络
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(global_state_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 旧网络（用于计算重要性采样比率）
        self.old_actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
    
    def update_old_actor(self):
        """更新旧Actor网络"""
        self.old_actor.load_state_dict(self.actor.state_dict())
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        获取动作
        
        Args:
            obs: 观察
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(obs_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        return self.actor.evaluate_actions(obs, actions)
    
    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """获取状态值"""
        return self.critic(global_state)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.old_actor.load_state_dict(self.actor.state_dict())


