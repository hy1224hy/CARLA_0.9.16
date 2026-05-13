"""
统一的 Actor-Critic 模型定义
用于 MAPPO (CTDE) 和 单Agent PPO 训练
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """2层MLP (128, Tanh) → mu_head + log_std, 用于 tanh-squashed Gaussian 策略"""

    def __init__(self, obs_dim: int = 19, act_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
        return action_tanh, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action_tanh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        clipped = torch.clamp(action_tanh, -0.999, 0.999)
        z = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - clipped.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """用于评估/推理：返回确定性动作 mu → tanh"""
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class CentralizedCritic(nn.Module):
    """集中式 Critic：输入全局状态(所有agent拼接) → 标量 value，CTDE用"""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class PerAgentCritic(nn.Module):
    """单Agent Critic：输入单agent观察(19维) → 标量 value，单agent训练用"""

    def __init__(self, obs_dim: int = 19, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
