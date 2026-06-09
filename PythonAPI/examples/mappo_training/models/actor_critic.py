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

    def __init__(self, obs_dim: int = 19, act_dim: int = 3, hidden: int = 128,
                 init_log_std: float = -0.5, min_log_std: float = -1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        # init_log_std=-0.5 → std≈0.6，对 tanh 输出更友好；
        # 旧默认 0.0 (std=1.0) 会让动作大量饱和到 ±1，导致 PPO 早期 approx_kl 爆表
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        # 破冻结(R5)：log_std 下限，防止策略过早收敛到"刹车/不动"使探索消失。
        # min_log_std=-1.0 → std≥0.37，保证持续有探索噪声逃出局部最优。
        self.min_log_std = float(min_log_std)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        # clamp 到下限：保证探索噪声不塌缩（上限放宽，不影响 init）
        log_std = self.log_std.clamp(min=self.min_log_std).expand_as(mu)
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

    def sample_action_z(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """与 sample_action 相同，但额外返回 pre-tanh 隐变量 z。

        训练时应存储 z（而非 action_tanh），评估走 evaluate_actions_z，
        避免 atanh 反推带来的精度损失，保证未更新策略时 PPO ratio 严格为 1（修复 #4）。
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
        return action_tanh, log_prob, z

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

    def evaluate_actions_z(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """用存储的 pre-tanh z 直接重算 log_prob/entropy（修复 #4）。

        不经 atanh 反推，因此采样分布与评估分布完全一致：
        策略参数未变时 new_logp == old_logp，ratio == 1。
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
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


class AgentConditionedCritic(nn.Module):
    """Agent-id 条件化集中式 Critic（修复 #1）。

    输入 [global_state, one_hot(agent_id)] → 该 agent 专属的 V(s, i)。
    解决"同一 global_state 对应多个 per-agent return"导致单一 V(s) 回归目标
    互相冲突、收敛到均值的问题：每个 agent 的 value 各自回归，互不污染。

    参数在所有 agent 间共享（通过 one-hot 条件化区分），既保留 CTDE 的
    全局信息，又允许每个 agent 拥有独立的价值估计。
    """

    def __init__(self, state_dim: int, num_agents: int, hidden: int = 128):
        super().__init__()
        self.num_agents = num_agents
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_agents, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor, agent_id) -> torch.Tensor:
        """agent_id 可以是 int（整个 batch 同一 agent）或 LongTensor [B]。"""
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        batch = global_state.shape[0]
        device = global_state.device
        onehot = torch.zeros(batch, self.num_agents, device=device)
        if isinstance(agent_id, int):
            onehot[:, agent_id] = 1.0
        else:
            idx = agent_id.to(device=device, dtype=torch.long).view(-1, 1)
            onehot.scatter_(1, idx, 1.0)
        x = torch.cat([global_state, onehot], dim=-1)
        return self.net(x).squeeze(-1)


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
