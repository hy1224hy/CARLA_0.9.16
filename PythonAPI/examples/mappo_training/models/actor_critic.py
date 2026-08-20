"""
统一的 Actor-Critic 模型定义
用于 MAPPO (CTDE) 和 单Agent PPO 训练
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """2层MLP (128, Tanh) → mu_head + log_std, 用于 tanh-squashed Gaussian 策略"""

    def __init__(self, obs_dim: int = 19, act_dim: int = 3, hidden: int = 128,
                 init_log_std: float = -0.5, min_log_std: float = -1.0,
                 max_log_std: Optional[float] = None):
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
        self.max_log_std = (
            None if max_log_std is None else float(max_log_std)
        )
        if not math.isfinite(self.min_log_std):
            raise ValueError("min_log_std must be finite")
        if self.max_log_std is not None:
            if not math.isfinite(self.max_log_std):
                raise ValueError("max_log_std must be finite when configured")
            if self.max_log_std <= self.min_log_std:
                raise ValueError("max_log_std must be greater than min_log_std")
            if not self.min_log_std <= float(init_log_std) <= self.max_log_std:
                raise ValueError(
                    "init_log_std must lie inside [min_log_std, max_log_std]"
                )
        elif float(init_log_std) < self.min_log_std:
            raise ValueError("init_log_std must be at or above min_log_std")

    @staticmethod
    def _log_tanh_jacobian(z: torch.Tensor) -> torch.Tensor:
        """Stable elementwise ``log(1 - tanh(z)^2)``.

        The equivalent ``log(1 - tanh(z)^2 + eps)`` saturates at the chosen
        epsilon and therefore gives the wrong density and entropy gradient for
        large ``|z|``.  This softplus form is the change-of-variables identity
        used by squashed-Gaussian policies.
        """

        return 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))

    def effective_log_std(self) -> torch.Tensor:
        """Return the bounded log standard deviation used by the policy."""

        if self.max_log_std is None:
            return self.log_std.clamp(min=self.min_log_std)
        return self.log_std.clamp(
            min=self.min_log_std, max=self.max_log_std
        )

    @torch.no_grad()
    def project_log_std_(self) -> None:
        """Project the stored parameter so checkpoints cannot hide clamp drift."""

        if self.max_log_std is None:
            self.log_std.clamp_(min=self.min_log_std)
        else:
            self.log_std.clamp_(
                min=self.min_log_std, max=self.max_log_std
            )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.effective_log_std().expand_as(mu)
        return mu, log_std

    @classmethod
    def _squashed_log_prob(
        cls,
        dist: Normal,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return (
            dist.log_prob(z) - cls._log_tanh_jacobian(z)
        ).sum(dim=-1)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = self._squashed_log_prob(dist, z)
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
        log_prob = self._squashed_log_prob(dist, z)
        return action_tanh, log_prob, z

    def evaluate_actions(self, obs: torch.Tensor, action_tanh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        clipped = torch.clamp(action_tanh, -0.999, 0.999)
        z = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        log_prob = self._squashed_log_prob(dist, z)
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
        log_prob = self._squashed_log_prob(dist, z)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def estimate_post_tanh_entropy(
        self,
        obs: torch.Tensor,
        *,
        sample_count: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate current squashed-policy entropy from fresh samples.

        Returns one post-tanh and one analytic pre-tanh entropy value per
        observation.  Rollout samples must not be passed here: using samples
        from the old policy would estimate cross-entropy rather than
        ``H[pi_current]``.
        """

        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer")
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        pre_tanh_entropy = dist.entropy().sum(dim=-1)
        if sample_count == 1:
            z = dist.rsample()
            post_tanh_entropy = -self._squashed_log_prob(dist, z)
        else:
            z = dist.rsample((int(sample_count),))
            expanded = Normal(mu.unsqueeze(0), std.unsqueeze(0))
            post_tanh_entropy = -self._squashed_log_prob(expanded, z).mean(
                dim=0
            )
        return post_tanh_entropy, pre_tanh_entropy

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
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be a positive integer")
        self.state_dim = state_dim
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
        if not isinstance(global_state, torch.Tensor):
            raise TypeError("global_state must be a torch.Tensor")
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if global_state.dim() != 2 or global_state.shape[1] != self.state_dim:
            raise ValueError(
                f"global_state must have shape [B, {self.state_dim}], "
                f"got {tuple(global_state.shape)}"
            )
        if not global_state.is_floating_point():
            raise TypeError("global_state must use a floating-point dtype")
        batch = global_state.shape[0]
        device = global_state.device
        onehot = torch.zeros(
            batch, self.num_agents, device=device, dtype=global_state.dtype
        )
        if isinstance(agent_id, int):
            if not 0 <= agent_id < self.num_agents:
                raise ValueError(
                    f"agent_id={agent_id} outside [0, {self.num_agents})"
                )
            onehot[:, agent_id] = 1.0
        else:
            if not isinstance(agent_id, torch.Tensor):
                raise TypeError("agent_id must be int or torch.Tensor")
            if agent_id.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise TypeError("agent_id tensor must use an integer dtype")
            flat_ids = agent_id.to(device=device, dtype=torch.long).view(-1)
            if flat_ids.numel() != batch:
                raise ValueError(
                    f"agent_id batch size={flat_ids.numel()} does not match "
                    f"global_state batch size={batch}"
                )
            if torch.any(flat_ids < 0).item() or torch.any(
                flat_ids >= self.num_agents
            ).item():
                raise ValueError(
                    f"agent_id tensor contains values outside "
                    f"[0, {self.num_agents})"
                )
            idx = flat_ids.view(-1, 1)
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
