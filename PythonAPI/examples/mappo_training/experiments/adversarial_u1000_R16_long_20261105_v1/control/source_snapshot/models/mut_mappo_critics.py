"""Local value functions for the controlled four-car IPPO baseline.

The public rollout format remains 80 values in physical-ID order.  This
adapter never exposes another vehicle's slot to a role's value network.  It
matches ``AgentConditionedCritic(global_state, agent_id)`` so the existing
rollout/GAE/PPO path can be reused rather than copied.

For *fully independent* PPO updates, construct one adapter per blue role
(``agent_ids=(aid,)``), with its own optimizer and ValueNormalizer, and call
the common PPO update per role.  Using the default two-role adapter with a
pooled ValueNorm or global gradient clipping produces a controlled local
value variant, but does not make those training operations independent.
"""

from __future__ import annotations

import torch
from torch import nn

from models.actor_critic import PerAgentCritic


LOCAL_CRITIC_PROTOCOL = "mut_ippo_local_observation_value_v1"


class LocalObservationCritic(nn.Module):
    """Separate local value MLPs with physical-ID-compatible dispatch.

    ``global_state[..., aid * obs_dim:(aid + 1) * obs_dim]`` is the same
    local observation available to that role's actor, including the local
    sensing features.  No agent one-hot or other physical slot is an input.
    ``num_agents`` describes the public state layout, not shared parameters.
    """

    def __init__(self, state_dim: int = 80, num_agents: int = 4,
                 hidden: int = 128, *, obs_dim: int = 20,
                 agent_ids=(2, 3)):
        super().__init__()
        for name, value in (("state_dim", state_dim), ("num_agents", num_agents),
                            ("hidden", hidden), ("obs_dim", obs_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if state_dim != num_agents * obs_dim:
            raise ValueError("state_dim must contain one equal local observation slot per physical ID")
        owned = tuple(agent_ids)
        if not owned or len(set(owned)) != len(owned):
            raise ValueError("agent_ids must be nonempty and unique")
        if any(isinstance(aid, bool) or not isinstance(aid, int)
               or not 0 <= aid < num_agents for aid in owned):
            raise ValueError("agent_ids must be integer physical IDs inside the state layout")
        self.state_dim, self.num_agents = state_dim, num_agents
        self.obs_dim, self.hidden = obs_dim, hidden
        self.agent_ids = owned
        self.critics = nn.ModuleDict({
            str(aid): PerAgentCritic(obs_dim, hidden) for aid in self.agent_ids
        })

    def model_spec(self) -> dict:
        """Persist this alongside the learner spec to reject incompatible resumes."""
        return {
            "critic_protocol": LOCAL_CRITIC_PROTOCOL,
            "critic_kind": "ippo_local",
            "critic_input_scope": "own_physical_slot_local_observation_only",
            "critic_state_dim": self.state_dim,
            "critic_local_obs_dim": self.obs_dim,
            "critic_physical_ids": list(self.agent_ids),
            "critic_hidden": self.hidden,
            "critic_parameter_sharing": False,
        }

    def forward(self, global_state: torch.Tensor, agent_id) -> torch.Tensor:
        if not isinstance(global_state, torch.Tensor):
            raise TypeError("global_state must be a torch.Tensor")
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if global_state.dim() != 2 or global_state.shape[1] != self.state_dim:
            raise ValueError(f"global_state must have shape [B, {self.state_dim}]")
        if not global_state.is_floating_point():
            raise TypeError("global_state must use a floating-point dtype")
        if isinstance(agent_id, bool):
            raise TypeError("agent_id must be an integer physical ID, not bool")
        if isinstance(agent_id, int):
            if agent_id not in self.agent_ids:
                raise ValueError(f"agent_id={agent_id} has no owned local critic")
            start = agent_id * self.obs_dim
            return self.critics[str(agent_id)](global_state[:, start:start + self.obs_dim])
        if not isinstance(agent_id, torch.Tensor):
            raise TypeError("agent_id must be int or torch.Tensor")
        if agent_id.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise TypeError("agent_id tensor must use an integer dtype")
        ids = agent_id.to(device=global_state.device, dtype=torch.long).reshape(-1)
        if ids.numel() != global_state.shape[0]:
            raise ValueError("agent_id batch size does not match global_state")
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for aid in self.agent_ids:
            valid |= ids == aid
        if not valid.all().item():
            raise ValueError("agent_id tensor contains a role with no owned local critic")
        values = global_state.new_zeros(global_state.shape[0])
        for aid in self.agent_ids:
            rows = (ids == aid).nonzero(as_tuple=True)[0]
            if rows.numel():
                start = aid * self.obs_dim
                predictions = self.critics[str(aid)](
                    global_state[rows, start:start + self.obs_dim]
                )
                values = values.index_copy(0, rows, predictions)
        return values
