"""Two-blue CTDE value network with the original 40+2 input and weight keys.

The rollout transport remains four physical slots (80 values), so existing
GAE/bootstrap code can be reused. Only slots 2/3 reach the network; physical
IDs 2/3 are explicitly mapped to the base critic's local IDs 0/1. Public
state_dim/num_agents describe transport, not the MLP's actual input width.
"""
import torch
from models.actor_critic import AgentConditionedCritic


class BlueTeamCritic(AgentConditionedCritic):
    def __init__(self, hidden=128):
        super().__init__(40, 2, hidden)
        self.state_dim = 80
        self.num_agents = 4
        self.value_state_dim = 40
        self.value_id_dim = 2

    def forward(self, global_state, agent_id):
        if not isinstance(global_state, torch.Tensor):
            raise TypeError('global_state must be a tensor')
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if global_state.dim() != 2 or global_state.shape[1] != 80:
            raise ValueError('rollout transport must have shape [B,80]')
        if not global_state.is_floating_point():
            raise TypeError('global_state must use a floating-point dtype')
        if isinstance(agent_id, bool):
            raise TypeError('agent_id must be a physical blue ID, not bool')
        if isinstance(agent_id, int):
            if agent_id not in (2,3):
                raise ValueError('team critic only predicts physical blue IDs 2/3')
            ids=torch.full((global_state.shape[0],),agent_id,device=global_state.device,dtype=torch.long)
        else:
            if not isinstance(agent_id,torch.Tensor) or agent_id.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
                raise TypeError('agent_id must be an integer or integer tensor')
            ids=agent_id.to(device=global_state.device,dtype=torch.long).reshape(-1)
            if ids.numel()!=global_state.shape[0] or not ((ids==2)|(ids==3)).all().item():
                raise ValueError('agent_id batch must contain only matching physical blue IDs 2/3')
        local_ids=torch.nn.functional.one_hot(ids-2,num_classes=2).to(dtype=global_state.dtype)
        return self.net(torch.cat((global_state[:,40:80],local_ids),dim=-1)).squeeze(-1)
