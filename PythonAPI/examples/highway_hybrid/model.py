"""Factorized Gaussian acceleration and masked categorical lane policy."""
import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical, kl_divergence


class HybridActorCritic(nn.Module):
    def __init__(self,hidden=128,obs_dim=28):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.acceleration_head=nn.Linear(hidden,1)
        self.lane_head=nn.Linear(hidden,3)
        self.critic=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.orthogonal_(self.acceleration_head.weight,gain=.01)
        nn.init.zeros_(self.acceleration_head.bias)
        nn.init.orthogonal_(self.lane_head.weight,gain=.01)
        nn.init.zeros_(self.lane_head.bias)
        self.log_std=nn.Parameter(torch.tensor([-.7]))

    def distributions(self,obs,mask):
        mask=mask.to(dtype=torch.bool,device=obs.device)
        if mask.shape!=(*obs.shape[:-1],3) or not mask.any(-1).all():
            raise ValueError('Every observation needs a nonempty three-action lane mask')
        features=self.actor(obs)
        acceleration=Normal(self.acceleration_head(features),self.log_std.clamp(-3,0).exp())
        # Finite sentinel avoids undefined inf-inf derivatives in categorical KL.
        # exp(-1e9) underflows to exactly zero for invalid actions in float32.
        lane=Categorical(logits=self.lane_head(features).masked_fill(~mask,-1e9))
        return acceleration,lane

    def value(self,obs):return self.critic(obs).squeeze(-1)

    @staticmethod
    def acceleration_log_prob(distribution,z):
        jacobian=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
        return (distribution.log_prob(z)-jacobian).sum(-1)

    def log_prob(self,obs,mask,z,lane):
        normal,category=self.distributions(obs,mask)
        return self.acceleration_log_prob(normal,z)+category.log_prob(lane)

    def sample(self,obs,mask):
        normal,category=self.distributions(obs,mask)
        z=normal.sample();lane=category.sample()
        return z.tanh(),lane,z,self.acceleration_log_prob(normal,z)+category.log_prob(lane)

    def deterministic(self,obs,mask):
        normal,category=self.distributions(obs,mask)
        return normal.loc.tanh(),category.probs.argmax(-1)


def mutual_terms(student,teacher,obs,mask):
    normal,lane=student.distributions(obs,mask)
    with torch.no_grad():peer_normal,peer_lane=teacher.distributions(obs,mask)
    return (kl_divergence(normal,peer_normal).sum(-1).mean(),
            kl_divergence(lane,peer_lane).mean())
