"""Single controlled attacker; scalar critic and squashed Gaussian actor."""
import math
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self,hidden=128,obs_dim=24):
        super().__init__()
        def network(n):
            return nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,n))
        self.obs_dim=obs_dim
        self.actor=network(2)
        self.critic=network(1)
        self.log_std=nn.Parameter(torch.tensor([-.7,-1.9]))

    def distribution(self,obs):
        return Normal(self.actor(obs),self.log_std.clamp(-3,0).exp())

    def value(self,obs):
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def log_prob(dist,z):
        jac=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
        return (dist.log_prob(z)-jac).sum(-1)

    def sample(self,obs):
        d=self.distribution(obs);z=d.sample()
        return z.tanh(),z,self.log_prob(d,z)
