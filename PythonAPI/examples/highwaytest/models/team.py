import math
import torch
from torch import nn
from torch.distributions import Normal


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


class Team(nn.Module):
    """No actor sharing. Critic reads state; actors read only local obs/messages.

    Encoder modules also exist in no-comm controls for matched allocated parameter
    counts. Their outputs are zeroed and have no effect on the action distribution.
    """
    def __init__(self, config):
        super().__init__()
        self.communication = config.communication
        self.exploration = config.exploration
        h, m = config.hidden, config.message_dim
        if config.task.research_protocol == "legacy_v2":
            obs_dim, state_dim = 24, 28
        elif config.task.research_protocol == 'bottleneck_paired_attack_v4':
            from ..bottleneck.contract import OBS_DIM, STATE_DIM
            obs_dim, state_dim = OBS_DIM, STATE_DIM
        else:
            from ..common.contract import OBS_DIM, STATE_DIM
            obs_dim, state_dim = OBS_DIM, STATE_DIM
        self.obs_dim, self.state_dim = obs_dim, state_dim
        self.encoders = nn.ModuleList([mlp(obs_dim, h, m) for _ in range(2)])
        self.actors = nn.ModuleList([mlp(obs_dim + m + 1, h, 2) for _ in range(2)])
        self.log_std = nn.Parameter(torch.full((2, 2), -0.5))
        if self.exploration is not None:
            with torch.no_grad():
                self.log_std.copy_(torch.tensor(self.exploration['initial']).log().expand(2,-1))
        self.critic = mlp(state_dim, h, 2)
        for actor in self.actors:
            nn.init.orthogonal_(actor[-1].weight, gain=0.01)
            nn.init.zeros_(actor[-1].bias)

    def distribution(self, obs, links, *, intervention="normal", stale_obs=None):
        message_obs = stale_obs if intervention == "stale" and stale_obs is not None else obs
        messages = [torch.tanh(enc(message_obs[..., i, :])) for i, enc in enumerate(self.encoders)]
        enabled = float(self.communication and intervention != "zero")
        means = []
        for i in range(2):
            mask = links[..., i, :] * enabled
            inputs = torch.cat((obs[..., i, :], messages[1 - i] * mask, mask), -1)
            means.append(self.actors[i](inputs))
        mu = torch.stack(means, -2)
        return Normal(mu, self.action_std())

    def action_std(self):
        if self.exploration is None:
            return self.log_std.clamp(-2, 0.5).exp()
        lo = self.log_std.new_tensor(self.exploration['minimum']).log()
        hi = self.log_std.new_tensor(self.exploration['maximum']).log()
        return self.log_std.clamp(lo, hi).exp()

    @staticmethod
    def log_prob(dist, z):
        jacobian = 2 * (math.log(2) - z - nn.functional.softplus(-2 * z))
        return (dist.log_prob(z) - jacobian).sum(-1)

    def sample(self, obs, links):
        dist = self.distribution(obs, links)
        z = dist.sample()
        return z.tanh(), z, self.log_prob(dist, z)

    def values(self, state):
        return self.critic(state)
