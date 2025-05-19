import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp
import math

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def sigmoid(x, scale=1.0):
    if x < 0:
        return 0.0
    return 1 / (1 + math.exp(-scale * x))


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        raw_mean = self.net(obs)
        #mean = torch.sigmoid(raw_mean)
        mean = torch.tanh(raw_mean) * 50.0 + 50.0
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False, exp_prob=0.15, sample=False, noise=0.0):
        with torch.set_grad_enabled(enable_grad):
            #print("obs",obs.shape)
            dist = self(obs)

            if deterministic:
                action = dist.mean
                exp = False
            else:
                if torch.rand(1).item() < exp_prob:
                    obs = obs.detach()  # 그래디언트 방지

                    eT1 = obs[1] - obs[0]
                    eT2 = obs[4] - obs[3]
                    
                    action = dist.mean

                    noise = torch.randn_like(action) * noise
                    
                    # eT1에 대한 노이즈
                    if eT1 >= 0:
                        noise[0] = torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[0]) // 2
                    else:
                        noise[0] = -torch.abs(noise[0]) if torch.rand(1).item() > 0.2 else torch.abs(noise[0]) // 2
                    
                    # eT2에 대한 노이즈
                    if eT2 >= 0:
                        noise[1] = torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else -torch.abs(noise[1]) // 2
                    else:
                        noise[1] = -torch.abs(noise[1]) if torch.rand(1).item() > 0.2 else torch.abs(noise[1]) // 2

                    action = action + noise
                    exp = False
                else:
                    action = dist.mean
                    exp = False
            #print(action)    
            return torch.clamp(action, 0.0, 100.0), exp
            #return torch.clamp(action, 0.0, 1.0)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)