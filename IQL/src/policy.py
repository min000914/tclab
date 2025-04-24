import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        raw_mean = self.net(obs)
        mean = torch.tanh(raw_mean) * 50.0 + 50.0
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False, noise_D=0.03):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
                noise = torch.randn_like(action) * noise_D  # 표준편차 0.03의 가우시안 노이즈
                action = action + noise
            return torch.clamp(action, 0.0, 100.0)


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