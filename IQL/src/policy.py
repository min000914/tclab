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

    def act(self, obs, deterministic=False, enable_grad=False, bias_prob=0.15, sample=False):
        with torch.set_grad_enabled(enable_grad):
            #print("obs",obs.shape)
            dist = self(obs)

            if deterministic:
                action = dist.mean
            else:
                if torch.rand(1).item() < bias_prob:
                    obs = obs.detach()  # 그래디언트 방지
                    action = torch.empty(2,device=obs.device)
                        
                    eT1= obs[1] - obs[0]
                    #eT2= obs[3] - obs[2]
                    eT2= obs[4] - obs[3]
                    
                    scale = 12  # 스케일 클수록 극단적 (10~20 추천)
                    action[0] = 100.0 * sigmoid(scale * eT1)
                    action[1] = 100.0 * sigmoid(scale * eT2)
                else:
                    if sample:
                        action = dist.sample()
                        noise = torch.randn_like(action) * 5  # 표준편차 0.03의 가우시안 노이즈
                        #action = action + noise
                    else:
                        action = dist.mean
                '''action = dist.sample()
                noise = torch.randn_like(action) * noise_D  # 표준편차 0.03의 가우시안 노이즈
                action = action + noise'''
                
            #print(action)    
            return torch.clamp(action, 0.0, 100.0)
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