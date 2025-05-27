import torch
import torch.nn as nn
import math
from torch.distributions import MultivariateNormal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def sigmoid(x, scale=1.0):
    if x < 0:
        return 0.0
    return 1 / (1 + math.exp(-scale * x))

class RNNPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, rnn_layers=2, normalization=False):
        super().__init__()
        self.rnn = nn.RNN(input_size=obs_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.normalization = normalization
        
    def forward(self, obs, hidden_state=None):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)  # (Batch, Seq, Feature)
        
        batch_size = obs.size(0)
        
        if hidden_state is None:
            h0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_dim, device=obs.device)
            hidden_state = h0
        
        rnn_out, hidden_state = self.rnn(obs, hidden_state)
        rnn_out = rnn_out[:, -1, :]  # 마지막 시점의 출력 사용
        
        raw_mean = self.fc_mean(rnn_out)
        if self.normalization:
            mean = torch.sigmoid(raw_mean)
        else:
            mean = 50 * (torch.tanh(raw_mean) + 1)

        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril), hidden_state

    def act(self, obs, hidden_state=None, deterministic=False, enable_grad=False, bias_prob=0.15, sample=False):
        with torch.set_grad_enabled(enable_grad):
            dist, hidden_state = self(obs, hidden_state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()  # 누락된 부분 보완
            
            if self.normalization:    
                return torch.clamp(action, 0.0, 1.0), hidden_state
            else:
                return torch.clamp(action, 0.0, 100.0), hidden_state
