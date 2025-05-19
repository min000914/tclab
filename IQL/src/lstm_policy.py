import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import math

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
def sigmoid(x, scale=1.0):
    if x < 0:
        return 0.0
    return 1 / (1 + math.exp(-scale * x))

class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, lstm_layers=2, normalization= True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.normalization = normalization
        
    def forward(self, obs, hidden_state=None):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)  # (Batch, Seq, Feature)
        
        batch_size = obs.size(0)
        
        if hidden_state is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=obs.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=obs.device)
            hidden_state = (h0, c0)
        
        lstm_out, hidden_state = self.lstm(obs, hidden_state)
        #print(lstm_out.shape)
        #print(lstm_out)
        lstm_out = lstm_out[:, -1, :]  # Use the last output for action prediction
        
        raw_mean = self.fc_mean(lstm_out)
        #print(raw_mean)
        if self.normalization:
            mean = torch.sigmoid(raw_mean)
        else:
            mean= 50 * (torch.tanh(raw_mean) + 1)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        #scale_tril = torch.diag_embed(std) 
        return MultivariateNormal(mean, scale_tril=scale_tril), hidden_state

    def act(self, obs, hidden_state=None, deterministic=False, enable_grad=False, bias_prob=0.15, sample=False):
        with torch.set_grad_enabled(enable_grad):
            dist, hidden_state = self(obs, hidden_state)

            if deterministic:
                action = dist.mean
            '''else:
                if torch.rand(1).item() < bias_prob:
                    obs = obs.detach()
                    action = torch.empty(2, device=obs.device)

                    eT1 = obs[1] - obs[0]
                    eT2 = obs[3] - obs[2]
                    #eT3 = obs[4] - obs[3]
                    
                    scale = 12
                    action[0] = 100.0 * sigmoid(scale * eT1)
                    action[1] = 100.0 * sigmoid(scale * eT2)
                else:
                    if sample:
                        action = dist.sample()
                    else:
                        action = dist.mean'''
            
            if self.normalization:    
                return torch.clamp(action, 0.0, 1.0), hidden_state
            else:
                return torch.clamp(action, 0.0, 100.0), hidden_state
            #return action,hidden_state

