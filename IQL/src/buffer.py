import numpy as np
import random
from collections import deque, namedtuple
from src.util import torchify

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, act, next_obs,rew, done = map(np.stack, zip(*samples))
        return {
            'observations': torchify(obs),
            'actions': torchify(act),
            'rewards': torchify(rew),
            'next_observations': torchify(next_obs),
            'terminals': torchify(done),
        }

    def load_dataset(self, dataset_path):
        dataset = np.load(dataset_path)
        for i in range(len(dataset['observations'])):
            transition = (
                dataset['observations'][i],
                dataset['actions'][i],
                dataset['next_observations'][i],
                dataset['rewards'][i],
                dataset['terminals'][i]
            )
            self.add(transition)
    def __len__(self):
        return len(self.buffer)
    
    
import torch
class LSTMReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, seq_transition):
        self.buffer.append(seq_transition)

    def sample(self, batch_size):
        """
        시퀀스 단위로 샘플링
        """
        batch = random.sample(self.buffer, batch_size)
        obs_seq, action_seq, next_obs_seq, reward_seq, done_seq = zip(*batch)

        # === Tensor 변환 ===
        obs_seq = torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs_seq])
        action_seq = torch.stack([torch.tensor(a, dtype=torch.float32) for a in action_seq])
        next_obs_seq = torch.stack([torch.tensor(no, dtype=torch.float32) for no in next_obs_seq])
        reward_seq = torch.stack([torch.tensor(r, dtype=torch.float32) for r in reward_seq])
        done_seq = torch.stack([torch.tensor(d, dtype=torch.float32) for d in done_seq])

        return {
            'observations': obs_seq,
            'actions': action_seq,
            'next_observations': next_obs_seq,
            'rewards': reward_seq,
            'terminals': done_seq
        }
    def __len__(self):
        return len(self.buffer)    