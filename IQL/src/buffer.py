import numpy as np
import random
from collections import deque, namedtuple
from src.util import torchify

class ReplayBuffer:
    def __init__(self, capacity=100_000):
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

    def __len__(self):
        return len(self.buffer)