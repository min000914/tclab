import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observations", "actions", "rewards", "next_observations", "terminals"])
    
    def add(self, obs, action, reward, next_obs, terminal):
        e = self.experience(obs, action, reward, next_obs, terminal)
        self.memory.append(e)
    
    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        def to_tensor(x): return torch.tensor(np.stack(x), dtype=torch.float32).to(self.device)

        obs = to_tensor([e.obs for e in experiences])
        actions = to_tensor([e.action for e in experiences])
        rewards = to_tensor([[e.reward] for e in experiences])
        next_obs = to_tensor([e.next_obs for e in experiences])
        terminals = to_tensor([[float(e.terminal)] for e in experiences])
  
        return (obs, actions, rewards, next_obs, terminals)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)