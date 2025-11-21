import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def add(self, transition: Tuple, priority: float = 1.0):
        priority = max(priority, 1e-6)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return [], np.array([])
        
        batch_size = min(batch_size, len(self.buffer))
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, weights
    
    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        num_nodes = state.shape[-1] - 4
        visited_mask = state[..., :num_nodes]
        features = state[..., num_nodes:]
        
        embedded = self.embedding(features)
        visited_sum = visited_mask.sum(dim=-1, keepdim=True)
        attended = embedded * (visited_sum / max(1, num_nodes))
        output = self.output(attended)
        return output.squeeze(-1)
