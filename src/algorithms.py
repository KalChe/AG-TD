import torch
import torch.optim as optim
from collections import deque
import random
from src.networks import ValueNetwork, PrioritizedReplayBuffer
from src.challenger import Challenger


class AGTD:
    def __init__(self, lr: float = 1e-3, buffer_size: int = 10000,
                 batch_size: int = 32, challenger_period: int = 100, priority_multiplier: float = 10.0):
        self.network = ValueNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        self.challenger = Challenger()
        self.batch_size = batch_size
        self.challenger_period = challenger_period
        self.priority_multiplier = priority_multiplier
        self.step_count = 0
        self.name = "AG-TD"
    
    def compute_violation(self, state, action, cost, next_state):
        self.network.eval()
        with torch.no_grad():
            v_s = self.network(state).item()
            v_next = self.network(next_state).item()
            violation = max(0, v_s - (-cost + v_next))
        self.network.train()
        return violation
    
    def add_transition(self, state, action, cost, next_state):
        violation = self.compute_violation(state, action, cost, next_state)
        self.buffer.add((state, action, cost, next_state), priority=violation)
        self.step_count += 1
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        samples, weights = self.buffer.sample(self.batch_size)
        total_loss = 0
        
        for (state, action, cost, next_state), weight in zip(samples, weights):
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            
            v_s = self.network(state_t)
            v_next = self.network(next_state_t).detach()
            
            violation = torch.relu(v_s - (-cost + v_next))
            loss = weight * (violation ** 2)
            total_loss += loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def challenger_step(self, env):
        if self.step_count % self.challenger_period == 0 and self.step_count > 0:
            counter_example, max_violation = self.challenger.find_violation(self.network, env)
            if counter_example is not None:
                self.buffer.add(counter_example, priority=max_violation * self.priority_multiplier)
                return max_violation
        return 0.0


class StandardTD:
    def __init__(self, lr: float = 1e-3):
        self.network = ValueNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.name = "TD-Baseline"
    
    def add_transition(self, state, action, cost, next_state):
        self.buffer.append((state, action, cost, next_state))
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        samples = random.sample(self.buffer, self.batch_size)
        total_loss = 0
        
        for state, action, cost, next_state in samples:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            
            v_s = self.network(state_t)
            v_next = self.network(next_state_t).detach()
            
            violation = torch.relu(v_s - (-cost + v_next))
            loss = violation ** 2
            total_loss += loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


class TDPER:
    def __init__(self, lr: float = 1e-3):
        self.network = ValueNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(10000)
        self.batch_size = 32
        self.name = "TD+PER"
    
    def compute_violation(self, state, action, cost, next_state):
        self.network.eval()
        with torch.no_grad():
            v_s = self.network(torch.FloatTensor(state)).item()
            v_next = self.network(torch.FloatTensor(next_state)).item()
            violation = max(0, v_s - (-cost + v_next))
        self.network.train()
        return violation
    
    def add_transition(self, state, action, cost, next_state):
        violation = self.compute_violation(state, action, cost, next_state)
        self.buffer.add((state, action, cost, next_state), priority=violation)
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        samples, weights = self.buffer.sample(self.batch_size)
        total_loss = 0
        
        for (state, action, cost, next_state), weight in zip(samples, weights):
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            
            v_s = self.network(state_t)
            v_next = self.network(next_state_t).detach()
            
            violation = torch.relu(v_s - (-cost + v_next))
            loss = weight * (violation ** 2)
            total_loss += loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
