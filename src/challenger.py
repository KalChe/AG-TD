import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import random
from src.environment import TSPEnvironment


class Challenger:
    def __init__(self, n_search: int = 50):
        self.n_search = n_search
    
    def find_violation(self, value_network: nn.Module, env: TSPEnvironment):
        max_violation = 0
        best_transition = None
        
        value_network.eval()
        with torch.no_grad():
            for _ in range(self.n_search):
                from src.environment import TSPEnvironment
                test_env = TSPEnvironment(env.num_nodes)
                test_env.nodes = env.nodes.copy()
                
                n_visited = random.randint(1, env.num_nodes - 1)
                visited_nodes = random.sample(range(env.num_nodes), n_visited)
                test_env.current = visited_nodes[-1]
                test_env.visited = set(visited_nodes)
                test_env.path = visited_nodes
                
                state = test_env._get_state()
                available = test_env.get_available_actions()
                
                if not available:
                    continue
                
                v_s = value_network(state).item()
                
                for action in available:
                    next_state, cost, done = test_env.step(action)
                    v_next = value_network(next_state).item()
                    
                    violation = max(0, v_s - (-cost + v_next))
                    
                    if violation > max_violation:
                        max_violation = violation
                        best_transition = (state.copy(), action, cost, next_state.copy())
                    
                    test_env.current = visited_nodes[-1]
                    test_env.visited = set(visited_nodes)
                    test_env.path = visited_nodes
        
        value_network.train()
        return best_transition, max_violation
