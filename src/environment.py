import numpy as np


class TSPEnvironment:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.nodes = np.random.uniform(0, 1, (num_nodes, 2))
        self.reset()
    
    def reset(self):
        self.current = 0
        self.visited = {0}
        self.path = [0]
        return self._get_state()
    
    def _get_state(self):
        state = []
        for i in range(self.num_nodes):
            if i in self.visited:
                state.append(1.0)
            else:
                state.append(0.0)
        state.append(self.current / max(1, self.num_nodes - 1))
        state.append(len(self.visited) / self.num_nodes)
        
        if self.current < len(self.nodes):
            state.extend(self.nodes[self.current].tolist())
        else:
            state.extend([0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def get_available_actions(self):
        return [i for i in range(self.num_nodes) if i not in self.visited]
    
    def step(self, action):
        cost = np.linalg.norm(self.nodes[self.current] - self.nodes[action])
        self.current = action
        self.visited.add(action)
        self.path.append(action)
        
        done = len(self.visited) == self.num_nodes
        if done:
            cost += np.linalg.norm(self.nodes[action] - self.nodes[0])
        
        return self._get_state(), cost, done
    
    def get_optimal_cost(self):
        visited = [False] * self.num_nodes
        current = 0
        visited[0] = True
        total_cost = 0
        
        for _ in range(self.num_nodes - 1):
            nearest = None
            min_dist = float('inf')
            for j in range(self.num_nodes):
                if not visited[j]:
                    dist = np.linalg.norm(self.nodes[current] - self.nodes[j])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
            if nearest is not None:
                total_cost += min_dist
                current = nearest
                visited[nearest] = True
        
        total_cost += np.linalg.norm(self.nodes[current] - self.nodes[0])
        return total_cost
