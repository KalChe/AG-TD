import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm
from src.environment import TSPEnvironment
from src.algorithms import AGTD, StandardTD, TDPER
import random


def evaluate_method(method, test_envs: List[TSPEnvironment]) -> Dict:
    method.network.eval()
    violations = []
    gaps = []
    
    with torch.no_grad():
        for env in test_envs:
            state = env.reset()
            predicted_value = method.network(state).item()
            optimal_cost = env.get_optimal_cost()
            
            if predicted_value > optimal_cost:
                violations.append(1)
                gaps.append(predicted_value - optimal_cost)
            else:
                violations.append(0)
                gaps.append(0)
    
    method.network.train()
    return {
        'violation_rate': np.mean(violations) * 100,
        'avg_gap': np.mean(gaps),
        'violations': violations,
        'gaps': gaps
    }


def run_episode(method, env: TSPEnvironment, epsilon: float = 0.1):
    state = env.reset()
    done = False
    
    while not done:
        available = env.get_available_actions()
        if not available:
            break
        
        if random.random() < epsilon:
            action = random.choice(available)
        else:
            method.network.eval()
            with torch.no_grad():
                best_action = None
                best_value = float('inf')
                for a in available:
                    temp_env = TSPEnvironment(env.num_nodes)
                    temp_env.nodes = env.nodes.copy()
                    temp_env.current = env.current
                    temp_env.visited = env.visited.copy()
                    next_state, cost, _ = temp_env.step(a)
                    value = -cost + method.network(next_state).item()
                    if value < best_value:
                        best_value = value
                        best_action = a
                action = best_action if best_action is not None else random.choice(available)
            method.network.train()
        
        next_state, cost, done = env.step(action)
        method.add_transition(state, action, cost, next_state)
        state = next_state
    
    loss = method.update()
    
    if hasattr(method, 'challenger_step'):
        method.challenger_step(env)
    
    return loss


def run_experiment(num_episodes: int = 1000, eval_interval: int = 50):
    methods = {
        'TD-Baseline': StandardTD(),
        'TD+PER': TDPER(),
        'AG-TD': AGTD()
    }
    
    train_envs = [TSPEnvironment(20) for _ in range(50)]
    test_sets = {
        'TSP-20': [TSPEnvironment(20) for _ in range(50)],
        'TSP-50': [TSPEnvironment(50) for _ in range(50)],
        'TSP-100': [TSPEnvironment(100) for _ in range(50)]
    }
    
    results = {name: {
        'losses': [],
        'episodes': [],
        'TSP-20': {'violation_rates': [], 'avg_gaps': []},
        'TSP-50': {'violation_rates': [], 'avg_gaps': []},
        'TSP-100': {'violation_rates': [], 'avg_gaps': []}
    } for name in methods.keys()}
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        env = random.choice(train_envs)
        
        for name, method in methods.items():
            loss = run_episode(method, env)
            results[name]['losses'].append(loss)
        
        if episode % eval_interval == 0:
            for name, method in methods.items():
                results[name]['episodes'].append(episode)
                for test_name, test_envs in test_sets.items():
                    eval_results = evaluate_method(method, test_envs)
                    results[name][test_name]['violation_rates'].append(eval_results['violation_rate'])
                    results[name][test_name]['avg_gaps'].append(eval_results['avg_gap'])
    
    return results


if __name__ == "__main__":
    results = run_experiment(num_episodes=1000, eval_interval=50)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for method_name in ['TD-Baseline', 'TD+PER', 'AG-TD']:
        print(f"\n{method_name}:")
        for test_set in ['TSP-20', 'TSP-50', 'TSP-100']:
            vr = results[method_name][test_set]['violation_rates'][-1]
            gap = results[method_name][test_set]['avg_gaps'][-1]
            print(f"  {test_set}: {vr:.1f}% violation rate, {gap:.3f} avg gap")
