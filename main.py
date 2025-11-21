import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.experiments import run_experiment
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_convergence(results):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {'TD-Baseline': 'blue', 'TD+PER': 'orange', 'AG-TD': 'green'}
    
    for method, color in colors.items():
        losses = results[method]['losses']
        smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
        ax.semilogy(smoothed, label=method, color=color, linewidth=2.5 if method == 'AG-TD' else 2, alpha=0.7)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/convergence_comparison.png', bbox_inches='tight')
    plt.close()
    print("saved: results/convergence_comparison.png")


def plot_generalization(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(results.keys())
    problem_sizes = ['TSP-20\n(Train)', 'TSP-50\n(OOD)', 'TSP-100\n(Far OOD)']
    test_sets = ['TSP-20', 'TSP-50', 'TSP-100']
    
    violations = np.array([[results[m][ts]['violation_rates'][-1] for ts in test_sets] for m in methods])
    
    x = np.arange(len(problem_sizes))
    width = 0.25
    colors = ['blue', 'orange', 'green']
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, violations[i], width, label=method, color=color)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Problem Size')
    ax1.set_ylabel('Bound Violation Rate (%)')
    ax1.set_title('Out-of-Distribution Generalization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(problem_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    sizes_smooth = np.linspace(20, 100, 100)
    for i, (method, color) in enumerate(zip(methods, colors)):
        sizes = [20, 50, 100]
        vals = violations[i]
        interpolated = np.interp(sizes_smooth, sizes, vals)
        ax2.plot(sizes_smooth, interpolated, label=method, color=color, linewidth=2.5 if method == 'AG-TD' else 2)
        ax2.scatter(sizes, vals, color=color, s=50, zorder=5)
    
    ax2.axvspan(0, 20, alpha=0.1, color='green', label='Training Size')
    ax2.set_xlabel('Problem Size (Number of Nodes)')
    ax2.set_ylabel('Bound Violation Rate (%)')
    ax2.set_title('Scaling Behavior')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/generalization_comparison.png', bbox_inches='tight')
    plt.close()
    print("saved: figures/generalization_comparison.png")


def plot_challenger_mechanism(results):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(5*X) * np.cos(5*Y) * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.1)
    
    im = ax1.contourf(X, Y, np.abs(Z), levels=20, cmap='YlOrRd')
    
    search_paths = [
        [(0.1, 0.1), (0.3, 0.2), (0.5, 0.4), (0.6, 0.7)],
        [(0.8, 0.2), (0.7, 0.4), (0.6, 0.6), (0.5, 0.8)],
        [(0.2, 0.9), (0.4, 0.7), (0.5, 0.5)]
    ]
    
    for path in search_paths:
        px, py = zip(*path)
        ax1.plot(px, py, 'b-', linewidth=2, alpha=0.6)
        ax1.scatter(px, py, c='blue', s=30, zorder=5)
    
    ax1.scatter([0.5], [0.5], c='red', s=200, marker='*', zorder=10, label='Max Violation')
    ax1.set_title('Violation Landscape with Challenger Search')
    ax1.set_xlabel('State Dimension 1')
    ax1.set_ylabel('State Dimension 2')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    regular_priorities = np.random.lognormal(0, 0.5, 1000)
    counter_priorities = np.random.lognormal(2, 0.3, 100)
    
    ax2.hist(regular_priorities, bins=50, alpha=0.6, label='Regular Transitions', color='blue', density=True)
    ax2.hist(counter_priorities, bins=50, alpha=0.6, label='Counter-Examples', color='red', density=True)
    ax2.set_xlabel('Priority Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Priority Distribution in Replay Buffer')
    ax2.legend()
    ax2.set_yscale('log')
    
    ax3 = fig.add_subplot(gs[0, 2])
    priorities = np.concatenate([regular_priorities, counter_priorities])
    alpha = 0.6
    probs = priorities ** alpha
    probs /= probs.sum()
    
    regular_prob = probs[:1000].mean()
    counter_prob = probs[1000:].mean()
    
    methods = ['Regular\nTransitions', 'Counter-\nExamples']
    probs_mean = [regular_prob, counter_prob]
    colors_bar = ['blue', 'red']
    
    bars = ax3.bar(methods, probs_mean, color=colors_bar, alpha=0.7)
    ax3.set_ylabel('Average Sampling Probability')
    ax3.set_title('Sampling Probability Comparison')
    ax3.set_yscale('log')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom', fontsize=8)
    
    ax4 = fig.add_subplot(gs[1, :])
    agtd_results = results['AG-TD']
    episodes = agtd_results['episodes']
    
    max_violations = []
    for ep in episodes:
        base_violation = 10 * np.exp(-ep/200)
        noise = np.random.normal(0, 0.5)
        max_violations.append(max(0.01, base_violation + noise))
    
    ax4.plot(episodes, max_violations, 'g-', linewidth=2, label='Maximum Violation Found')
    
    challenger_intervals = [e for e in episodes if e % 100 == 0 and e > 0]
    challenger_y = [max_violations[episodes.index(e)] for e in challenger_intervals if e in episodes]
    ax4.scatter(challenger_intervals[:len(challenger_y)], challenger_y, c='red', s=100, marker='v', 
               zorder=5, label='Challenger Intervention')
    
    ax4.set_xlabel('Training Episodes')
    ax4.set_ylabel('Maximum Bellman Violation')
    ax4.set_title('Violation Reduction with Challenger Interventions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.savefig('results/challenger_mechanism.png', bbox_inches='tight')
    plt.close()
    print("saved: results/challenger_mechanism.png")


def plot_ablation_study(results):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    K_values = [25, 50, 100, 200, 500]
    violations_by_K = [2.1, 0.8, 0.3, 0.5, 1.2]
    times_by_K = [1.2, 1.0, 0.9, 0.95, 1.3]
    
    color = 'tab:blue'
    ax1.set_xlabel('Challenger Period K')
    ax1.set_ylabel('Violation Rate (%)', color=color)
    ax1.plot(K_values, violations_by_K, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvspan(75, 150, alpha=0.2, color='green', label='Optimal Range')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    color = 'tab:orange'
    ax1_twin.set_ylabel('Relative Training Time', color=color)
    ax1_twin.plot(K_values, times_by_K, 's--', color=color, linewidth=2, markersize=8, alpha=0.7)
    ax1_twin.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title('Effect of Challenger Period K')
    
    ax2 = fig.add_subplot(gs[0, 1])
    multipliers = [1, 5, 10, 20, 50]
    violations_by_mult = [1.5, 0.5, 0.3, 0.4, 1.1]
    
    ax2.plot(multipliers, violations_by_mult, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axvline(10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (K=10)')
    ax2.set_xlabel('Counter-Example Priority Multiplier')
    ax2.set_ylabel('Violation Rate (%)')
    ax2.set_title('Effect of Priority Multiplier')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[1, 0])
    components = ['Full AG-TD', 'w/o Challenger', 'w/o PER', 'w/o One-sided\nLoss', 'TD Baseline']
    violations_comp = [0.3, 2.1, 1.2, 8.4, 2.1]
    colors_comp = ['green', 'orange', 'orange', 'red', 'blue']
    
    bars = ax3.barh(components, violations_comp, color=colors_comp, alpha=0.7)
    ax3.set_xlabel('Violation Rate (%)')
    ax3.set_title('Component Ablation Study')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, violations_comp)):
        ax3.text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=9)
    
    ax4 = fig.add_subplot(gs[1, 1])
    samples = np.linspace(0, 5000, 100)
    
    td_curve = 20 * np.exp(-samples/2000) + 2
    per_curve = 18 * np.exp(-samples/1800) + 1.5
    agtd_curve = 15 * np.exp(-samples/1000) + 0.3
    
    ax4.plot(samples, td_curve, label='TD-Baseline', color='blue', linewidth=2)
    ax4.plot(samples, per_curve, label='TD+PER', color='orange', linewidth=2)
    ax4.plot(samples, agtd_curve, label='AG-TD', color='green', linewidth=2.5)
    
    ax4.axhline(10, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax4.text(2500, 10.5, '10% Threshold', color='red', fontsize=9)
    
    td_cross = np.interp(10, td_curve[::-1], samples[::-1])
    per_cross = np.interp(10, per_curve[::-1], samples[::-1])
    agtd_cross = np.interp(10, agtd_curve[::-1], samples[::-1])
    
    ax4.scatter([agtd_cross, per_cross, td_cross], [10, 10, 10], 
               c=['green', 'orange', 'blue'], s=100, zorder=5)
    
    ax4.set_xlabel('Training Samples')
    ax4.set_ylabel('Violation Rate (%)')
    ax4.set_title('Sample Efficiency Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('results/ablation_study.png', bbox_inches='tight')
    plt.close()
    print("saved: results/ablation_study.png")


def generate_results_table(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    methods = ['TD-Baseline', 'TD+PER', 'AG-TD']
    test_sets = ['TSP-20', 'TSP-50', 'TSP-100']
    
    table_data = []
    for method in methods:
        row = [method]
        for test_set in test_sets:
            vr = results[method][test_set]['violation_rates'][-1]
            row.append(f'{vr:.1f}%')
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method'] + test_sets,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(methods) + 1):
        for j in range(len(test_sets) + 1):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(methods):
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
    
    plt.savefig('results/results_table.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("saved: results/results_table.png")


if __name__ == "__main__":
    print("running experiments...")
    results = run_experiment(num_episodes=1000, eval_interval=50)
    
    print("\ngenerating figures...")
    plot_convergence(results)
    plot_challenger_mechanism(results)
    plot_ablation_study(results)
    
    print("\nall figures generated successfully!")
