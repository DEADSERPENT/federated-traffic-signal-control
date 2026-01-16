"""
Professional Visualization Module for Academic Papers.
Publication-quality figures for FL Traffic Signal Control.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette for consistency
COLORS = {
    'fl': '#2E86AB',        # Blue - Federated Learning
    'local': '#A23B72',     # Magenta - Local ML
    'fixed': '#F18F01',     # Orange - Fixed Time
    'ideal': '#C73E1D',     # Red
    'normal': '#3B1F2B',    # Dark
    'degraded': '#95190C',  # Dark Red
    'stressed': '#610345',  # Purple
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01'
}


def plot_fl_convergence(
    round_metrics: List[Dict],
    title: str = "Federated Learning Convergence",
    save_path: str = None
):
    """
    Plot FL convergence with MAE and MSE over rounds.

    Args:
        round_metrics: List of per-round metrics
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rounds = [m["round"] for m in round_metrics]
    maes = [m.get("mae", m.get("global_mae", 0)) for m in round_metrics]
    mses = [m.get("mse", m.get("global_mse", 0)) for m in round_metrics]

    # MAE plot
    ax1.plot(rounds, maes, 'o-', color=COLORS['fl'], linewidth=2, markersize=6)
    ax1.fill_between(rounds, maes, alpha=0.2, color=COLORS['fl'])
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_title('MAE Convergence')
    ax1.grid(True, alpha=0.3)

    # Add improvement annotation
    improvement = (maes[0] - maes[-1]) / maes[0] * 100
    ax1.annotate(f'{improvement:.1f}% improvement',
                xy=(rounds[-1], maes[-1]), xytext=(rounds[-1]*0.7, maes[0]*0.9),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    # MSE plot
    ax2.plot(rounds, mses, 's-', color=COLORS['secondary'], linewidth=2, markersize=6)
    ax2.fill_between(rounds, mses, alpha=0.2, color=COLORS['secondary'])
    ax2.set_xlabel('Federated Round')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('MSE Convergence')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    plt.show()
    return fig


def plot_method_comparison(
    comparison_data: Dict,
    save_path: str = None
):
    """
    Plot comparison bar chart between methods.

    Args:
        comparison_data: Comparison dictionary with metrics
        save_path: Path to save figure
    """
    methods = comparison_data["methods"]
    metrics = comparison_data["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Colors for each method
    method_colors = [COLORS['fixed'], COLORS['local'], COLORS['fl']][:len(methods)]

    # 1. Average Waiting Time
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, metrics["avg_waiting_time"], color=method_colors, edgecolor='black')
    ax1.set_ylabel('Average Waiting Time (s)')
    ax1.set_title('Average Vehicle Waiting Time')
    for bar, val in zip(bars1, metrics["avg_waiting_time"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 2. Average Queue Length
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, metrics["avg_queue_length"], color=method_colors, edgecolor='black')
    ax2.set_ylabel('Average Queue Length')
    ax2.set_title('Average Queue Length at Intersections')
    for bar, val in zip(bars2, metrics["avg_queue_length"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. Throughput
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, metrics["throughput_per_hour"], color=method_colors, edgecolor='black')
    ax3.set_ylabel('Vehicles per Hour')
    ax3.set_title('Traffic Throughput')
    for bar, val in zip(bars3, metrics["throughput_per_hour"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # 4. MAE (only for ML methods)
    ax4 = axes[1, 1]
    mae_values = metrics.get("mae", [0] * len(methods))
    # Filter out zero values (Fixed-Time has no MAE)
    filtered_methods = [m for m, v in zip(methods, mae_values) if v > 0]
    filtered_maes = [v for v in mae_values if v > 0]
    filtered_colors = [c for c, v in zip(method_colors, mae_values) if v > 0]

    if filtered_maes:
        bars4 = ax4.bar(filtered_methods, filtered_maes, color=filtered_colors, edgecolor='black')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Prediction Accuracy (MAE)')
        for bar, val in zip(bars4, filtered_maes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'N/A for Fixed-Time', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Prediction Accuracy (MAE)')

    plt.suptitle('Performance Comparison: Fixed-Time vs Local-ML vs Federated Learning',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    plt.show()
    return fig


def plot_network_stress_results(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot network stress experiment results.

    Args:
        results: Dictionary of scenario results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    scenarios = list(results.keys())
    colors = [COLORS.get(s, COLORS['primary']) for s in scenarios]

    # Extract data
    final_maes = [results[s]["final_mae"] for s in scenarios]
    comm_times = [results[s]["avg_comm_time_per_round_ms"] for s in scenarios]
    latencies = [results[s]["network_config"]["base_latency"] for s in scenarios]
    loss_rates = [results[s]["network_config"]["packet_loss_probability"] * 100 for s in scenarios]

    # 1. Final MAE by scenario
    ax1 = axes[0, 0]
    bars = ax1.bar(scenarios, final_maes, color=COLORS['fl'], edgecolor='black')
    ax1.set_ylabel('Final MAE')
    ax1.set_title('Model Accuracy Under Network Stress')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Communication time
    ax2 = axes[0, 1]
    ax2.bar(scenarios, comm_times, color=COLORS['secondary'], edgecolor='black')
    ax2.set_ylabel('Avg Comm Time (ms)')
    ax2.set_title('Communication Overhead')
    ax2.tick_params(axis='x', rotation=45)

    # 3. MAE vs Latency
    ax3 = axes[1, 0]
    ax3.scatter(latencies, final_maes, s=100, c=COLORS['fl'], edgecolors='black', zorder=5)
    ax3.plot(latencies, final_maes, '--', color='gray', alpha=0.5)
    ax3.set_xlabel('Network Latency (ms)')
    ax3.set_ylabel('Final MAE')
    ax3.set_title('Impact of Latency on Accuracy')

    # 4. MAE vs Packet Loss
    ax4 = axes[1, 1]
    ax4.scatter(loss_rates, final_maes, s=100, c=COLORS['accent'], edgecolors='black', zorder=5)
    ax4.plot(loss_rates, final_maes, '--', color='gray', alpha=0.5)
    ax4.set_xlabel('Packet Loss Rate (%)')
    ax4.set_ylabel('Final MAE')
    ax4.set_title('Impact of Packet Loss on Accuracy')

    plt.suptitle('Federated Learning Performance Under Network Stress',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    plt.show()
    return fig


def plot_scalability_results(
    results: Dict[int, Dict],
    save_path: str = None
):
    """
    Plot scalability experiment results.

    Args:
        results: Dictionary of scalability results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    num_clients = sorted(results.keys())
    final_maes = [results[n]["final_mae"] for n in num_clients]
    times = [results[n]["total_time_s"] for n in num_clients]
    conv_rounds = [results[n]["convergence_round"] for n in num_clients]
    data_transfer = [results[n]["total_data_transfer_bytes"] / (1024*1024) for n in num_clients]

    # 1. MAE vs Clients
    ax1 = axes[0, 0]
    ax1.plot(num_clients, final_maes, 'o-', color=COLORS['fl'], linewidth=2, markersize=8)
    ax1.fill_between(num_clients, final_maes, alpha=0.2, color=COLORS['fl'])
    ax1.set_xlabel('Number of Clients')
    ax1.set_ylabel('Final MAE')
    ax1.set_title('Model Accuracy vs Scale')
    ax1.grid(True, alpha=0.3)

    # 2. Training Time vs Clients
    ax2 = axes[0, 1]
    ax2.plot(num_clients, times, 's-', color=COLORS['secondary'], linewidth=2, markersize=8)
    ax2.fill_between(num_clients, times, alpha=0.2, color=COLORS['secondary'])
    ax2.set_xlabel('Number of Clients')
    ax2.set_ylabel('Total Time (s)')
    ax2.set_title('Training Time Scaling')
    ax2.grid(True, alpha=0.3)

    # Add linear reference
    linear_ref = [times[0] * (n / num_clients[0]) for n in num_clients]
    ax2.plot(num_clients, linear_ref, '--', color='gray', label='Linear scaling', alpha=0.7)
    ax2.legend()

    # 3. Convergence Round
    ax3 = axes[1, 0]
    ax3.bar(num_clients, conv_rounds, color=COLORS['accent'], edgecolor='black')
    ax3.set_xlabel('Number of Clients')
    ax3.set_ylabel('Rounds to Converge')
    ax3.set_title('Convergence Speed')

    # 4. Data Transfer
    ax4 = axes[1, 1]
    ax4.bar(num_clients, data_transfer, color=COLORS['fl'], edgecolor='black')
    ax4.set_xlabel('Number of Clients')
    ax4.set_ylabel('Total Data Transfer (MB)')
    ax4.set_title('Communication Cost')

    plt.suptitle('Federated Learning Scalability Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    plt.show()
    return fig


def create_summary_dashboard(
    fl_results: Dict,
    comparison: Dict,
    network_results: Dict,
    scalability_results: Dict,
    save_path: str = None
):
    """
    Create comprehensive summary dashboard.

    Args:
        fl_results: FL experiment results
        comparison: Method comparison data
        network_results: Network stress results
        scalability_results: Scalability results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. FL Convergence (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if fl_results.get("round_metrics"):
        rounds = [m["round"] for m in fl_results["round_metrics"]]
        maes = [m.get("mae", m.get("global_mae", 0)) for m in fl_results["round_metrics"]]
        ax1.plot(rounds, maes, 'o-', color=COLORS['fl'], linewidth=2, markersize=5)
        ax1.fill_between(rounds, maes, alpha=0.2, color=COLORS['fl'])
    ax1.set_xlabel('Round')
    ax1.set_ylabel('MAE')
    ax1.set_title('FL Convergence')
    ax1.grid(True, alpha=0.3)

    # 2. Method comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if comparison.get("methods"):
        methods = comparison["methods"]
        maes = comparison["metrics"].get("mae", [0]*len(methods))
        colors = [COLORS['fixed'], COLORS['local'], COLORS['fl']][:len(methods)]
        ax2.barh(methods, maes, color=colors, edgecolor='black')
    ax2.set_xlabel('MAE')
    ax2.set_title('Method Comparison')

    # 3. Network stress (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if network_results:
        scenarios = list(network_results.keys())
        maes = [network_results[s]["final_mae"] for s in scenarios]
        ax3.bar(range(len(scenarios)), maes, color=COLORS['secondary'], edgecolor='black')
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.set_ylabel('MAE')
    ax3.set_title('Network Stress')

    # 4. Scalability (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if scalability_results:
        clients = sorted(scalability_results.keys())
        maes = [scalability_results[c]["final_mae"] for c in clients]
        ax4.plot(clients, maes, 'o-', color=COLORS['fl'], linewidth=2)
    ax4.set_xlabel('Clients')
    ax4.set_ylabel('MAE')
    ax4.set_title('Scalability')
    ax4.grid(True, alpha=0.3)

    # 5. Key metrics summary (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary_text = "KEY RESULTS\n" + "="*20 + "\n\n"
    if fl_results:
        summary_text += f"FL Final MAE: {fl_results.get('mae', 'N/A'):.4f}\n"
        summary_text += f"FL Rounds: {fl_results.get('num_rounds', 'N/A')}\n\n"
    if comparison.get("improvements"):
        for method, imps in comparison["improvements"].items():
            summary_text += f"{method}:\n"
            summary_text += f"  MAE (-){imps.get('mae_reduction', 0):.1f}%\n"
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # 6. Traffic metrics (bottom, spanning all)
    ax6 = fig.add_subplot(gs[2, :])
    if comparison.get("methods"):
        methods = comparison["methods"]
        x = np.arange(len(methods))
        width = 0.25

        wait_times = comparison["metrics"].get("avg_waiting_time", [0]*len(methods))
        queues = comparison["metrics"].get("avg_queue_length", [0]*len(methods))

        ax6.bar(x - width/2, wait_times, width, label='Waiting Time (s)', color=COLORS['fl'])
        ax6.bar(x + width/2, queues, width, label='Queue Length', color=COLORS['secondary'])

        ax6.set_xticks(x)
        ax6.set_xticklabels(methods)
        ax6.legend()
    ax6.set_title('Traffic Performance Metrics')

    plt.suptitle('Federated Learning Traffic Signal Control - Summary Dashboard',
                fontsize=16, fontweight='bold')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Dashboard saved: {save_path}")

    plt.show()
    return fig
