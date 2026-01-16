"""
Visualization utilities for Traffic Signal Control System.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path


def plot_training_metrics(
    loss_history: List[float],
    title: str = "Federated Learning Training Loss",
    save_path: str = None
):
    """
    Plot training loss over federated rounds.

    Args:
        loss_history: List of loss values per round
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_traffic_metrics(
    df: pd.DataFrame,
    intersection_id: int = None,
    save_path: str = None
):
    """
    Plot traffic simulation metrics.

    Args:
        df: DataFrame with traffic simulation data
        intersection_id: Specific intersection to plot (None for all)
        save_path: Path to save the figure
    """
    if intersection_id is not None:
        df = df[df['intersection_id'] == intersection_id]
        title_suffix = f" - Intersection {intersection_id}"
    else:
        title_suffix = " - All Intersections"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Queue Length over Time
    ax1 = axes[0, 0]
    for int_id in df['intersection_id'].unique():
        int_df = df[df['intersection_id'] == int_id]
        ax1.plot(int_df['time'], int_df['total_queue_length'],
                label=f'Intersection {int_id}', alpha=0.7)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Queue Length')
    ax1.set_title('Queue Length Over Time' + title_suffix)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average Waiting Time
    ax2 = axes[0, 1]
    for int_id in df['intersection_id'].unique():
        int_df = df[df['intersection_id'] == int_id]
        ax2.plot(int_df['time'], int_df['average_waiting_time'],
                label=f'Intersection {int_id}', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Average Waiting Time (seconds)')
    ax2.set_title('Average Waiting Time' + title_suffix)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Throughput
    ax3 = axes[1, 0]
    for int_id in df['intersection_id'].unique():
        int_df = df[df['intersection_id'] == int_id]
        ax3.plot(int_df['time'], int_df['throughput'],
                label=f'Intersection {int_id}', alpha=0.7)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Cumulative Throughput')
    ax3.set_title('Vehicle Throughput' + title_suffix)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Queue Length Distribution by Direction
    ax4 = axes[1, 1]
    if intersection_id is not None:
        directions = ['north_queue', 'south_queue', 'east_queue', 'west_queue']
        for direction in directions:
            ax4.plot(df['time'], df[direction],
                    label=direction.replace('_queue', '').capitalize(), alpha=0.7)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Queue Length')
        ax4.set_title('Queue by Direction' + title_suffix)
        ax4.legend()
    else:
        # Box plot for all intersections
        queue_data = df.groupby('intersection_id')['total_queue_length'].mean()
        ax4.bar(queue_data.index, queue_data.values)
        ax4.set_xlabel('Intersection ID')
        ax4.set_ylabel('Average Queue Length')
        ax4.set_title('Average Queue by Intersection')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_federated_convergence(
    client_losses: Dict[int, List[float]],
    global_loss: List[float] = None,
    save_path: str = None
):
    """
    Plot federated learning convergence across clients.

    Args:
        client_losses: Dictionary mapping client_id to loss history
        global_loss: Global model loss history
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))

    # Plot client losses
    for client_id, losses in client_losses.items():
        plt.plot(range(1, len(losses) + 1), losses,
                '--', alpha=0.5, label=f'Client {client_id}')

    # Plot global loss
    if global_loss:
        plt.plot(range(1, len(global_loss) + 1), global_loss,
                'k-o', linewidth=2, markersize=6, label='Global Model')

    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Federated Learning Convergence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_network_metrics(
    metrics_history: List[Dict],
    save_path: str = None
):
    """
    Plot network simulation metrics over time.

    Args:
        metrics_history: List of metric dictionaries over time
        save_path: Path to save the figure
    """
    if not metrics_history:
        print("No network metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    timestamps = range(len(metrics_history))

    # Extract metrics
    latencies = [m.get('average_latency_ms', 0) for m in metrics_history]
    packet_loss = [m.get('packet_loss_rate', 0) * 100 for m in metrics_history]
    bytes_sent = [m.get('total_bytes_sent', 0) / 1e6 for m in metrics_history]  # MB
    congestion = [m.get('current_congestion', 1) for m in metrics_history]

    # Plot 1: Latency
    axes[0, 0].plot(timestamps, latencies, 'b-')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Average Latency (ms)')
    axes[0, 0].set_title('Network Latency')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Packet Loss
    axes[0, 1].plot(timestamps, packet_loss, 'r-')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Packet Loss Rate (%)')
    axes[0, 1].set_title('Packet Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Data Transfer
    axes[1, 0].plot(timestamps, bytes_sent, 'g-')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Total Data Sent (MB)')
    axes[1, 0].set_title('Cumulative Data Transfer')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Congestion
    axes[1, 1].plot(timestamps, congestion, 'm-')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Congestion Factor')
    axes[1, 1].set_title('Network Congestion')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing visualization utilities...")

    # Test training metrics plot
    dummy_losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12]
    plot_training_metrics(dummy_losses, save_path="results/test_training.png")

    print("Visualization test complete!")
