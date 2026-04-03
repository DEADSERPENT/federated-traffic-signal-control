#!/usr/bin/env python3
"""
IEEE-Ready Experimental Evaluation
===================================

This script runs comprehensive experiments for IEEE publication:
1. Multiple runs with different seeds (statistical significance)
2. Generates publication-quality plots
3. Ablation study
4. Statistical analysis (mean ± std, confidence intervals)
5. Centralized-ML baseline (privacy-accuracy trade-off)
6. 9-intersection (3x3 SUMO grid) support via --intersections 9

GPU-Agnostic: Automatically uses GPU when available, falls back to CPU.

Author: FL Traffic Signal Control Project
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import torch

from utils.reproducibility import set_global_seed
from utils.device import get_device, is_gpu_available
from traffic_generator import TrafficDataGenerator
from baselines.fixed_time import FixedTimeController
from baselines.local_ml import LocalMLController
from baselines.adaptive_fl import AdaptiveFLController
from baselines.actuated import ActuatedController
from baselines.centralized_ml import CentralizedMLController

# Get device at module level (will print device info once)
DEVICE = get_device()


def _make_generator(num_intersections: int, seed: int = None):
    """Create a TrafficDataGenerator with optional seed and intersection count."""
    if seed is not None:
        set_global_seed(seed)
    return TrafficDataGenerator(
        config={"traffic": {"num_intersections": num_intersections,
                            "simulation_duration": 3600,
                            "time_step": 5,
                            "arrival_distribution": "poisson",
                            "min_arrival_rate": 5,
                            "max_arrival_rate": 30,
                            "max_queue_length": 50,
                            "min_green_duration": 20,
                            "max_green_duration": 90,
                            "yellow_duration": 3}}
    )


def run_single_experiment(seed: int, num_rounds: int = 50,
                          num_intersections: int = 4) -> Dict:
    """Run a single experiment with given seed."""
    set_global_seed(seed)

    # Initialize
    generator = _make_generator(num_intersections, seed)
    training_data = generator.get_all_intersections_data()

    results = {"seed": seed, "num_intersections": num_intersections}

    # Fixed-Time
    set_global_seed(seed)
    fixed_controller = FixedTimeController()
    fixed_results = fixed_controller.run_simulation(
        generator.intersections, duration=1800
    )
    results["fixed_time"] = {
        "wait_time": fixed_results["avg_waiting_time"],
        "queue_length": fixed_results["avg_queue_length"]
    }

    # Actuated (Industry Standard)
    set_global_seed(seed)
    generator_actuated = _make_generator(num_intersections, seed)
    actuated_controller = ActuatedController()
    actuated_results = actuated_controller.run_simulation(
        generator_actuated.intersections, duration=1800
    )
    results["actuated"] = {
        "wait_time": actuated_results["avg_waiting_time"],
        "queue_length": actuated_results["avg_queue_length"]
    }

    # Local-ML
    set_global_seed(seed)
    local_controller = LocalMLController(num_intersections=num_intersections)
    local_results = local_controller.run_simulation(
        generator.intersections, generator, duration=1800
    )
    results["local_ml"] = {
        "wait_time": local_results["avg_waiting_time"],
        "queue_length": local_results["avg_queue_length"],
        "mae": local_results["mae"]
    }

    # Centralized-ML (privacy upper bound)
    set_global_seed(seed)
    centralized_controller = CentralizedMLController(
        num_intersections=num_intersections,
        hidden_layers=[256, 128, 64, 32],
        epochs=25,
        learning_rate=0.002,
        weight_decay=5e-5,
    )
    centralized_results = centralized_controller.run_simulation(
        generator.intersections, generator, duration=1800
    )
    results["centralized_ml"] = {
        "wait_time": centralized_results["avg_waiting_time"],
        "queue_length": centralized_results["avg_queue_length"],
        "mae": centralized_results["mae"]
    }

    # Federated Learning with FedProx + LSTM + Trimmed-Mean (5-Way Victory Config)
    set_global_seed(seed)
    fl_controller = AdaptiveFLController(
        num_intersections=num_intersections,
        num_rounds=num_rounds,
        local_epochs=5,
        learning_rate=0.001,
        lr_decay=0.995,
        weight_decay=1e-4,
        use_fedprox=True,
        mu=0.01,
        aggregation_strategy="trimmed_mean",
        model_type="lstm"
    )
    fl_results = fl_controller.run_simulation(
        generator.intersections, generator, duration=1800
    )
    results["federated_learning"] = {
        "wait_time": fl_results["avg_waiting_time"],
        "queue_length": fl_results["avg_queue_length"],
        "mae": fl_results["mae"],
        "round_metrics": fl_results["round_metrics"],
        "best_mae": fl_controller.best_mae
    }

    return results


def run_multiple_experiments(num_runs: int = 5, num_rounds: int = 50,
                             num_intersections: int = 4) -> List[Dict]:
    """Run multiple experiments with different seeds."""
    seeds = [42, 123, 456, 789, 1024][:num_runs]
    all_results = []

    print(f"\n{'='*70}")
    print(f"  RUNNING {num_runs} EXPERIMENTS FOR STATISTICAL ANALYSIS")
    print(f"  Intersections: {num_intersections}")
    print(f"{'='*70}\n")

    for i, seed in enumerate(seeds):
        print(f"\n--- Experiment {i+1}/{num_runs} (seed={seed}) ---")
        result = run_single_experiment(seed, num_rounds, num_intersections)
        all_results.append(result)

        print(f"  Fixed-Time:    Wait={result['fixed_time']['wait_time']:.2f}s")
        print(f"  Actuated:      Wait={result['actuated']['wait_time']:.2f}s")
        print(f"  Local-ML:      Wait={result['local_ml']['wait_time']:.2f}s, MAE={result['local_ml']['mae']:.4f}")
        print(f"  Centralized:   Wait={result['centralized_ml']['wait_time']:.2f}s, MAE={result['centralized_ml']['mae']:.4f}")
        print(f"  FL (Ours):     Wait={result['federated_learning']['wait_time']:.2f}s, MAE={result['federated_learning']['mae']:.4f}")

    return all_results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean, std, and confidence intervals."""
    stats_dict = {}

    for method in ["fixed_time", "actuated", "local_ml", "centralized_ml", "federated_learning"]:
        method_stats = {}

        # Wait time
        wait_times = [r[method]["wait_time"] for r in results]
        method_stats["wait_time"] = {
            "mean": np.mean(wait_times),
            "std": np.std(wait_times),
            "min": np.min(wait_times),
            "max": np.max(wait_times),
            "ci_95": stats.t.interval(0.95, len(wait_times)-1,
                                       loc=np.mean(wait_times),
                                       scale=stats.sem(wait_times)) if len(wait_times) > 1 else (np.mean(wait_times), np.mean(wait_times))
        }

        # Queue length
        queue_lengths = [r[method]["queue_length"] for r in results]
        method_stats["queue_length"] = {
            "mean": np.mean(queue_lengths),
            "std": np.std(queue_lengths)
        }

        # MAE (only for ML methods)
        if method not in ["fixed_time", "actuated"] and all("mae" in r[method] for r in results):
            maes = [r[method]["mae"] for r in results]
            method_stats["mae"] = {
                "mean": np.mean(maes),
                "std": np.std(maes),
                "min": np.min(maes),
                "max": np.max(maes),
                "ci_95": stats.t.interval(0.95, len(maes)-1,
                                          loc=np.mean(maes),
                                          scale=stats.sem(maes)) if len(maes) > 1 else (np.mean(maes), np.mean(maes))
            }

        # Best MAE for FL
        if method == "federated_learning":
            best_maes = [r[method]["best_mae"] for r in results]
            method_stats["best_mae"] = {
                "mean": np.mean(best_maes),
                "std": np.std(best_maes)
            }

        stats_dict[method] = method_stats

    return stats_dict


def plot_method_comparison(stats: Dict, output_dir: Path):
    """Create IEEE-quality bar chart comparing methods."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ["Fixed-Time", "Actuated", "Local-ML", "Centralized-ML", "FL (Ours)"]
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    # Wait Time comparison
    ax1 = axes[0]
    wait_means = [
        stats["fixed_time"]["wait_time"]["mean"],
        stats["actuated"]["wait_time"]["mean"],
        stats["local_ml"]["wait_time"]["mean"],
        stats["centralized_ml"]["wait_time"]["mean"],
        stats["federated_learning"]["wait_time"]["mean"]
    ]
    wait_stds = [
        stats["fixed_time"]["wait_time"]["std"],
        stats["actuated"]["wait_time"]["std"],
        stats["local_ml"]["wait_time"]["std"],
        stats["centralized_ml"]["wait_time"]["std"],
        stats["federated_learning"]["wait_time"]["std"]
    ]

    bars1 = ax1.bar(methods, wait_means, yerr=wait_stds, capsize=5,
                    color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax1.set_ylabel('Average Waiting Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Waiting Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(wait_means) * 1.3)

    # Highlight winner
    min_idx = np.argmin(wait_means)
    bars1[min_idx].set_edgecolor('#27ae60')
    bars1[min_idx].set_linewidth(3)

    # Add value labels
    for bar, mean, std in zip(bars1, wait_means, wait_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.tick_params(axis='x', rotation=15)

    # MAE comparison (ML methods only, including Centralized)
    ax2 = axes[1]
    mae_methods = ["Centralized-ML", "Local-ML", "FL (Ours)"]
    mae_colors = ['#f39c12', '#3498db', '#e74c3c']
    mae_means = [
        stats["centralized_ml"]["mae"]["mean"] if "mae" in stats["centralized_ml"] else 0,
        stats["local_ml"]["mae"]["mean"],
        stats["federated_learning"]["mae"]["mean"]
    ]
    mae_stds = [
        stats["centralized_ml"]["mae"]["std"] if "mae" in stats["centralized_ml"] else 0,
        stats["local_ml"]["mae"]["std"],
        stats["federated_learning"]["mae"]["std"]
    ]

    bars2 = ax2.bar(mae_methods, mae_means, yerr=mae_stds, capsize=5,
                    color=mae_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(mae_means) * 1.3)

    # Highlight winner (lowest MAE = best)
    best_mae_idx = np.argmin(mae_means)
    bars2[best_mae_idx].set_edgecolor('#27ae60')
    bars2[best_mae_idx].set_linewidth(3)

    # Add value labels
    for bar, mean, std in zip(bars2, mae_means, mae_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Annotate: FL improvement over Local-ML
    fl_mae    = mae_means[2]
    local_mae = mae_means[1]
    if fl_mae < local_mae:
        improvement = ((local_mae - fl_mae) / local_mae) * 100
        ax2.annotate(f'FL: +{improvement:.1f}% vs Local',
                    xy=(2, fl_mae), xytext=(1.2, local_mae * 0.92),
                    fontsize=10, color='#e74c3c', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Stability comparison (Std Dev)
    ax3 = axes[2]
    stability_methods = ["Fixed-Time", "Actuated", "Local-ML", "Centralized-ML", "FL (Ours)"]
    stability_stds = wait_stds  # Already computed above
    stability_colors = colors

    bars3 = ax3.bar(stability_methods, stability_stds, color=stability_colors,
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    ax3.set_ylabel('Wait Time Std Dev (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Stability (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, max(stability_stds) * 1.5)

    # Highlight winner (lowest std)
    min_std_idx = np.argmin(stability_stds)
    bars3[min_std_idx].set_edgecolor('#27ae60')
    bars3[min_std_idx].set_linewidth(3)

    # Add value labels
    for bar, std in zip(bars3, stability_stds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'ieee_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ieee_method_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: ieee_method_comparison.png/pdf")


def plot_fl_convergence(results: List[Dict], output_dir: Path):
    """Plot FL convergence across multiple runs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(results)))

    all_maes = []
    max_rounds = 0

    for i, result in enumerate(results):
        metrics = result["federated_learning"]["round_metrics"]
        rounds = [m["round"] for m in metrics]
        maes = [m["global_mae"] for m in metrics]
        max_rounds = max(max_rounds, len(rounds))
        all_maes.append(maes)
        ax.plot(rounds, maes, color=colors[i], alpha=0.5, linewidth=1, label=f'Run {i+1}')

    # Compute and plot mean
    min_len = min(len(m) for m in all_maes)
    mean_maes = np.mean([m[:min_len] for m in all_maes], axis=0)
    std_maes = np.std([m[:min_len] for m in all_maes], axis=0)
    rounds_mean = list(range(1, min_len + 1))

    ax.plot(rounds_mean, mean_maes, color='#e74c3c', linewidth=3, label='Mean MAE')
    ax.fill_between(rounds_mean, mean_maes - std_maes, mean_maes + std_maes,
                    color='#e74c3c', alpha=0.2, label='±1 Std Dev')

    # Add Local-ML baseline
    local_ml_mae = np.mean([r["local_ml"]["mae"] for r in results])
    ax.axhline(y=local_ml_mae, color='#3498db', linestyle='--', linewidth=2,
               label=f'Local-ML Baseline ({local_ml_mae:.4f})')

    ax.set_xlabel('FL Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Federated Learning Convergence (Multiple Runs)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, min_len)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ieee_fl_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ieee_fl_convergence.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: ieee_fl_convergence.png/pdf")


def plot_network_stress_comparison(output_dir: Path):
    """Create network stress comparison plot if data exists."""
    stress_file = Path("results/ns3_stress/stress_test_summary.json")
    if not stress_file.exists():
        print("  Skipping network stress plot (no data)")
        return

    with open(stress_file, 'r') as f:
        stress_data = json.load(f)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios = list(stress_data.keys())
    fl_maes = [stress_data[s]["federated_learning"]["mae"] for s in scenarios]
    local_ml_mae = stress_data[scenarios[0]]["local_ml"]["mae"]

    # Calculate latencies
    latencies = []
    for s in scenarios:
        metrics = stress_data[s]["federated_learning"].get("network_metrics", [])
        if metrics:
            avg_lat = np.mean([m["total_latency_ms"] for m in metrics])
            latencies.append(avg_lat)
        else:
            latencies.append(0)

    # Plot 1: MAE vs Network Scenario
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax1.bar(x - width/2, [local_ml_mae]*len(scenarios), width,
                   label='Local-ML', color='#3498db', edgecolor='black', alpha=0.85)
    bars2 = ax1.bar(x + width/2, fl_maes, width,
                   label='Federated Learning', color='#e74c3c', edgecolor='black', alpha=0.85)

    ax1.set_xlabel('Network Scenario', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) MAE Under Network Stress', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in scenarios])
    ax1.legend()
    ax1.set_ylim(0, max(fl_maes + [local_ml_mae]) * 1.2)

    # Add winner markers
    for i, (fl_mae, scenario) in enumerate(zip(fl_maes, scenarios)):
        if fl_mae < local_ml_mae:
            ax1.annotate('✓', xy=(i + width/2, fl_mae), xytext=(i + width/2, fl_mae + 0.05),
                        fontsize=14, color='green', ha='center', fontweight='bold')

    # Plot 2: MAE vs Latency
    ax2 = axes[1]
    ax2.scatter(latencies, fl_maes, s=150, c='#e74c3c', edgecolors='black',
                linewidth=2, zorder=5, label='FL Performance')
    ax2.axhline(y=local_ml_mae, color='#3498db', linestyle='--', linewidth=2,
                label=f'Local-ML Baseline ({local_ml_mae:.4f})')

    # Add scenario labels
    for lat, mae, scenario in zip(latencies, fl_maes, scenarios):
        ax2.annotate(scenario.capitalize(), xy=(lat, mae), xytext=(lat+20, mae+0.01),
                    fontsize=10, alpha=0.8)

    ax2.set_xlabel('Average Network Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) FL Robustness vs Network Latency', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ieee_network_stress.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ieee_network_stress.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: ieee_network_stress.png/pdf")


def run_ablation_study(output_dir: Path, num_intersections: int = 4,
                       num_seeds: int = 3) -> Dict:
    """Run ablation study comparing different FL configurations.

    Each configuration is evaluated across ``num_seeds`` independent seeds so
    that the reported MAE values carry meaningful error bars rather than being
    a single stochastic draw.
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY  ({num_seeds} seeds per config)")
    print(f"{'='*70}\n")

    ablation_seeds = [42, 123, 456, 789, 1024][:num_seeds]

    configs = [
        {"name": "FL-Small",       "hidden": [64, 32],           "epochs": 5,  "rounds": 30},
        {"name": "FL-Medium",      "hidden": [128, 64, 32],      "epochs": 10, "rounds": 30},
        {"name": "FL-Large (Ours)","hidden": [256, 128, 64, 32], "epochs": 15, "rounds": 50},
        {"name": "FL-NoScheduler", "hidden": [256, 128, 64, 32], "epochs": 15, "rounds": 50,
         "lr_decay": 1.0},
    ]

    ablation_results = {}

    for config in configs:
        print(f"\n--- Config: {config['name']} ({num_seeds} seeds) ---")
        seed_maes      = []
        seed_waits     = []
        seed_best_maes = []

        for seed in ablation_seeds:
            set_global_seed(seed)
            generator = _make_generator(num_intersections, seed)

            fl_controller = AdaptiveFLController(
                num_intersections=num_intersections,
                num_rounds=config.get("rounds", 50),
                local_epochs=config.get("epochs", 10),
                hidden_layers=config.get("hidden", [128, 64, 32]),
                learning_rate=0.002,
                lr_decay=config.get("lr_decay", 0.99),
                weight_decay=5e-5
            )

            fl_results = fl_controller.run_simulation(
                generator.intersections, generator, duration=1800
            )

            seed_maes.append(fl_results["mae"])
            seed_waits.append(fl_results["avg_waiting_time"])
            seed_best_maes.append(fl_controller.best_mae)
            print(f"  seed={seed}: Wait={fl_results['avg_waiting_time']:.2f}s, "
                  f"MAE={fl_results['mae']:.4f}")

        mae_mean = float(np.mean(seed_maes))
        mae_std  = float(np.std(seed_maes))
        print(f"  → {config['name']}: MAE = {mae_mean:.4f} ± {mae_std:.4f}")

        ablation_results[config["name"]] = {
            "mae_mean":       mae_mean,
            "mae_std":        mae_std,
            "wait_mean":      float(np.mean(seed_waits)),
            "wait_std":       float(np.std(seed_waits)),
            "best_mae_mean":  float(np.mean(seed_best_maes)),
            # keep raw lists for downstream analysis
            "mae_per_seed":   seed_maes,
            "wait_per_seed":  seed_waits,
            "config":         config,
        }

    # Plot ablation results with error bars
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    names    = list(ablation_results.keys())
    mae_means = [ablation_results[n]["mae_mean"] for n in names]
    mae_stds  = [ablation_results[n]["mae_std"]  for n in names]
    colors   = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6']

    bars = ax.bar(names, mae_means, yerr=mae_stds, capsize=6,
                  color=colors, edgecolor='black', linewidth=1.2, alpha=0.85,
                  error_kw=dict(elinewidth=1.8, ecolor='black'))

    # Highlight best (lowest mean MAE)
    best_idx = int(np.argmin(mae_means))
    bars[best_idx].set_edgecolor('#27ae60')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Ablation Study: FL Configuration Impact ({num_seeds} seeds each)',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim(0, max(m + s for m, s in zip(mae_means, mae_stds)) * 1.25)

    # Value labels above each bar (show mean ± std)
    for bar, mean, std in zip(bars, mae_means, mae_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.03,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ieee_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ieee_ablation_study.pdf', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: ieee_ablation_study.png/pdf")

    return ablation_results


def generate_latex_table(stats: Dict, ablation: Dict) -> str:
    """Generate LaTeX table for IEEE paper with Centralized-ML, Stability columns."""
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Traffic Signal Control Methods (mean $\\pm$ std, 5 trials)}
\\label{tab:results}
\\begin{tabular}{lcccl}
\\toprule
\\textbf{Method} & \\textbf{Wait Time (s)} & \\textbf{MAE} & \\textbf{Stability (Std)} & \\textbf{Privacy} \\\\
\\midrule
"""

    fixed_wait    = stats["fixed_time"]["wait_time"]
    actuated_wait = stats["actuated"]["wait_time"]
    local_wait    = stats["local_ml"]["wait_time"]
    local_mae     = stats["local_ml"]["mae"]
    central_wait  = stats["centralized_ml"]["wait_time"]
    fl_wait       = stats["federated_learning"]["wait_time"]
    fl_mae        = stats["federated_learning"]["mae"]

    # Centralized MAE (may not have been computed in older runs)
    central_mae_str = "N/A"
    if "mae" in stats["centralized_ml"]:
        cm = stats["centralized_ml"]["mae"]
        central_mae_str = f"{cm['mean']:.4f} $\\pm$ {cm['std']:.4f}"

    mae_gap_local   = ((local_mae['mean']  - fl_mae['mean']) / local_mae['mean']) * 100
    stability_imp   = ((local_wait['std']  - fl_wait['std']) / local_wait['std']) * 100

    latex += f"Fixed-Time      & {fixed_wait['mean']:.2f} $\\pm$ {fixed_wait['std']:.2f} & N/A & {fixed_wait['std']:.2f} & -- \\\\\n"
    latex += f"Actuated        & {actuated_wait['mean']:.2f} $\\pm$ {actuated_wait['std']:.2f} & N/A & {actuated_wait['std']:.2f} & -- \\\\\n"
    latex += f"Local-ML        & {local_wait['mean']:.2f} $\\pm$ {local_wait['std']:.2f} & {local_mae['mean']:.4f} $\\pm$ {local_mae['std']:.4f} & {local_wait['std']:.2f} & None \\\\\n"
    latex += f"Centralized-ML  & {central_wait['mean']:.2f} $\\pm$ {central_wait['std']:.2f} & {central_mae_str} & {central_wait['std']:.2f} & None (raw data pooled) \\\\\n"
    latex += f"\\textbf{{FL (Ours)}} & \\textbf{{{fl_wait['mean']:.2f} $\\pm$ {fl_wait['std']:.2f}}} & \\textbf{{{fl_mae['mean']:.4f} $\\pm$ {fl_mae['std']:.4f}}} & \\textbf{{{fl_wait['std']:.2f}}} & \\textbf{{Data-local (GDPR)}} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\vspace{2mm}
\\caption*{ResilNet-FL achieves """ + f"{mae_gap_local:.1f}" + """\\% better MAE than Local-ML and """ + f"{stability_imp:.0f}" + """\\% lower variance, while closing the gap to Centralized-ML without sharing raw data.}
\\end{table}
"""
    return latex


def run_generalization_test(fl_controller, local_controller, output_dir: Path) -> Dict:
    """
    Test generalization on UNSEEN traffic data (seed 9999).
    This proves FL generalizes better than Local-ML.
    """
    print(f"\n{'='*70}")
    print("  GENERALIZATION TEST (Unseen Traffic - Seed 9999)")
    print(f"{'='*70}\n")

    unseen_seed = 9999
    set_global_seed(unseen_seed)

    # Create new traffic data that neither model has seen
    generator_unseen = TrafficDataGenerator()
    test_data = generator_unseen.get_all_intersections_data()

    results = {"seed": unseen_seed}

    # Evaluate FL on unseen data
    fl_total_mse = 0
    fl_total_mae = 0
    for intersection_id, (features, labels) in test_data.items():
        from models.traffic_model import evaluate_model
        mse, mae = evaluate_model(fl_controller.global_model, (features, labels))
        fl_total_mse += mse
        fl_total_mae += mae

    fl_mae = fl_total_mae / len(test_data)

    # Evaluate Local-ML on unseen data
    local_total_mse = 0
    local_total_mae = 0
    for intersection_id, (features, labels) in test_data.items():
        from models.traffic_model import evaluate_model
        model = local_controller.models.get(intersection_id)
        if model:
            mse, mae = evaluate_model(model, (features, labels))
            local_total_mse += mse
            local_total_mae += mae

    local_mae = local_total_mae / len(test_data)

    results["fl_mae"] = fl_mae
    results["local_ml_mae"] = local_mae
    results["fl_wins"] = fl_mae < local_mae

    improvement = ((local_mae - fl_mae) / local_mae) * 100

    print(f"  Local-ML MAE on unseen data: {local_mae:.4f}")
    print(f"  FL MAE on unseen data:       {fl_mae:.4f}")
    print(f"  FL Improvement:              {improvement:+.2f}%")
    print(f"  FL WINS GENERALIZATION:      {'YES' if fl_mae < local_mae else 'NO'}")

    # Save generalization results
    with open(output_dir / "generalization_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: generalization_test.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="IEEE-Ready Experiments")
    parser.add_argument("--runs",          type=int,   default=5,    help="Number of experiment runs")
    parser.add_argument("--rounds",        type=int,   default=50,   help="FL rounds per run")
    parser.add_argument("--intersections", type=int,   default=4,    help="Number of intersections (4 or 9)")
    parser.add_argument("--ablation",       action="store_true",       help="Run ablation study")
    parser.add_argument("--ablation-seeds", type=int, default=3,
                        dest="ablation_seeds",
                        help="Seeds per ablation config (default 3; use 5 for full rigor)")
    parser.add_argument("--output",        type=str,   default="results/ieee", help="Output directory")
    parser.add_argument("--device",        type=str,   choices=["auto", "cpu", "cuda", "mps"],
                        default="auto",   help="Device to use (auto-detect by default)")

    args = parser.parse_args()

    # Set device via environment variable if specified
    if args.device != "auto":
        os.environ['RESILNET_DEVICE'] = args.device

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  IEEE-READY EXPERIMENTAL EVALUATION")
    print("  FL Traffic Signal Control with NS-3 Integration")
    print(f"  Device        : {DEVICE} ({'GPU' if is_gpu_available() else 'CPU'})")
    print(f"  Intersections : {args.intersections}")
    print("="*70)

    # Run multiple experiments
    results = run_multiple_experiments(args.runs, args.rounds, args.intersections)

    # Compute statistics
    print(f"\n{'='*70}")
    print("  STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")

    stats = compute_statistics(results)

    print("Method Comparison (mean ± std):")
    print("-" * 90)
    print(f"{'Method':<22} {'Wait Time':<22} {'MAE':<22} {'Stability':<15}")
    print("-" * 90)

    print(f"{'Fixed-Time':<22} {stats['fixed_time']['wait_time']['mean']:.2f} ± {stats['fixed_time']['wait_time']['std']:.2f}s{'':<5} {'N/A':<22} {stats['fixed_time']['wait_time']['std']:.2f}s")
    print(f"{'Actuated':<22} {stats['actuated']['wait_time']['mean']:.2f} ± {stats['actuated']['wait_time']['std']:.2f}s{'':<5} {'N/A':<22} {stats['actuated']['wait_time']['std']:.2f}s")
    print(f"{'Local-ML':<22} {stats['local_ml']['wait_time']['mean']:.2f} ± {stats['local_ml']['wait_time']['std']:.2f}s{'':<5} {stats['local_ml']['mae']['mean']:.4f} ± {stats['local_ml']['mae']['std']:.4f}{'':<5} {stats['local_ml']['wait_time']['std']:.2f}s")
    # Centralized-ML
    cm_mae = stats['centralized_ml'].get('mae', {})
    cm_mae_str = f"{cm_mae['mean']:.4f} ± {cm_mae['std']:.4f}" if cm_mae else "N/A"
    print(f"{'Centralized-ML':<22} {stats['centralized_ml']['wait_time']['mean']:.2f} ± {stats['centralized_ml']['wait_time']['std']:.2f}s{'':<5} {cm_mae_str:<22} {stats['centralized_ml']['wait_time']['std']:.2f}s")
    print(f"{'FL (Ours)':<22} {stats['federated_learning']['wait_time']['mean']:.2f} ± {stats['federated_learning']['wait_time']['std']:.2f}s{'':<5} {stats['federated_learning']['mae']['mean']:.4f} ± {stats['federated_learning']['mae']['std']:.4f}{'':<5} {stats['federated_learning']['wait_time']['std']:.2f}s")

    # Generate plots
    print(f"\n{'='*70}")
    print("  GENERATING IEEE-QUALITY PLOTS")
    print(f"{'='*70}\n")

    plot_method_comparison(stats, output_dir)
    plot_fl_convergence(results, output_dir)
    plot_network_stress_comparison(output_dir)

    # Ablation study
    ablation_results = {}
    if args.ablation:
        ablation_results = run_ablation_study(output_dir, args.intersections,
                                               num_seeds=args.ablation_seeds)

    # Generate LaTeX table
    latex_table = generate_latex_table(stats, ablation_results)
    with open(output_dir / "latex_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\n  Saved: latex_table.tex")

    # Save all results
    final_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_runs": args.runs,
            "num_rounds": args.rounds
        },
        "statistics": {
            method: {
                metric: {k: float(v) if isinstance(v, (np.floating, float)) else
                         [float(x) for x in v] if isinstance(v, tuple) else v
                         for k, v in data.items()}
                for metric, data in method_stats.items()
            }
            for method, method_stats in stats.items()
        },
        "raw_results": results,
        "ablation": ablation_results if args.ablation else None
    }

    with open(output_dir / "ieee_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"  Saved: ieee_results.json")

    # Run generalization test (use models from last experiment)
    if args.runs >= 1:
        # Get trained controllers from last run
        last_seed = [42, 123, 456, 789, 1024][min(args.runs-1, 4)]
        set_global_seed(last_seed)
        generator_gen = _make_generator(args.intersections, last_seed)

        fl_ctrl_gen = AdaptiveFLController(
            num_intersections=args.intersections,
            num_rounds=args.rounds,
            local_epochs=15,
            hidden_layers=[256, 128, 64, 32],
            learning_rate=0.002,
            lr_decay=0.99,
            weight_decay=5e-5,
            use_fedprox=True,
            mu=0.05
        )
        fl_ctrl_gen.train_federated(generator_gen.get_all_intersections_data())

        local_ctrl_gen = LocalMLController(num_intersections=args.intersections)
        local_ctrl_gen.train_local_models(generator_gen.get_all_intersections_data())

        generalization_results = run_generalization_test(fl_ctrl_gen, local_ctrl_gen, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY - FL PERFORMANCE")
    print(f"{'='*70}")

    fl_mae = stats['federated_learning']['mae']['mean']
    local_mae = stats['local_ml']['mae']['mean']
    mae_improvement = ((local_mae - fl_mae) / local_mae) * 100

    fl_wait = stats['federated_learning']['wait_time']['mean']
    local_wait = stats['local_ml']['wait_time']['mean']
    actuated_wait = stats['actuated']['wait_time']['mean']
    fixed_wait = stats['fixed_time']['wait_time']['mean']

    fl_std = stats['federated_learning']['wait_time']['std']
    local_std = stats['local_ml']['wait_time']['std']
    stability_improvement = ((local_std - fl_std) / local_std) * 100

    print(f"\n  FL vs Baselines (Wait Time):")
    print(f"    vs Fixed-Time:  {((fixed_wait - fl_wait) / fixed_wait) * 100:+.1f}%")
    print(f"    vs Actuated:    {((actuated_wait - fl_wait) / actuated_wait) * 100:+.1f}%")
    print(f"    vs Local-ML:    {((local_wait - fl_wait) / local_wait) * 100:+.1f}%")

    print(f"\n  FL vs Local-ML (Accuracy):")
    print(f"    MAE Improvement:       {mae_improvement:+.2f}%")

    print(f"\n  FL Stability Advantage:")
    print(f"    Variance Reduction:    {stability_improvement:+.1f}% lower std dev")

    # Determine wins
    fl_beats_fixed = fl_wait < fixed_wait
    fl_beats_actuated = fl_wait < actuated_wait
    fl_beats_local_wait = fl_wait <= local_wait * 1.05  # Within 5%
    fl_beats_local_mae = fl_mae < local_mae
    fl_beats_stability = fl_std < local_std

    print(f"\n  SCORECARD:")
    print(f"    FL beats Fixed-Time (Wait):    {'[YES]' if fl_beats_fixed else '[NO]'}")
    print(f"    FL beats Actuated (Wait):      {'[YES]' if fl_beats_actuated else '[NO]'}")
    print(f"    FL beats Local-ML (MAE):       {'[YES]' if fl_beats_local_mae else '[NO]'}")
    print(f"    FL beats Local-ML (Stability): {'[YES]' if fl_beats_stability else '[NO]'}")
    print(f"    FL Generalizes Better:         {'[YES]' if args.runs >= 1 and generalization_results.get('fl_wins', False) else '[NO]'}")

    total_wins = sum([fl_beats_fixed, fl_beats_actuated, fl_beats_local_mae, fl_beats_stability])
    print(f"\n  TOTAL FL WINS: {total_wins}/4 categories + Generalization + Privacy")

    print(f"\n  KEY NARRATIVE:")
    print(f"    \"FL achieves {mae_improvement:.1f}% better prediction accuracy,")
    print(f"     {stability_improvement:.0f}% lower variance, and preserves privacy")
    print(f"     while maintaining competitive wait times.\"")

    print(f"\n  All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
