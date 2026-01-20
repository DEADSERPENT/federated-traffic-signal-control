#!/usr/bin/env python3
"""
IEEE-Ready Experimental Evaluation
===================================

This script runs comprehensive experiments for IEEE publication:
1. Multiple runs with different seeds (statistical significance)
2. Generates publication-quality plots
3. Ablation study
4. Statistical analysis (mean ± std, confidence intervals)

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from utils.reproducibility import set_global_seed
from traffic_generator import TrafficDataGenerator
from baselines.fixed_time import FixedTimeController
from baselines.local_ml import LocalMLController
from baselines.adaptive_fl import AdaptiveFLController


def run_single_experiment(seed: int, num_rounds: int = 50) -> Dict:
    """Run a single experiment with given seed."""
    set_global_seed(seed)

    # Initialize
    generator = TrafficDataGenerator()
    training_data = generator.get_all_intersections_data()

    results = {"seed": seed}

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

    # Local-ML
    set_global_seed(seed)
    local_controller = LocalMLController(num_intersections=4)
    local_results = local_controller.run_simulation(
        generator.intersections, generator, duration=1800
    )
    results["local_ml"] = {
        "wait_time": local_results["avg_waiting_time"],
        "queue_length": local_results["avg_queue_length"],
        "mae": local_results["mae"]
    }

    # Federated Learning
    set_global_seed(seed)
    fl_controller = AdaptiveFLController(
        num_intersections=4,
        num_rounds=num_rounds,
        local_epochs=15,
        hidden_layers=[256, 128, 64, 32],
        learning_rate=0.002,
        lr_decay=0.99,
        weight_decay=5e-5
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


def run_multiple_experiments(num_runs: int = 5, num_rounds: int = 50) -> List[Dict]:
    """Run multiple experiments with different seeds."""
    seeds = [42, 123, 456, 789, 1024][:num_runs]
    all_results = []

    print(f"\n{'='*70}")
    print(f"  RUNNING {num_runs} EXPERIMENTS FOR STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")

    for i, seed in enumerate(seeds):
        print(f"\n--- Experiment {i+1}/{num_runs} (seed={seed}) ---")
        result = run_single_experiment(seed, num_rounds)
        all_results.append(result)

        print(f"  Fixed-Time: Wait={result['fixed_time']['wait_time']:.2f}s")
        print(f"  Local-ML:   Wait={result['local_ml']['wait_time']:.2f}s, MAE={result['local_ml']['mae']:.4f}")
        print(f"  FL:         Wait={result['federated_learning']['wait_time']:.2f}s, MAE={result['federated_learning']['mae']:.4f}")

    return all_results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean, std, and confidence intervals."""
    stats_dict = {}

    for method in ["fixed_time", "local_ml", "federated_learning"]:
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
        if method != "fixed_time":
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["Fixed-Time", "Local-ML", "Federated Learning"]
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

    # Wait Time comparison
    ax1 = axes[0]
    wait_means = [
        stats["fixed_time"]["wait_time"]["mean"],
        stats["local_ml"]["wait_time"]["mean"],
        stats["federated_learning"]["wait_time"]["mean"]
    ]
    wait_stds = [
        stats["fixed_time"]["wait_time"]["std"],
        stats["local_ml"]["wait_time"]["std"],
        stats["federated_learning"]["wait_time"]["std"]
    ]

    bars1 = ax1.bar(methods, wait_means, yerr=wait_stds, capsize=5,
                    color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax1.set_ylabel('Average Waiting Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Waiting Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(wait_means) * 1.3)

    # Add value labels
    for bar, mean, std in zip(bars1, wait_means, wait_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE comparison (only ML methods)
    ax2 = axes[1]
    mae_methods = ["Local-ML", "Federated Learning"]
    mae_colors = ['#3498db', '#e74c3c']
    mae_means = [
        stats["local_ml"]["mae"]["mean"],
        stats["federated_learning"]["mae"]["mean"]
    ]
    mae_stds = [
        stats["local_ml"]["mae"]["std"],
        stats["federated_learning"]["mae"]["std"]
    ]

    bars2 = ax2.bar(mae_methods, mae_means, yerr=mae_stds, capsize=5,
                    color=mae_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(mae_means) * 1.3)

    # Add value labels
    for bar, mean, std in zip(bars2, mae_means, mae_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
                f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add winner annotation
    if mae_means[1] < mae_means[0]:
        improvement = ((mae_means[0] - mae_means[1]) / mae_means[0]) * 100
        ax2.annotate(f'FL wins by {improvement:.1f}%',
                    xy=(1, mae_means[1]), xytext=(0.5, mae_means[0] * 0.7),
                    fontsize=11, color='#e74c3c', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'))

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


def run_ablation_study(output_dir: Path) -> Dict:
    """Run ablation study comparing different FL configurations."""
    print(f"\n{'='*70}")
    print("  ABLATION STUDY")
    print(f"{'='*70}\n")

    ablation_results = {}
    seed = 42

    configs = [
        {"name": "FL-Small", "hidden": [64, 32], "epochs": 5, "rounds": 30},
        {"name": "FL-Medium", "hidden": [128, 64, 32], "epochs": 10, "rounds": 30},
        {"name": "FL-Large (Ours)", "hidden": [256, 128, 64, 32], "epochs": 15, "rounds": 50},
        {"name": "FL-NoScheduler", "hidden": [256, 128, 64, 32], "epochs": 15, "rounds": 50, "lr_decay": 1.0},
    ]

    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        set_global_seed(seed)

        generator = TrafficDataGenerator()

        fl_controller = AdaptiveFLController(
            num_intersections=4,
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

        ablation_results[config["name"]] = {
            "wait_time": fl_results["avg_waiting_time"],
            "mae": fl_results["mae"],
            "best_mae": fl_controller.best_mae,
            "config": config
        }

        print(f"  Wait: {fl_results['avg_waiting_time']:.2f}s, MAE: {fl_results['mae']:.4f}")

    # Plot ablation results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(ablation_results.keys())
    maes = [ablation_results[n]["mae"] for n in names]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6']

    bars = ax.bar(names, maes, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    # Highlight best
    best_idx = np.argmin(maes)
    bars[best_idx].set_edgecolor('#27ae60')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: FL Configuration Impact', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(maes) * 1.2)

    # Add value labels
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ieee_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ieee_ablation_study.pdf', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: ieee_ablation_study.png/pdf")

    return ablation_results


def generate_latex_table(stats: Dict, ablation: Dict) -> str:
    """Generate LaTeX table for IEEE paper."""
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Traffic Signal Control Methods}
\\label{tab:results}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Method} & \\textbf{Wait Time (s)} & \\textbf{MAE} & \\textbf{Improvement} \\\\
\\midrule
"""

    fixed_wait = stats["fixed_time"]["wait_time"]
    local_wait = stats["local_ml"]["wait_time"]
    local_mae = stats["local_ml"]["mae"]
    fl_wait = stats["federated_learning"]["wait_time"]
    fl_mae = stats["federated_learning"]["mae"]

    latex += f"Fixed-Time & {fixed_wait['mean']:.2f} $\\pm$ {fixed_wait['std']:.2f} & N/A & Baseline \\\\\n"
    latex += f"Local-ML & {local_wait['mean']:.2f} $\\pm$ {local_wait['std']:.2f} & {local_mae['mean']:.4f} $\\pm$ {local_mae['std']:.4f} & - \\\\\n"

    wait_imp = ((local_wait['mean'] - fl_wait['mean']) / local_wait['mean']) * 100
    mae_imp = ((local_mae['mean'] - fl_mae['mean']) / local_mae['mean']) * 100
    latex += f"\\textbf{{FL (Ours)}} & \\textbf{{{fl_wait['mean']:.2f} $\\pm$ {fl_wait['std']:.2f}}} & \\textbf{{{fl_mae['mean']:.4f} $\\pm$ {fl_mae['std']:.4f}}} & \\textbf{{{mae_imp:.1f}\\%}} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="IEEE-Ready Experiments")
    parser.add_argument("--runs", type=int, default=5, help="Number of experiment runs")
    parser.add_argument("--rounds", type=int, default=50, help="FL rounds per run")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--output", type=str, default="results/ieee", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  IEEE-READY EXPERIMENTAL EVALUATION")
    print("  FL Traffic Signal Control with NS-3 Integration")
    print("="*70)

    # Run multiple experiments
    results = run_multiple_experiments(args.runs, args.rounds)

    # Compute statistics
    print(f"\n{'='*70}")
    print("  STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")

    stats = compute_statistics(results)

    print("Method Comparison (mean ± std):")
    print("-" * 60)
    print(f"{'Method':<20} {'Wait Time':<20} {'MAE':<20}")
    print("-" * 60)

    print(f"{'Fixed-Time':<20} {stats['fixed_time']['wait_time']['mean']:.2f} ± {stats['fixed_time']['wait_time']['std']:.2f}s{'':<5} N/A")
    print(f"{'Local-ML':<20} {stats['local_ml']['wait_time']['mean']:.2f} ± {stats['local_ml']['wait_time']['std']:.2f}s{'':<5} {stats['local_ml']['mae']['mean']:.4f} ± {stats['local_ml']['mae']['std']:.4f}")
    print(f"{'FL (Ours)':<20} {stats['federated_learning']['wait_time']['mean']:.2f} ± {stats['federated_learning']['wait_time']['std']:.2f}s{'':<5} {stats['federated_learning']['mae']['mean']:.4f} ± {stats['federated_learning']['mae']['std']:.4f}")

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
        ablation_results = run_ablation_study(output_dir)

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

    # Print summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")

    fl_mae = stats['federated_learning']['mae']['mean']
    local_mae = stats['local_ml']['mae']['mean']
    mae_improvement = ((local_mae - fl_mae) / local_mae) * 100

    fl_wait = stats['federated_learning']['wait_time']['mean']
    local_wait = stats['local_ml']['wait_time']['mean']
    wait_improvement = ((local_wait - fl_wait) / local_wait) * 100

    print(f"\n  FL vs Local-ML:")
    print(f"    MAE Improvement:       {mae_improvement:+.2f}%")
    print(f"    Wait Time Improvement: {wait_improvement:+.2f}%")
    print(f"\n  FL WINS: {'YES' if fl_mae < local_mae else 'NO'} (MAE)")
    print(f"  FL WINS: {'YES' if fl_wait <= local_wait else 'NO'} (Wait Time)")

    print(f"\n  All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
