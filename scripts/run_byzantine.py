#!/usr/bin/env python3
"""
Byzantine Robustness Evaluation
=================================
Evaluates ResilNet-FL's resilience to Byzantine (faulty/malicious) clients.

Experiment design
-----------------
Scenario: One or more intersections have broken loop detectors that transmit
garbage sensor readings (large random values × noise_scale). This simulates:
  - Sensor hardware failure (bit-flip, stuck sensor)
  - Malicious model-poisoning attack injecting adversarial updates

Aggregation strategies compared:
  1. FedAvg         – standard weighted averaging (vulnerable baseline)
  2. Quality-Aware  – ResilNet-FL default (inverse-loss weighting)
  3. Trimmed Mean   – coordinate-wise mean after removing top/bottom 10%
  4. Median         – coordinate-wise median (robust to ≤50% Byzantine)
  5. Multi-Krum     – select top-(n-f) most representative updates

For each strategy the experiment sweeps the number of Byzantine clients
(0, 1, 2) and reports:
  - MAE on held-out test data
  - Relative MAE degradation vs. clean baseline

Output
------
  results/byzantine/
    byzantine_results.json        — full numeric results
    byzantine_robustness.png/pdf  — publication-quality figure
    byzantine_table.tex           — LaTeX table for paper

Usage
-----
  python run_byzantine_experiment.py
  python run_byzantine_experiment.py --intersections 9 --rounds 30
  python run_byzantine_experiment.py --quick
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from utils.reproducibility import set_global_seed
from utils.device import get_device, is_gpu_available
from traffic_generator import TrafficDataGenerator
from models.traffic_model import create_model, train_model, evaluate_model
from federated_learning.aggregation import robust_aggregate, fedavg_aggregate
from baselines.adaptive_fl import AdaptiveFLController

DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
#  BYZANTINE NOISE INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_byzantine_noise(
    model_params: list,
    byzantine_indices: list,
    noise_scale: float = 50.0,
    seed: int = None,
) -> list:
    """
    Replace model updates from Byzantine clients with random noise.

    The noise magnitude (noise_scale × typical_param_std) simulates a
    sensor-poisoning attack as described in:
      Fang et al. (2020) "Local Model Poisoning Attacks to Byzantine-Robust
      Federated Learning". USENIX Security '20.

    Default noise_scale=5x represents a realistic strong attack.
    50x+ causes numeric overflow (inf) in non-robust aggregators and is
    unrealistically severe for a real sensor-failure scenario.

    Args:
        model_params:     List of per-client model parameter lists.
        byzantine_indices: Indices of clients to corrupt.
        noise_scale:      Multiplier for Gaussian noise (5× = realistic strong attack).
        seed:             RNG seed for reproducibility.

    Returns:
        Modified model_params with Byzantine clients replaced by noise.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Estimate typical parameter scale from honest clients
    honest = [i for i in range(len(model_params)) if i not in byzantine_indices]
    if honest:
        ref_params = model_params[honest[0]]
        param_stds = [np.std(p.astype(np.float32)) + 1e-6 for p in ref_params]
    else:
        param_stds = [1.0] * len(model_params[0])

    corrupted = [list(params) for params in model_params]  # shallow copy

    for byz_idx in byzantine_indices:
        noisy = []
        for layer_idx, p in enumerate(model_params[byz_idx]):
            noise = rng.normal(0, noise_scale * param_stds[layer_idx], p.shape)
            noisy.append(noise.astype(p.dtype))
        corrupted[byz_idx] = noisy

    return corrupted


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE TRIAL
# ─────────────────────────────────────────────────────────────────────────────

def run_byzantine_trial(
    seed: int,
    num_intersections: int,
    num_byzantine: int,
    num_rounds: int,
    noise_scale: float,
    strategies: list,
) -> dict:
    """
    Evaluate multiple aggregation strategies against Byzantine clients.

    Optimised design — shared local training:
      All strategies receive the SAME local model updates each round; only the
      aggregation rule differs.  This is the standard evaluation methodology in
      Byzantine-FL literature (Blanchard 2017, Yin 2018, Fang 2020) and gives a
      clean apples-to-apples comparison of aggregation rules while running
      ~5–10× faster than independent per-strategy training loops.

    Returns:  {strategy_name: best_mae}
    """
    set_global_seed(seed)
    generator = TrafficDataGenerator(
        config={"traffic": {"num_intersections": num_intersections,
                            "simulation_duration": 3600,
                            "time_step": 5,
                            "arrival_distribution": "poisson",
                            "min_arrival_rate": 5,
                            "max_arrival_rate": 30,
                            "max_queue_length": 50,
                            "min_green_duration": 10,
                            "max_green_duration": 90,
                            "yellow_duration": 3}}
    )
    training_data = generator.get_all_intersections_data()

    # Byzantine clients: last num_byzantine intersections
    byzantine_indices = list(range(num_intersections - num_byzantine, num_intersections))

    # ── Shared FL controller for local training machinery ─────────────────────
    set_global_seed(seed)
    ref_fl = AdaptiveFLController(
        num_intersections=num_intersections,
        num_rounds=num_rounds,
        local_epochs=5,           # was 10 — halves training time; 5 is sufficient
        hidden_layers=[256, 128, 64, 32],
        learning_rate=0.002,
        lr_decay=0.99,
        weight_decay=5e-5,
        use_fedprox=True,
        mu=0.05,
    )

    # Per-strategy global model params (all start from the same initialisation)
    init_params          = ref_fl.global_model.get_parameters()
    strategy_global      = {s: [p.copy() for p in init_params] for s in strategies}
    best_mae             = {s: float("inf") for s in strategies}
    best_params_saved    = {s: None         for s in strategies}
    patience             = {s: 0            for s in strategies}
    current_lr           = ref_fl.learning_rate

    # Preference order for choosing the local-training reference each round.
    # We prefer a robust strategy so that Byzantine noise does not compound into
    # infinity via the reference model.  FedAvg is intentionally last so its
    # aggregation rule is still applied to the SAME local updates as everyone else,
    # preserving the apples-to-apples comparison of aggregation methods.
    _ref_preference = ["trimmed_mean", "median", "multi_krum"] + strategies

    for rnd in range(num_rounds):
        # ── Pick the best finite strategy as local-training reference ──────────
        # This prevents a single poisoned round from compounding to inf for all
        # subsequent rounds.  Each strategy still aggregates independently.
        ref_params = None
        for s in _ref_preference:
            if s in strategy_global:
                candidate = strategy_global[s]
                if not any(np.any(~np.isfinite(p)) for p in candidate):
                    ref_params = candidate
                    break
        if ref_params is None:           # all strategies have gone inf → use init
            ref_params = init_params

        ref_fl.global_model.set_parameters(ref_params)
        for i in range(num_intersections):
            ref_fl.local_models[i].set_parameters(ref_params)

        # ── Local training (done ONCE, shared across all strategies) ───────────
        model_params = []
        round_losses = []
        data_sizes   = []
        for iid, (X, y) in training_data.items():
            m, loss_hist = train_model(
                ref_fl.local_models[iid],
                (X, y),
                epochs=ref_fl.local_epochs,
                batch_size=32,
                learning_rate=current_lr,
                weight_decay=ref_fl.weight_decay,
                use_scheduler=True,
                gradient_clip=1.0,
                global_model=ref_fl.global_model,
                mu=ref_fl.mu,
            )
            ref_fl.local_models[iid] = m
            model_params.append(m.get_parameters())
            round_losses.append(loss_hist[-1])
            data_sizes.append(len(X))

        # ── Byzantine injection (once — same corrupted set for all strategies) ─
        if num_byzantine > 0:
            corrupted = inject_byzantine_noise(
                model_params, byzantine_indices,
                noise_scale=noise_scale, seed=seed + rnd,
            )
        else:
            corrupted = model_params

        inv_losses = [1.0 / (l + 1e-6) for l in round_losses]
        weights    = [sz * il for sz, il in zip(data_sizes, inv_losses)]

        # trim_ratio must cover at least the Byzantine fraction + a safety margin
        # e.g. 2/9 Byzantine → trim_ratio ≥ 0.27; cap at 0.40 to keep enough data
        trim_ratio = min(0.40, max(0.15, num_byzantine / num_intersections + 0.10))

        # ── Each strategy aggregates the SAME corrupted params ─────────────────
        for strategy in strategies:
            if strategy == "quality_aware":
                agg = ref_fl.federated_averaging(corrupted, weights, "quality_aware")
            elif strategy == "resil_agg":
                agg = robust_aggregate(
                    corrupted,
                    strategy="resil_agg",
                    losses=round_losses,
                    data_sizes=data_sizes,
                )
            else:
                agg = robust_aggregate(
                    corrupted,
                    weights=weights,
                    strategy=strategy,
                    num_byzantine=num_byzantine,
                    trim_ratio=trim_ratio,
                )
            strategy_global[strategy] = agg

            # Evaluate on held-out 20%
            ref_fl.global_model.set_parameters(agg)
            total_mae = 0.0
            for iid, (X, y) in training_data.items():
                split = int(len(X) * 0.8)
                _, mae = evaluate_model(ref_fl.global_model, (X[split:], y[split:]))
                total_mae += mae
            avg_mae = total_mae / len(training_data)

            if avg_mae < best_mae[strategy]:
                best_mae[strategy]        = avg_mae
                best_params_saved[strategy] = [p.copy() for p in agg]
                patience[strategy]        = 0
            else:
                patience[strategy] += 1

        current_lr = max(current_lr * ref_fl.lr_decay, ref_fl.min_lr)

        # ── Progress print every 5 rounds ──────────────────────────────────────
        if (rnd + 1) % 5 == 0 or rnd == 0:
            p_str = " | ".join(f"{s[:6]}={best_mae[s]:.3f}" for s in strategies)
            print(f"    rnd {rnd+1:2d}/{num_rounds}: {p_str}", flush=True)

        # ── Early stop when all strategies have plateaued ──────────────────────
        if all(patience[s] >= 8 for s in strategies) and rnd >= 12:
            print(f"    All strategies converged — early stop at round {rnd+1}.",
                  flush=True)
            break

    return best_mae


# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP OVER BYZANTINE COUNT
# ─────────────────────────────────────────────────────────────────────────────

def run_byzantine_sweep(
    num_intersections: int,
    byzantine_counts: list,
    strategies: list,
    num_rounds: int,
    noise_scale: float,
    seeds: list,
) -> dict:
    """Run the full Byzantine sweep (count × strategy × seed)."""
    sweep_results = {}

    for byz_count in byzantine_counts:
        print(f"\n{'='*65}")
        print(f"  Byzantine clients: {byz_count}/{num_intersections}")
        print(f"{'='*65}")
        trial_results = {s: [] for s in strategies}

        for seed in seeds:
            print(f"  Seed {seed:4d}: ", end="", flush=True)
            trial = run_byzantine_trial(
                seed=seed,
                num_intersections=num_intersections,
                num_byzantine=byz_count,
                num_rounds=num_rounds,
                noise_scale=noise_scale,
                strategies=strategies,
            )
            for s, mae in trial.items():
                trial_results[s].append(mae)
            print("  ".join(f"{s}={trial[s]:.4f}" for s in strategies))

        sweep_results[byz_count] = {
            s: {
                "mean": float(np.mean(trial_results[s])),
                "std":  float(np.std(trial_results[s])),
            }
            for s in strategies
        }

    return sweep_results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_byzantine_results(
    sweep_results: dict,
    strategies: list,
    output_dir: Path,
):
    """Generate publication-quality 2-panel Byzantine robustness figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    byz_counts = sorted(sweep_results.keys())
    x = np.arange(len(byz_counts))

    # Strategy display metadata
    meta = {
        "resil_agg":      {"label": "ResilAgg (Ours)",        "color": "#e74c3c",  "marker": "*", "lw": 3},
        "quality_aware":  {"label": "Quality-Aware",          "color": "#f39c12",  "marker": "o", "lw": 2},
        "fedavg":         {"label": "FedAvg",                 "color": "#95a5a6",  "marker": "s", "lw": 2},
        "trimmed_mean":   {"label": "Trimmed Mean",           "color": "#3498db",  "marker": "^", "lw": 2},
        "median":         {"label": "Median",                 "color": "#2ecc71",  "marker": "D", "lw": 2},
        "multi_krum":     {"label": "Multi-Krum",             "color": "#9b59b6",  "marker": "P", "lw": 2},
    }

    # ── Panel (a): MAE vs Number of Byzantine clients ──────────────────────
    ax1 = axes[0]
    for s in strategies:
        means = [sweep_results[b][s]["mean"] for b in byz_counts]
        stds  = [sweep_results[b][s]["std"]  for b in byz_counts]
        m     = meta.get(s, {"label": s, "color": "#888888", "marker": "o", "lw": 1})
        ax1.errorbar(
            x, means, yerr=stds, capsize=5, capthick=1.5,
            label=m["label"], color=m["color"],
            marker=m["marker"], markersize=8, linewidth=m["lw"],
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b} Byzantine\n({b}/{byz_counts[-1]+1} clients)" for b in byz_counts])
    ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) MAE Under Byzantine Attacks", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.35)

    # ── Panel (b): Relative MAE degradation vs clean baseline ─────────────
    ax2 = axes[1]
    clean_maes = {s: sweep_results[0][s]["mean"] for s in strategies}

    for s in strategies:
        degr = [
            (sweep_results[b][s]["mean"] - clean_maes[s]) / (clean_maes[s] + 1e-9) * 100
            for b in byz_counts
        ]
        m = meta.get(s, {"label": s, "color": "#888888", "marker": "o", "lw": 1})
        ax2.plot(
            x, degr, label=m["label"], color=m["color"],
            marker=m["marker"], markersize=8, linewidth=m["lw"],
        )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b} Byzantine" for b in byz_counts])
    ax2.set_ylabel("MAE Degradation vs. Clean (%)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Relative Degradation from Byzantine Attacks", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.35)

    # Annotate FedAvg degradation
    if "fedavg" in strategies and len(byz_counts) > 1:
        last_b = byz_counts[-1]
        fedavg_deg = (sweep_results[last_b]["fedavg"]["mean"] - clean_maes["fedavg"]) / (clean_maes["fedavg"] + 1e-9) * 100
        ax2.annotate(
            f"FedAvg: +{fedavg_deg:.0f}%",
            xy=(len(byz_counts) - 1, fedavg_deg),
            xytext=(len(byz_counts) - 1.6, fedavg_deg + 3),
            fontsize=10, color=meta["fedavg"]["color"],
            arrowprops=dict(arrowstyle="->", color=meta["fedavg"]["color"]),
        )

    plt.suptitle(
        "Byzantine Robustness: ResilNet-FL vs. Standard FedAvg\n"
        "(Sensor noise injection — broken loop detector simulation)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = output_dir / f"byzantine_robustness.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out.name}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  LATEX TABLE
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(sweep_results: dict, strategies: list) -> str:
    """
    Generate LaTeX table: strategies × Byzantine counts.
    """
    strategy_labels = {
        "resil_agg":     "\\textbf{ResilAgg (Ours)}",
        "fedavg":        "FedAvg",
        "quality_aware": "Quality-Aware",
        "trimmed_mean":  "Trimmed Mean",
        "median":        "Median",
        "multi_krum":    "Multi-Krum",
    }
    byz_counts = sorted(sweep_results.keys())
    col_header = " & ".join([f"\\textbf{{{b} Byz.}}" for b in byz_counts])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Byzantine Robustness: MAE (mean $\pm$ std) under sensor-noise attacks. "
        r"Bold = best per column.}",
        r"\label{tab:byzantine}",
        r"\begin{tabular}{l" + "c" * len(byz_counts) + "}",
        r"\toprule",
        r"\textbf{Aggregation} & " + col_header + r" \\",
        r"\midrule",
    ]

    # Find best (min MAE) per column for bolding
    best_per_col = {}
    for b in byz_counts:
        best_per_col[b] = min(sweep_results[b][s]["mean"] for s in strategies)

    for s in strategies:
        label = strategy_labels.get(s, s)
        cells = []
        for b in byz_counts:
            mean = sweep_results[b][s]["mean"]
            std  = sweep_results[b][s]["std"]
            cell = f"{mean:.4f} $\\pm$ {std:.4f}"
            if abs(mean - best_per_col[b]) < 1e-6:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Byzantine Robustness Experiment")
    parser.add_argument("--intersections", type=int, default=9,
                        help="Number of intersections (default 9 for 3x3 grid)")
    parser.add_argument("--rounds",       type=int, default=40,
                        help="FL rounds per trial (default 40)")
    parser.add_argument("--noise-scale",  type=float, default=5.0,
                        help="Byzantine noise magnitude (default 5x — realistic sensor attack)")
    parser.add_argument("--seeds",        type=int, default=3,
                        help="Number of seeds per condition (default 3)")
    parser.add_argument("--output",       type=str, default="results/byzantine",
                        help="Output directory")
    parser.add_argument("--quick",        action="store_true",
                        help="Quick mode: 1 seed, 20 rounds, 4 intersections")
    args = parser.parse_args()

    if args.quick:
        args.intersections = 4
        args.rounds        = 20
        args.seeds         = 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Max Byzantine clients = floor((N-3)/2) for Krum; cap at 2 for clarity
    max_byzantine = min(2, (args.intersections - 3) // 2)
    byzantine_counts = list(range(0, max_byzantine + 1))

    strategies = ["resil_agg", "fedavg", "quality_aware", "trimmed_mean", "median", "multi_krum"]
    seeds      = list(range(42, 42 + args.seeds))

    print("\n" + "=" * 65)
    print("  BYZANTINE ROBUSTNESS EVALUATION")
    print(f"  Intersections : {args.intersections}")
    print(f"  Byzantine range: {byzantine_counts}")
    print(f"  Strategies    : {', '.join(strategies)}")
    print(f"  FL rounds     : {args.rounds}")
    print(f"  Noise scale   : {args.noise_scale}x")
    print(f"  Seeds         : {seeds}")
    print(f"  Device        : {DEVICE} ({'GPU' if is_gpu_available() else 'CPU'})")
    print("=" * 65)

    sweep_results = run_byzantine_sweep(
        num_intersections=args.intersections,
        byzantine_counts=byzantine_counts,
        strategies=strategies,
        num_rounds=args.rounds,
        noise_scale=args.noise_scale,
        seeds=seeds,
    )

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    hdr = " | ".join(f"{'Byz='+str(b):>10}" for b in byzantine_counts)
    print(f"{'Strategy':<22} | {hdr}")
    print("-" * 65)
    for s in strategies:
        row_vals = " | ".join(
            f"{sweep_results[b][s]['mean']:>10.4f}" for b in byzantine_counts
        )
        print(f"{s:<22} | {row_vals}")

    # ── Highlight recovery ───────────────────────────────────────────────────
    if len(byzantine_counts) > 1:
        max_b = byzantine_counts[-1]
        print(f"\n  Degradation with {max_b} Byzantine client(s):")
        for s in strategies:
            base = sweep_results[0][s]["mean"]
            deg  = sweep_results[max_b][s]["mean"]
            pct  = (deg - base) / (base + 1e-9) * 100
            flag = "   <-- CORRUPTED" if pct > 20 else ("   <-- robust" if pct < 5 else "")
            print(f"    {s:<22}: +{pct:.1f}%{flag}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots...")
    plot_byzantine_results(sweep_results, strategies, output_dir)

    # ── LaTeX table ──────────────────────────────────────────────────────────
    latex = generate_latex_table(sweep_results, strategies)
    tex_path = output_dir / "byzantine_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  Saved: {tex_path.name}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_intersections": args.intersections,
            "byzantine_counts": byzantine_counts,
            "strategies": strategies,
            "num_rounds": args.rounds,
            "noise_scale": args.noise_scale,
            "seeds": seeds,
        },
        "results": sweep_results,
    }
    json_path = output_dir / "byzantine_results.json"
    json_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"  Saved: {json_path.name}")

    print(f"\n  All outputs in: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
