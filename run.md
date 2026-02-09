# ResilNet-FL: Complete Run Guide

## Setup

```bash
# 1. Activate virtual environment
# Windows Command Prompt
venv\Scripts\activate

# Windows Git Bash / MSYS2
source venv/Scripts/activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt
```

**Detected hardware:** PyTorch 2.9.1+cu130 | NVIDIA RTX 3050 4 GB | CUDA 13.0
GPU is auto-detected — no extra flags needed.

---

## Project Structure

```
TRAFFIC SIGNALS/
├── run_ieee_experiments.py        # IEEE paper results (statistical, 5 runs)
├── run_comprehensive.py           # Full experiment suite + dashboard
├── run_publication_experiments.py  # Extended publication suite (privacy, stats)
├── run_simulation.py              # Basic traffic data generation
├── run_demo.py                    # System walkthrough demo
├── run_with_ns3.py                # FL + NS-3 network simulation
├── run_cloudsim.py                # Edge/cloud computing simulation
├── run_fl_server.py               # Distributed FL server (Flower)
├── run_fl_client.py               # Distributed FL client (Flower)
├── generate_radar_plot.py         # Radar trade-off chart for paper
├── generate_architecture_diagram.py # System architecture diagram
├── generate_comparison.py         # Before/after optimization plot
├── generate_gen_plot.py           # Generalization bar chart
├── config/config.yaml             # Main configuration file
├── src/                           # Source code
│   ├── baselines/                 #   Fixed-Time, Actuated, Local-ML, FL controllers
│   ├── models/                    #   TrafficSignalModel, LSTM, GRU
│   ├── traffic_generator/         #   Synthetic traffic (Poisson arrivals)
│   ├── federated_learning/        #   Flower server/client, aggregation strategies
│   ├── network_simulation/        #   Network layer abstraction
│   ├── cloudsim_python/           #   Edge/cloud simulation
│   ├── experiments/               #   Comprehensive, stress, scalability runners
│   └── utils/                     #   Device, visualization, privacy, stats, config
├── ns3_simulation/                # NS-3 bridge (Windows <-> WSL via ZeroMQ)
├── paper/                         # LaTeX paper (ResilNet_FL_IEEE_Paper.tex)
├── data/                          # Generated CSV data
└── results/                       # All output plots, JSON, LaTeX tables
```

---

## Experiment Runners

### 1. IEEE Publication Experiments (Recommended First Run)

Runs 5 independent trials (seeds 42, 123, 456, 789, 1024), computes mean/std/CI, generates publication plots, LaTeX table, and generalization test.

```bash
# Full run — 5 trials, 50 FL rounds each (~15-20 min)
python run_ieee_experiments.py

# Full run + ablation study (architecture comparison)
python run_ieee_experiments.py --runs 5 --rounds 50 --ablation

# Quick test — 1 trial, 20 rounds (~3 min)
python run_ieee_experiments.py --runs 1 --rounds 20

# Force CPU
python run_ieee_experiments.py --device cpu

# Custom output directory
python run_ieee_experiments.py --output results/my_run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 5 | Number of independent trials |
| `--rounds N` | 50 | FL rounds per trial |
| `--ablation` | off | Run ablation study (FL-Small, FL-Medium, FL-Large, FL-NoScheduler) |
| `--output DIR` | `results/ieee` | Output directory |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

**Output** `results/ieee/`:
```
ieee_method_comparison.png/pdf   — Bar chart: Fixed-Time vs Actuated vs Local-ML vs FL
ieee_fl_convergence.png/pdf      — MAE convergence across FL rounds (all runs + mean)
ieee_ablation_study.png/pdf      — Architecture comparison (with --ablation)
ieee_network_stress.png/pdf      — MAE under network stress (if NS-3 data exists)
ieee_results.json                — All raw results + statistics
latex_table.tex                  — Ready to paste into LaTeX paper
generalization_test.json         — FL vs Local-ML on unseen traffic (seed 9999)
```

---

### 2. Comprehensive Experiments

Baseline comparisons + network stress tests + scalability (2/4/6/8 clients) + summary dashboard.

```bash
# Full suite (~10-15 min)
python run_comprehensive.py

# Quick mode — skip scalability tests (~5 min)
python run_comprehensive.py --quick

# With specific seed and device
python run_comprehensive.py --seed 42 --device cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | Skip scalability tests |
| `--seed N` | 42 | Random seed |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

**Output** `results/comprehensive/`:
```
summary_dashboard.png      — All results in one image
fl_convergence.png         — Training convergence curve
method_comparison.png      — Baseline comparisons
network_stress.png         — Latency/packet-loss resilience
scalability.png            — 2-8 client scaling (full mode only)
experiment_report.txt      — Text summary
results.json               — All data
```

---

### 3. Publication Experiments (Extended)

Full paper-ready suite: statistical significance tests, privacy quantification, communication efficiency, live data robustness, network stress.

```bash
# Full run — 5 trials, 100 rounds (~25-30 min)
python run_publication_experiments.py

# Quick mode — 3 trials, 50 rounds (~10 min)
python run_publication_experiments.py --quick

# Custom
python run_publication_experiments.py --runs 5 --rounds 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 5 | Number of trials |
| `--rounds N` | 100 | FL rounds per trial |
| `--quick` | off | 3 runs, 50 rounds |

**Output** `results/publication/`:
```
complete_results.json       — All experiment data
publication_report.txt      — Full text report
statistical_analysis.txt    — Statistical test results
statistical_table.tex       — LaTeX table with significance
privacy_analysis.txt        — Privacy metrics report
privacy_comparison.json     — Centralized vs FL privacy
```

---

### 4. Basic Simulation

Generates traffic data and per-intersection statistics. Good for verifying the setup works.

```bash
python run_simulation.py
```

No CLI arguments. Uses `config/config.yaml`.

**Output:**
- `data/traffic_simulation.csv` — Raw traffic data (queue lengths, wait times, throughput)
- `results/traffic_metrics.png` — Traffic visualization
- Console: per-intersection summary stats

---

### 5. Demo Mode

Step-by-step walkthrough: traffic generation, model training, evaluation, network simulation.

```bash
python run_demo.py
```

No CLI arguments. Uses `config/config.yaml`.

**Output:**
- `data/demo_simulation.csv` — Demo traffic data
- Console: full system walkthrough

---

## Network Simulation

### 6. FL + NS-3 Integration

Runs FL training with realistic V2I (Vehicle-to-Infrastructure) network simulation using IEEE 802.11p/DSRC parameters.

```bash
# With NS-3 (requires WSL bridge server running — see below)
python run_with_ns3.py

# Specific network scenario
python run_with_ns3.py --scenario degraded

# Without NS-3 (uses built-in statistical simulator)
python run_with_ns3.py --no-ns3

# Network stress test — runs all 5 scenarios sequentially
python run_with_ns3.py --stress-test

# Custom FL rounds
python run_with_ns3.py --rounds 100 --scenario stressed
```

| Flag | Default | Description |
|------|---------|-------------|
| `--no-ns3` | off | Skip NS-3, use built-in network sim |
| `--scenario` | `normal` | `ideal`, `normal`, `degraded`, `stressed`, `extreme` |
| `--rounds N` | 50 | FL training rounds |
| `--stress-test` | off | Run all 5 scenarios |

**Network scenarios:**

| Scenario | Latency | Jitter | Packet Loss | Bandwidth |
|----------|---------|--------|-------------|-----------|
| `ideal` | 5 ms | +/- 2 ms | 0% | 54 Mbps |
| `normal` | 15 ms | +/- 5 ms | 1% | 27 Mbps |
| `degraded` | 50 ms | +/- 15 ms | 5% | 12 Mbps |
| `stressed` | 100 ms | +/- 30 ms | 10% | 6 Mbps |
| `extreme` | 200 ms | +/- 50 ms | 20% | 3 Mbps |

**NS-3 bridge setup (one-time):**
```bash
# Terminal 1 — WSL
wsl python3 ns3_simulation/ns3_bridge_server.py

# Terminal 2 — Windows
python run_with_ns3.py
```

**Output** `results/ns3_integrated/`:
```
ns3_results.json             — Per-round metrics with network stats
```

**Output** `results/ns3_stress/` (with `--stress-test`):
```
ideal/ns3_results.json
normal/ns3_results.json
degraded/ns3_results.json
stressed/ns3_results.json
extreme/ns3_results.json
stress_test_summary.json     — Summary across all scenarios
```

---

### 7. CloudSim Edge Computing

Simulates edge/cloud offloading: VMs, edge servers, cloudlets, task scheduling for FL aggregation.

```bash
python run_cloudsim.py
```

No CLI arguments. Output: console analysis.

---

## Distributed FL (Flower Framework)

### 8. FL Server + Clients (Multi-Terminal)

Run actual distributed federated learning with separate server and client processes.

**Terminal 1 — Start server:**
```bash
python run_fl_server.py --rounds 50 --min-clients 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--address` | `0.0.0.0:8080` | Server bind address |
| `--rounds N` | 10 | Number of FL rounds |
| `--min-clients N` | 2 | Minimum clients before training starts |
| `--config PATH` | `config/config.yaml` | Config file |

**Terminals 2-5 — Start clients (one per intersection):**
```bash
python run_fl_client.py --server localhost:8080 --intersection 0
python run_fl_client.py --server localhost:8080 --intersection 1
python run_fl_client.py --server localhost:8080 --intersection 2
python run_fl_client.py --server localhost:8080 --intersection 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `localhost:8080` | Server address |
| `--intersection N` | 0 | Intersection ID (0-3) |
| `--config PATH` | `config/config.yaml` | Config file |

---

## Plot Generators (for Paper Figures)

These scripts generate standalone figures from existing results or hardcoded data.

```bash
# Radar chart — multi-objective trade-off comparison
python generate_radar_plot.py
# Output: results/ieee/ieee_tradeoff_radar.png/pdf

# System architecture diagram — 3-layer block diagram
python generate_architecture_diagram.py
# Output: results/ieee/system_architecture.png/pdf

# Before/after optimization comparison
python generate_comparison.py
# Output: results/comprehensive/optimization_comparison.png

# Generalization bar chart (FL vs Local-ML on unseen data)
python generate_gen_plot.py
# Output: results/ieee/ieee_generalization.png
```

---

## Module Tests

Verify individual components work correctly.

```bash
# GPU/CPU auto-detection — prints device name and memory
python src/utils/device.py

# Traffic model — creates model, trains, evaluates on dummy data
python src/models/traffic_model.py

# Byzantine-robust aggregation — tests FedAvg, Krum, TrimmedMean, Median
python src/federated_learning/aggregation.py

# Trace-driven network simulation — IEEE 802.11p, LTE-V2X, 5G-V2X params
python ns3_simulation/network_traces.py

# NS-3 bridge client — connection test (needs server running)
python ns3_simulation/ns3_bridge_client.py

# Visualization functions — generates test plots
python src/utils/visualization.py
```

---

## Device Selection

GPU is auto-detected by default across all scripts.

```bash
# Auto-detect (default) — uses RTX 3050 if available
python run_ieee_experiments.py

# Force CUDA GPU
python run_ieee_experiments.py --device cuda

# Force CPU
python run_ieee_experiments.py --device cpu

# Environment variable method (works with ALL scripts)
set RESILNET_DEVICE=cuda
python run_simulation.py

# Check what device will be used
python src/utils/device.py
```

Note: `--device` flag is available in `run_ieee_experiments.py` and `run_comprehensive.py`. For other scripts, use the `RESILNET_DEVICE` environment variable.

---

## Configuration

All settings are in `config/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| **traffic** | `num_intersections: 4`, `simulation_duration: 3600`, `arrival_distribution: poisson` |
| **federated_learning** | `num_rounds: 100`, `local_epochs: 10`, `learning_rate: 0.001`, `patience: 15` |
| **model** | `hidden_layers: [128, 64, 32]`, `use_batch_norm: true`, `dropout_rate: 0.1` |
| **network** | `base_latency: 10ms`, `bandwidth: 100 Mbps`, `packet_loss: 0.01` |
| **cloudsim** | `edge_vm_mips: 1000`, `cloud_host_mips: 10000` |

Note: The CLI experiment runners (`run_ieee_experiments.py`, `run_comprehensive.py`) override some of these with optimized values (e.g., `[256,128,64,32]` architecture, FedProx mu=0.05).

---

## Quick Reference

| Command | What it does | Time | Output Directory |
|---------|-------------|------|-----------------|
| `python run_ieee_experiments.py` | Full IEEE paper results (5 runs) | ~15-20 min | `results/ieee/` |
| `python run_ieee_experiments.py --runs 1 --rounds 20` | Quick single run | ~3 min | `results/ieee/` |
| `python run_ieee_experiments.py --ablation` | + ablation study | ~25 min | `results/ieee/` |
| `python run_comprehensive.py` | All experiments + dashboard | ~10-15 min | `results/comprehensive/` |
| `python run_comprehensive.py --quick` | Skip scalability | ~5 min | `results/comprehensive/` |
| `python run_publication_experiments.py` | Extended publication suite | ~25-30 min | `results/publication/` |
| `python run_publication_experiments.py --quick` | Quick publication suite | ~10 min | `results/publication/` |
| `python run_simulation.py` | Basic traffic simulation | ~2 min | `data/` + `results/` |
| `python run_demo.py` | System demo walkthrough | ~2 min | `data/` + console |
| `python run_with_ns3.py` | FL + NS-3 network sim | ~10 min | `results/ns3_integrated/` |
| `python run_with_ns3.py --stress-test` | All 5 network scenarios | ~30 min | `results/ns3_stress/` |
| `python run_with_ns3.py --no-ns3` | FL + built-in network sim | ~10 min | `results/ns3_integrated/` |
| `python run_cloudsim.py` | Edge/cloud simulation | ~2 min | console |
| `python run_fl_server.py` | Start FL server | blocks | — |
| `python run_fl_client.py --intersection 0` | Start FL client | blocks | — |
| `python generate_radar_plot.py` | Radar chart for paper | ~5 sec | `results/ieee/` |
| `python generate_architecture_diagram.py` | Architecture diagram | ~5 sec | `results/ieee/` |
| `python generate_comparison.py` | Before/after comparison | ~5 sec | `results/comprehensive/` |
| `python generate_gen_plot.py` | Generalization plot | ~5 sec | `results/ieee/` |
| `python src/utils/device.py` | Verify GPU detection | ~1 sec | console |
| `python src/models/traffic_model.py` | Test model training | ~5 sec | console |
| `python src/federated_learning/aggregation.py` | Test aggregation | ~2 sec | console |

---

## Suggested Run Order

```bash
# Step 1: Verify setup
python src/utils/device.py

# Step 2: Quick test (3 min)
python run_ieee_experiments.py --runs 1 --rounds 20

# Step 3: Full IEEE results (15-20 min)
python run_ieee_experiments.py --runs 5 --rounds 50 --ablation

# Step 4: Generate paper figures
python generate_radar_plot.py
python generate_architecture_diagram.py
python generate_gen_plot.py

# Step 5: Network resilience (optional, needs NS-3 or --no-ns3)
python run_with_ns3.py --stress-test --no-ns3

# Step 6: Extended publication suite (optional)
python run_publication_experiments.py --quick
```
