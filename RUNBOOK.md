# ResilNet-FL — Complete Operational Runbook

**System:** Hierarchical Byzantine-Robust Federated Learning for Intelligent Traffic Signal Control  
**Version:** 2026 (CODE AI-2026 Submission Branch)  
**Hardware Reference:** Intel Core i7-14700 · NVIDIA RTX 3050 4 GB · CUDA 13.0 · WSL2 Ubuntu  
**Python Runtime:** `python3` (native, no conda)

---

## Table of Contents

1. [Prerequisites & Hardware Requirements](#1-prerequisites--hardware-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Project Structure](#3-project-structure)
4. [Configuration Reference](#4-configuration-reference)
5. [Experiment Modes — Quick Reference](#5-experiment-modes--quick-reference)
6. [Detailed Experiment Guides](#6-detailed-experiment-guides)
   - 6.1 Sanity Check
   - 6.2 IEEE Publication Experiments
   - 6.3 Byzantine Robustness (ResilAgg + H-FL)
   - 6.4 Comprehensive Evaluation
   - 6.5 Publication Suite
   - 6.6 NS-3 Network Simulation
   - 6.7 CloudSim Edge Computing
   - 6.8 Distributed FL (Flower)
   - 6.9 SUMO Traffic Simulation
7. [Novel Algorithm Reference](#7-novel-algorithm-reference)
8. [Paper Figure Generation](#8-paper-figure-generation)
9. [LaTeX Integration](#9-latex-integration)
10. [GPU / Device Management](#10-gpu--device-management)
11. [Module-Level Tests](#11-module-level-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [Production Deployment Checklist](#13-production-deployment-checklist)

---

## 1. Prerequisites & Hardware Requirements

### Minimum Hardware

| Component | Minimum | Recommended (paper hardware) |
|-----------|---------|-------------------------------|
| CPU | 4-core, 3 GHz | Intel Core i7-14700 (20 cores) |
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU fallback) | NVIDIA RTX 3050 4 GB (CUDA 13.0) |
| Storage | 5 GB free | 20 GB (for all result sets) |
| OS | Linux / WSL2 | WSL2 Ubuntu 22.04 LTS |

### Software Requirements

| Package | Version | Install |
|---------|---------|---------|
| Python | 3.10+ | `sudo apt-get install python3` |
| pip | 23+ | `sudo apt-get install python3-pip` |
| CUDA Toolkit | 12.0+ (optional) | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| NS-3 | 3.40+ (optional) | See Section 6.6 |
| SUMO | 1.18+ (optional) | `sudo apt-get install sumo sumo-tools` |

> The system runs fully on CPU. GPU accelerates training ~10× and distance computation ~430×. All results in the paper were produced with the RTX 3050 on WSL2.

---

## 2. Environment Setup

### First-Time Installation

```bash
# Step 1: Navigate to project root
cd /home/deadserpent/federated-traffic-signal-control

# Step 2: Install system pip if not present
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Step 3: Create virtual environment
python3 -m venv venv

# Step 4: Activate virtual environment
source venv/bin/activate

# Step 5: Upgrade pip
pip install --upgrade pip

# Step 6: Install all project dependencies
pip install -r requirements.txt

# Step 7: Verify installation
python3 src/utils/device.py
```

Expected output from Step 7:
```
Device: cuda (NVIDIA GeForce RTX 3050 4GB Laptop GPU)
CUDA available: True
GPU memory: 4.0 GB
```

### Activating the Environment (Every Session)

```bash
cd /home/deadserpent/federated-traffic-signal-control
source venv/bin/activate
```

### Deactivating

```bash
deactivate
```

### Verifying All Core Modules

```bash
python3 -c "import torch, numpy, flwr, sklearn, matplotlib, scipy, yaml; print('All OK')"
```

---

## 3. Project Structure

```
federated-traffic-signal-control/
│
├── main.py                          # Central launcher — all modes via --mode flag
├── config/
│   └── config.yaml                  # Master configuration file
│
├── scripts/                         # Experiment runner scripts
│   ├── run_ieee.py                  # IEEE publication: 5 methods, 5 runs
│   ├── run_byzantine.py             # Byzantine robustness: ResilAgg + H-FL sweep
│   ├── run_comprehensive.py         # Full baseline comparison + dashboard
│   ├── run_publication.py           # Extended suite: privacy + stats + network
│   ├── run_simulation.py            # Basic traffic data generation
│   ├── run_demo.py                  # System walkthrough demo
│   ├── run_with_ns3.py              # FL + NS-3 V2I network simulation
│   ├── run_cloudsim.py              # Edge/cloud computing simulation
│   ├── run_fl_server.py             # Distributed Flower FL server
│   ├── run_fl_client.py             # Distributed Flower FL client
│   └── run_sumo_gui.py              # SUMO visual simulation
│
├── src/
│   ├── federated_learning/
│   │   ├── aggregation.py           # All aggregation strategies incl. ResilAgg, H-FL
│   │   ├── hierarchical.py          # H-FL: FogNode, HierarchicalFLController  [NEW]
│   │   ├── cuda_krum.py             # GPU-accelerated distance backend            [NEW]
│   │   ├── cuda_krum.cu             # Custom CUDA kernel (tiled GEMM)             [NEW]
│   │   ├── client.py                # Flower FL client
│   │   └── server.py                # Flower FL server
│   │
│   ├── baselines/
│   │   ├── adaptive_fl.py           # AdaptiveFLController + PrioritizedReplay   [UPDATED]
│   │   ├── fixed_time.py            # Fixed-time baseline
│   │   ├── local_ml.py              # Local-ML baseline
│   │   ├── centralized_ml.py        # Centralized-ML baseline
│   │   ├── actuated.py              # Actuated (gap-out) baseline
│   │   └── adaptive_fl.py           # FL controller (main)
│   │
│   ├── models/
│   │   └── traffic_model.py         # MLP / LSTM / GRU + train/evaluate functions
│   │
│   ├── traffic_generator/
│   │   ├── generator.py             # Poisson traffic simulator
│   │   ├── intersection.py          # Intersection state machine
│   │   └── sumo_integration.py      # SUMO 3×3 grid bridge
│   │
│   ├── cloudsim_python/
│   │   └── edge_cloud_sim.py        # CloudSim edge/cloud simulation
│   │
│   ├── experiments/
│   │   ├── comprehensive_runner.py  # Full experiment orchestrator
│   │   ├── network_stress.py        # NS-3 / network stress scenarios
│   │   └── scalability.py           # 2-9 client scaling experiments
│   │
│   └── utils/
│       ├── device.py                # GPU/CPU auto-detection
│       ├── reproducibility.py       # Global seed management
│       ├── visualization.py         # Plot generation
│       ├── professional_plots.py    # Publication-quality figures
│       ├── metrics.py               # MAE / MSE / wait-time calculations
│       ├── privacy_metrics.py       # Differential privacy quantification
│       └── statistical_tests.py     # Wilcoxon / t-test / confidence intervals
│
├── ns3_simulation/                  # NS-3 network simulation bridge
│   ├── ns3_bridge_server.py         # WSL-side ZeroMQ server
│   ├── ns3_bridge_client.py         # Windows-side ZeroMQ client
│   └── network_traces.py            # 802.11p / LTE-V2X / 5G trace generator
│
├── tools/                           # Standalone figure generators
│   ├── generate_radar_plot.py
│   ├── generate_comparison.py
│   ├── generate_gen_plot.py
│   └── generate_architecture.py
│
├── results/                         # All output (auto-created)
│   ├── ieee/
│   ├── byzantine/
│   ├── comprehensive/
│   ├── publication/
│   ├── ns3_integrated/
│   └── ns3_stress/
│
├── data/                            # Generated CSV traffic data
├── paper/                           # LaTeX paper source
├── requirements.txt
├── RUNBOOK.md                       # This file
└── PROJECT_BRIEF.md                 # Problem statement + stakeholder brief
```

---

## 4. Configuration Reference

All defaults live in `config/config.yaml`. CLI flags in individual scripts override these at runtime.

### Key Sections

```yaml
traffic:
  num_intersections: 4          # 4 = 2x2 grid; 9 = 3x3 grid (used for paper)
  simulation_duration: 3600     # Seconds of simulated traffic (1 hour)
  time_step: 5                  # State update interval in seconds
  arrival_distribution: poisson # Vehicle arrivals follow Poisson process
  min_arrival_rate: 5           # Vehicles/minute (residential)
  max_arrival_rate: 30          # Vehicles/minute (CBD rush hour)
  max_queue_length: 50          # Physical lane capacity
  min_green_duration: 20        # Minimum green phase (seconds)
  max_green_duration: 90        # Maximum green phase (seconds)

federated_learning:
  num_rounds: 150               # Total FL communication rounds
  local_epochs: 5               # Epochs per client per round (FedProx-optimal)
  batch_size: 32
  learning_rate: 0.001
  lr_decay: 0.995               # Per-round LR multiplier
  weight_decay: 0.0001
  mu: 0.01                      # FedProx proximal weight
  strategy: "resil_agg"         # Default: ResilAgg (change to h_fl for hierarchical)

model:
  type: "lstm"                  # lstm | gru | neural_network
  hidden_dim: 128               # LSTM hidden state size
  num_layers: 2                 # LSTM stacked layers
  dropout_rate: 0.15
```

### Switching Aggregation Strategy

Edit `config/config.yaml`:

| Strategy | `strategy` value | Notes |
|----------|-----------------|-------|
| Standard FedAvg | `fedavg` | Vulnerable to Byzantine |
| Trimmed Mean | `trimmed_mean` | Fast, 2017 baseline |
| Coordinate Median | `median` | 2018 baseline |
| Multi-Krum | `multi_krum` | Requires known f |
| **ResilAgg (Ours)** | `resil_agg` | Dynamic MAD filter + quality-aware |
| **H-FL (Ours)** | `h_fl` | Hierarchical fog + cloud |
| Quality-Aware | `quality_aware` | Inverse-loss weighting only |

---

## 5. Experiment Modes — Quick Reference

All modes launch via `main.py` or directly via the script in `scripts/`.

```bash
# Via main.py (recommended)
python3 main.py --mode <mode> [extra flags]

# Direct script
python3 scripts/run_<mode>.py [flags]
```

| Mode | Command | Time | Primary Output |
|------|---------|------|----------------|
| Sanity check | `python3 src/utils/device.py` | 2 sec | Console |
| Demo | `python3 main.py --mode demo` | 2 min | Console + `data/` |
| IEEE full | `python3 main.py --mode ieee` | 15-20 min | `results/ieee/` |
| Byzantine | `python3 main.py --mode byzantine` | 25-35 min | `results/byzantine/` |
| Byzantine quick | `python3 scripts/run_byzantine.py --quick` | 5 min | `results/byzantine/` |
| Comprehensive | `python3 main.py --mode comprehensive` | 10-15 min | `results/comprehensive/` |
| Publication | `python3 main.py --mode publication` | 25-30 min | `results/publication/` |
| NS-3 network | `python3 main.py --mode ns3 -- --no-ns3` | 10 min | `results/ns3_integrated/` |
| NS-3 stress test | `python3 main.py --mode ns3 -- --stress-test --no-ns3` | 30 min | `results/ns3_stress/` |
| CloudSim | `python3 main.py --mode cloudsim` | 2 min | Console |
| Simulation only | `python3 main.py --mode simulation` | 2 min | `data/` |

---

## 6. Detailed Experiment Guides

### 6.1 Sanity Check (Run First)

Before any experiment, verify the environment is working correctly.

```bash
# Check GPU detection
python3 src/utils/device.py

# Check aggregation module (includes ResilAgg smoke test)
python3 src/federated_learning/aggregation.py

# Check traffic model
python3 src/models/traffic_model.py

# Check SUMO integration (falls back gracefully without SUMO)
python3 src/traffic_generator/sumo_integration.py
```

All four should complete without errors and print "OK" or summary statistics.

---

### 6.2 IEEE Publication Experiments

Runs N independent trials, compares all 5 methods (Fixed-Time, Actuated, Local-ML, Centralized-ML, FL), and generates all LaTeX-ready outputs.

```bash
# Recommended full run for paper (15-20 min, RTX 3050)
python3 scripts/run_ieee.py --runs 5 --rounds 50

# With ablation study of model architectures (+10 min)
python3 scripts/run_ieee.py --runs 5 --rounds 50 --ablation

# 9-intersection 3x3 grid (uses SUMO if installed, else Poisson)
python3 scripts/run_ieee.py --intersections 9 --runs 5 --rounds 50

# Sanity run: 1 trial, 20 rounds (~3 min)
python3 scripts/run_ieee.py --runs 1 --rounds 20

# Force CPU (slower but reproducible on any machine)
python3 scripts/run_ieee.py --device cpu --runs 3 --rounds 30
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 5 | Independent trials (use ≥5 for statistical significance) |
| `--rounds N` | 50 | FL rounds per trial |
| `--intersections N` | 4 | Grid size (4 or 9) |
| `--ablation` | off | Compare MLP vs LSTM vs GRU architectures |
| `--output DIR` | `results/ieee` | Output directory |
| `--device` | auto | `auto` / `cpu` / `cuda` |

#### Outputs (`results/ieee/`)

```
ieee_method_comparison.png/pdf   — Bar chart: 5-method wait time + MAE comparison
ieee_fl_convergence.png/pdf      — MAE convergence across FL rounds (all 5 runs + mean)
ieee_ablation_study.png/pdf      — Architecture ablation (with --ablation flag)
ieee_results.json                — All raw numeric results
latex_table.tex                  — Paste into paper Section VII
generalization_test.json         — FL vs Local-ML on unseen traffic (seed 9999)
```

---

### 6.3 Byzantine Robustness — ResilAgg + H-FL

This is the primary novel contribution experiment. Sweeps Byzantine client count (0, 1, 2) across all 7 aggregation strategies including the two novel contributions.

```bash
# Standard run (recommended for paper, ~25-35 min)
python3 scripts/run_byzantine.py

# Quick mode: 4 intersections, 1 seed, 20 rounds (~5 min)
python3 scripts/run_byzantine.py --quick

# Extended: 9 intersections, 5 seeds, 50 rounds (most rigorous, ~60 min)
python3 scripts/run_byzantine.py --intersections 9 --seeds 5 --rounds 50

# Realistic sensor-fault attack (5x noise — default and recommended)
python3 scripts/run_byzantine.py --noise-scale 5

# Strong model-poisoning attack (50x noise)
python3 scripts/run_byzantine.py --noise-scale 50

# Change output directory
python3 scripts/run_byzantine.py --output results/byzantine_v2
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--intersections N` | 9 | Total intersections (Byzantine = last N) |
| `--rounds N` | 40 | FL rounds per trial |
| `--noise-scale X` | 5.0 | Byzantine noise multiplier (5x = realistic sensor fault) |
| `--seeds N` | 3 | Seeds per condition |
| `--output DIR` | `results/byzantine` | Output directory |
| `--quick` | off | Fast mode: 4 intersections, 1 seed, 20 rounds |

#### Strategies Compared

| Strategy | Type | Novel |
|----------|------|-------|
| H-FL | Hierarchical fog+cloud | **Ours (2026)** |
| ResilAgg | MAD-filtered quality-aware | **Ours (2026)** |
| FedAvg | Plain averaging | Baseline |
| Quality-Aware | Inverse-loss weighting | Ours (prior work) |
| Trimmed Mean | Coordinate trim | Yin et al. 2018 |
| Median | Coordinate median | Yin et al. 2018 |
| Multi-Krum | Nearest-neighbor selection | Blanchard et al. 2017 |

#### What to Look for in Results

- **FedAvg** should degrade 20-60% at 1-2 Byzantine clients
- **ResilAgg** should stay within 5% of clean baseline (automatic f-detection)
- **H-FL** should show lowest absolute MAE (Byzantine fault contained at fog layer)
- **Multi-Krum** should degrade when Byzantine clients == f+1 (fixed tolerance limit)

#### Outputs (`results/byzantine/`)

```
byzantine_robustness.png/pdf   — 2-panel: MAE vs Byzantine count + relative degradation %
byzantine_table.tex            — LaTeX table for paper (paste into Section VIII)
byzantine_results.json         — Full numeric results with mean ± std per strategy
```

---

### 6.4 Comprehensive Evaluation

Runs baseline comparison, network stress tests, and scalability analysis together.

```bash
# Full suite (~10-15 min)
python3 scripts/run_comprehensive.py

# Skip scalability tests (~5 min)
python3 scripts/run_comprehensive.py --quick

# With specific seed
python3 scripts/run_comprehensive.py --seed 42
```

#### Outputs (`results/comprehensive/`)

```
summary_dashboard.png      — All results in one 4-panel figure
fl_convergence.png         — Training convergence
method_comparison.png      — Bar chart: all baselines
network_stress.png         — MAE under latency/packet-loss scenarios
scalability.png            — MAE vs number of clients (2 to 9)
experiment_report.txt      — Full text summary
results.json               — All raw data
```

---

### 6.5 Publication Suite (Extended)

Full conference-submission-ready suite including privacy quantification, communication efficiency, and statistical significance tests.

```bash
# Full run: 5 trials, 100 rounds (~25-30 min)
python3 scripts/run_publication.py

# Quick: 3 trials, 50 rounds (~10 min)
python3 scripts/run_publication.py --quick

# Custom
python3 scripts/run_publication.py --runs 5 --rounds 100
```

#### Outputs (`results/publication/`)

```
complete_results.json          — All experiment data
publication_report.txt         — Full text report
statistical_analysis.txt       — Wilcoxon / t-test results
statistical_table.tex          — LaTeX significance table
privacy_analysis.txt           — Differential privacy metrics
privacy_comparison.json        — Centralized vs FL privacy trade-off
```

---

### 6.6 NS-3 Network Simulation

Simulates IEEE 802.11p DSRC V2I communication with realistic latency, jitter, and packet loss. NS-3 runs in WSL2 and communicates with the Python FL stack via ZeroMQ.

#### Without NS-3 (built-in statistical simulator — recommended)

```bash
# Single scenario
python3 scripts/run_with_ns3.py --no-ns3 --scenario normal

# Run all 5 scenarios sequentially (~30 min)
python3 scripts/run_with_ns3.py --no-ns3 --stress-test

# Custom FL rounds
python3 scripts/run_with_ns3.py --no-ns3 --rounds 100 --scenario stressed
```

#### With NS-3 (requires WSL setup)

```bash
# Terminal 1 — Start the ZeroMQ bridge server in WSL
python3 ns3_simulation/ns3_bridge_server.py

# Terminal 2 — Run the FL experiment
python3 scripts/run_with_ns3.py --rounds 50
```

#### NS-3 Installation (One-Time, WSL Only)

```bash
sudo apt-get install -y g++ python3-dev cmake ninja-build
# Follow docs/NS3_INTEGRATION_GUIDE.md for full NS-3 3.40 build
```

#### Network Scenarios

| Scenario | Latency | Jitter | Packet Loss | Bandwidth | Maps to |
|----------|---------|--------|-------------|-----------|---------|
| `ideal` | 5 ms | ±2 ms | 0% | 54 Mbps | 5G NR V2X |
| `normal` | 15 ms | ±5 ms | 1% | 27 Mbps | Hybrid 5G/LTE |
| `degraded` | 50 ms | ±15 ms | 5% | 12 Mbps | LTE-V2X |
| `stressed` | 100 ms | ±30 ms | 10% | 6 Mbps | 802.11p DSRC |
| `extreme` | 200 ms | ±50 ms | 20% | 3 Mbps | Congested DSRC |

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--no-ns3` | off | Use built-in simulator instead of NS-3 |
| `--scenario` | `normal` | Network scenario (see table above) |
| `--rounds N` | 50 | FL training rounds |
| `--stress-test` | off | Run all 5 scenarios |

#### Outputs

```
results/ns3_integrated/ns3_results.json      — Per-round metrics
results/ns3_stress/<scenario>/               — Per-scenario results (--stress-test)
results/ns3_stress/stress_test_summary.json  — Cross-scenario summary
```

---

### 6.7 CloudSim Edge Computing

Simulates edge/cloud resource allocation for FL aggregation tasks.

```bash
python3 scripts/run_cloudsim.py
```

No flags. Output is printed to console showing:
- Edge server computation time per intersection
- Cloud aggregation latency
- Resource utilization per VM
- FL task scheduling timeline

---

### 6.8 Distributed FL (Flower Framework)

Runs true distributed federated learning where each intersection is a separate process communicating with a central server. Requires multiple terminals.

```bash
# Terminal 1 — Start server (waits for min_clients before training begins)
python3 scripts/run_fl_server.py --rounds 50 --min-clients 4

# Terminals 2-5 — Start one client per intersection
python3 scripts/run_fl_client.py --server localhost:8080 --intersection 0
python3 scripts/run_fl_client.py --server localhost:8080 --intersection 1
python3 scripts/run_fl_client.py --server localhost:8080 --intersection 2
python3 scripts/run_fl_client.py --server localhost:8080 --intersection 3
```

For 9-intersection grid, launch clients 0 through 8.

#### Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--address` | `0.0.0.0:8080` | Bind address |
| `--rounds N` | 10 | FL rounds |
| `--min-clients N` | 2 | Clients required before starting |
| `--config PATH` | `config/config.yaml` | Config file |

#### Client Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `localhost:8080` | Server address |
| `--intersection N` | 0 | Intersection ID (0-indexed) |
| `--config PATH` | none | Config file |

---

### 6.9 SUMO Traffic Simulation

SUMO provides a realistic microsimulation of vehicle movements in the 3×3 intersection grid. Without SUMO installed, the system transparently falls back to the enhanced Poisson simulator.

```bash
# Check SUMO availability and test the integration
python3 src/traffic_generator/sumo_integration.py

# Run full simulation with SUMO GUI
python3 main.py --mode sumo-gui
```

#### Installing SUMO (Optional)

```bash
sudo apt-get install -y sumo sumo-tools sumo-doc
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

---

## 7. Novel Algorithm Reference

### ResilAgg (`strategy="resil_agg"`)

Two-stage aggregation. No need to pre-specify the number of Byzantine clients.

**Stage 1 — Dynamic Byzantine Filter**
```
1. Flatten each client's model parameters → vector u_i
2. Compute hybrid distance: d(i,j) = ||u_i - u_j||_2 × (1 + cosine_dist(u_i, u_j))
3. Score each client: score_i = Σ_j d(i,j)
4. Modified Z-score: z_i = 0.6745 × (score_i − median) / MAD
5. Drop clients with z_i > 3.0 (anomalous distance to the honest cluster)
```

**Stage 2 — Quality-Aware Aggregation**
```
6. For survivors: weight_k = data_size_k / (loss_k + ε)
7. Normalize weights, compute weighted average
```

**When to use:** Default for all production experiments. Best when f is unknown.

---

### H-FL (`strategy="h_fl"`)

Hierarchical two-level aggregation.

**Cluster Assignment (9 intersections)**
```
Cluster 0 (CBD):        intersections [0, 2, 4, 6, 8] — corners + centre
Cluster 1 (Arterial):   intersections [1, 3, 5, 7]    — edge midpoints
Cluster 2 (Residential):intersections []               — (absorbed into clusters 0-1)
```

**Fog Level (per cluster)**
```
Apply ResilAgg within each cluster → one fog model per cluster
```

**Cloud Level (across clusters)**
```
Apply Multi-Krum across the K fog models → global model
```

**When to use:** Large grids (9+ intersections) with known geographic clustering. Provides the best Byzantine containment.

---

### Prioritized Replay (`use_prioritized_replay=True`)

Enabled by default in `AdaptiveFLController`.

```
Per round per intersection:
1. After local training, compute per-sample |ŷ - y| errors
2. Add (features, label, error^alpha) to ring buffer (capacity=2000)
3. Next round: sample 30% of training data from buffer (proportional to priority)
4. Concatenate replay samples with current round data before training
```

**Parameters:**
- `replay_alpha=0.6` — priority exponent (0=uniform, 1=fully proportional)
- `replay_blend_ratio=0.30` — fraction of training batch from buffer
- `replay_buffer_capacity=2000` — samples stored per intersection

---

### GPU Distance Backend (`cuda_krum.py`)

Automatically selected when available. No user action required.

```
Priority order:
1. Custom CUDA kernel (cuda_krum.cu) — JIT-compiled via torch.utils.cpp_extension
2. torch.cdist (cuBLAS-backed on GPU)
3. NumPy loops (CPU fallback)
```

To force the CPU path (e.g., for reproducibility verification):
```bash
CUDA_KRUM_USE_CDIST=1 python3 scripts/run_byzantine.py
```

To benchmark all backends:
```bash
python3 src/federated_learning/cuda_krum.py
```

---

## 8. Paper Figure Generation

Run after completing experiments. These scripts read from `results/` and generate publication-quality figures.

```bash
# Radar chart — multi-objective trade-off (wait time, MAE, privacy, latency)
python3 tools/generate_radar_plot.py
# Output: results/ieee/ieee_tradeoff_radar.png / .pdf

# System architecture — 3-tier block diagram
python3 tools/generate_architecture.py
# Output: results/ieee/system_architecture.png / .pdf

# Before/after optimization comparison plot
python3 tools/generate_comparison.py
# Output: results/comprehensive/optimization_comparison.png

# Generalization bar chart (FL vs Local-ML on unseen intersections)
python3 tools/generate_gen_plot.py
# Output: results/ieee/ieee_generalization.png
```

---

## 9. LaTeX Integration

After running the full experiment suite, copy these files directly into the paper:

| Generated File | Paper Section | Usage |
|----------------|---------------|-------|
| `results/ieee/latex_table.tex` | Section VII | Method comparison table |
| `results/byzantine/byzantine_table.tex` | Section VIII | Byzantine robustness table (H-FL + ResilAgg bolded) |
| `results/publication/statistical_table.tex` | Section VII | Significance test results |
| `results/ieee/ieee_method_comparison.png` | Section VII Fig. 3 | Bar chart |
| `results/ieee/ieee_fl_convergence.png` | Section VII Fig. 4 | Convergence curve |
| `results/byzantine/byzantine_robustness.png` | Section VIII Fig. 5 | 2-panel Byzantine figure |
| `results/ieee/ieee_tradeoff_radar.png` | Section IX Fig. 6 | Radar trade-off chart |
| `results/ieee/system_architecture.png` | Section III Fig. 1 | Architecture diagram |

### LaTeX Include Commands

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\columnwidth]{figures/byzantine_robustness.pdf}
  \caption{Byzantine Robustness: H-FL and ResilAgg vs. baselines under sensor-noise attacks.}
  \label{fig:byzantine}
\end{figure}

\input{tables/byzantine_table.tex}
```

---

## 10. GPU / Device Management

```bash
# Check detected device
python3 src/utils/device.py

# Force CUDA in any script
RESILNET_DEVICE=cuda python3 scripts/run_ieee.py

# Force CPU in any script
RESILNET_DEVICE=cpu python3 scripts/run_ieee.py

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

### Expected GPU Memory Usage

| Experiment | GPU Memory |
|------------|-----------|
| Demo / simulation | < 0.5 GB |
| IEEE experiments (4 intersections) | ~0.8 GB |
| Byzantine sweep (9 intersections) | ~1.2 GB |
| Full publication suite | ~1.5 GB |
| All LSTM + batch norm | < 2.0 GB |

The RTX 3050 4 GB handles all experiments with headroom. On CPU, multiply all time estimates by ~10.

---

## 11. Module-Level Tests

Use these to isolate and debug individual components.

```bash
# GPU detection and memory info
python3 src/utils/device.py

# All aggregation strategies smoke test (7 strategies, 7 clients with 2 Byzantine)
python3 src/federated_learning/aggregation.py

# H-FL cluster assignment + two-level aggregation
python3 -c "
from src.federated_learning.hierarchical import assign_clusters_balanced, HierarchicalFLController
print(assign_clusters_balanced(9, 3))
ctrl = HierarchicalFLController(num_intersections=9)
"

# GPU distance benchmark (prints ms per backend)
python3 src/federated_learning/cuda_krum.py

# Traffic model: create, train 10 epochs, evaluate
python3 src/models/traffic_model.py

# SUMO integration (auto-detects SUMO, falls back gracefully)
python3 src/traffic_generator/sumo_integration.py

# Centralized-ML baseline (5 min test)
python3 src/baselines/centralized_ml.py

# Network trace generation
python3 ns3_simulation/network_traces.py

# Visualization
python3 src/utils/visualization.py
```

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'flwr'`

```bash
pip install flwr>=1.25.0
```

### CUDA out of memory

```bash
# Force CPU for this run
RESILNET_DEVICE=cpu python3 scripts/run_byzantine.py
```

Or reduce batch size in `config/config.yaml`:
```yaml
federated_learning:
  batch_size: 16   # reduce from 32
```

### Byzantine experiment crashes with `n_clients < 3`

ResilAgg requires at least 3 clients. Use `--quick` mode (4 intersections) or increase `--intersections`:
```bash
python3 scripts/run_byzantine.py --intersections 6 --quick
```

### NS-3 bridge timeout / `ConnectionRefusedError`

```bash
# In WSL Terminal 1 — start bridge server first
python3 ns3_simulation/ns3_bridge_server.py

# Wait for "Server ready" message, then run:
python3 scripts/run_with_ns3.py
```

### Custom CUDA kernel compilation fails

This is non-fatal. The system falls back to `torch.cdist` automatically. To confirm:
```bash
CUDA_KRUM_USE_CDIST=1 python3 src/federated_learning/cuda_krum.py
```
Results are numerically identical.

### SUMO not found

No action needed. The system automatically falls back to the enhanced Poisson simulator. A message like `SUMO not available, using Poisson fallback` is expected and correct.

### Flower `Address already in use`

```bash
# Kill any existing Flower server
pkill -f "run_fl_server"
# Or change the port
python3 scripts/run_fl_server.py --address 0.0.0.0:8081
```

### Results directory not created

```bash
mkdir -p results/{ieee,byzantine,comprehensive,publication,ns3_integrated,ns3_stress}
```

---

## 13. Production Deployment Checklist

This checklist is for deploying the trained ResilNet-FL system to a live intersection grid.

### Phase 1: Infrastructure

- [ ] Verify edge compute hardware at each intersection (min: Raspberry Pi 5 / Jetson Nano, recommended: NVIDIA Jetson AGX Orin)
- [ ] Verify central aggregation server (min: 16-core CPU + 4 GB GPU)
- [ ] Confirm V2I communication protocol (802.11p DSRC / LTE-V2X / 5G NR V2X)
- [ ] Set network latency budget ≤ 200 ms per FL round (system validated to 658 ms)
- [ ] Establish secure channel for model parameter transmission (TLS 1.3 minimum)

### Phase 2: Data & Privacy

- [ ] Confirm raw vehicle trajectory data never leaves intersection edge node
- [ ] Verify only model gradients / parameters are transmitted (privacy-preserving by design)
- [ ] Run `scripts/run_publication.py` and review `privacy_analysis.txt` for DP budget
- [ ] Obtain data governance approval for model parameter sharing (no PII transmitted)

### Phase 3: Training & Validation

- [ ] Run full IEEE experiment suite on production-representative traffic data
- [ ] Run Byzantine robustness sweep: system must show < 5% MAE degradation at expected faulty-sensor rate
- [ ] Record baseline wait times for minimum 1 week before FL deployment
- [ ] Validate FL MAE on held-out intersections (generalization test)

### Phase 4: Live Deployment

- [ ] Deploy `AdaptiveFLController` with `aggregation_strategy="h_fl"` (hierarchical) and `use_prioritized_replay=True`
- [ ] Set `num_byzantine` in `config.yaml` to match expected faulty sensors in deployment zone
- [ ] Configure cluster assignments in `hierarchical.py` to match geographic intersection layout
- [ ] Set FL round frequency (recommended: 1 round per 15 minutes during active hours)
- [ ] Monitor: if fog-level MAD filter rejects > 30% of clients in a cluster, trigger manual sensor inspection

### Phase 5: Monitoring

- [ ] Log per-round MAE, Byzantine client count, survivor indices from ResilAgg
- [ ] Alert if global MAE degrades > 10% vs. 7-day rolling average
- [ ] Log average vehicle wait time at each intersection (primary production KPI)
- [ ] Monthly: retrain on updated traffic data, validate on held-out week

### Rollback

If the FL system underperforms:
1. Switch to `strategy: "trimmed_mean"` (no training required)
2. If issue persists, fall back to `strategy: "fedavg"` with no `num_byzantine`
3. Last resort: revert to hardware-actuated control (no ML required)

All fallback strategies are implemented in `aggregation.py` and require only a config change and server restart — no redeployment.
