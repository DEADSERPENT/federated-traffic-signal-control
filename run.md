# ResilNet-FL: How to Run

## Prerequisites

```bash
# Activate virtual environment first
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (Git Bash / MSYS2)
source venv/Scripts/activate
```

Your system: **PyTorch 2.9.1+cu130 | NVIDIA RTX 3050 (4 GB) | CUDA 13.0**
GPU is auto-detected. No extra flags needed.

---

## 1. IEEE Publication Experiments (Start Here)

Runs 5 trials with different seeds, generates publication-quality plots, LaTeX tables, and statistical analysis.

```bash
# Full run (5 trials, 50 FL rounds each) — ~15-20 min
python run_ieee_experiments.py

# Full run + ablation study
python run_ieee_experiments.py --runs 5 --rounds 50 --ablation

# Quick test (1 trial, 20 rounds) — ~3 min
python run_ieee_experiments.py --runs 1 --rounds 20
```

**Output:** `results/ieee/`
- `ieee_method_comparison.png/pdf` — Bar chart: Fixed-Time vs Actuated vs Local-ML vs FL
- `ieee_fl_convergence.png/pdf` — FL MAE convergence across rounds
- `ieee_ablation_study.png/pdf` — Architecture comparison (if `--ablation`)
- `ieee_network_stress.png/pdf` — MAE under network stress (if NS-3 data exists)
- `ieee_results.json` — All raw numbers
- `latex_table.tex` — Copy-paste into your paper
- `generalization_test.json` — FL vs Local-ML on unseen data

---

## 2. Comprehensive Experiments

Runs baseline comparisons + network stress tests + scalability tests + generates a dashboard.

```bash
# Full suite — ~10-15 min
python run_comprehensive.py

# Skip scalability tests — ~5 min
python run_comprehensive.py --quick

# With specific seed
python run_comprehensive.py --seed 42
```

**Output:** `results/comprehensive/`
- `summary_dashboard.png` — All results in one image
- `fl_convergence.png` — Training convergence
- `method_comparison.png` — Baseline comparisons
- `network_stress.png` — Latency/packet-loss resilience
- `scalability.png` — 2 to 8 clients (full mode only)
- `results.json` — All data
- `experiment_report.txt` — Text summary

---

## 3. Publication Experiments (Extended)

Full paper-ready suite: statistical significance, privacy metrics, communication efficiency, live data robustness.

```bash
python run_publication_experiments.py
```

**Output:** `results/` — Complete results for IEEE/ACM submission

---

## 4. Basic Simulation

Generates traffic data and baseline statistics. Good for verifying the setup works.

```bash
python run_simulation.py
```

**Output:**
- `data/traffic_simulation.csv` — Raw traffic data
- `results/traffic_metrics.png` — Traffic visualization
- Console: per-intersection queue/wait/throughput stats

---

## 5. Demo Mode

Step-by-step walkthrough of the full system: simulation, training, evaluation.

```bash
python run_demo.py
```

**Output:** Console (interactive walkthrough)

---

## 6. NS-3 Network Simulation

Realistic V2I communication with IEEE 802.11p/DSRC latency and packet loss.

```bash
# Step 1: Start NS-3 bridge server in WSL (separate terminal)
wsl python3 ns3_simulation/ns3_bridge_server.py

# Step 2: Run FL with NS-3 integration
python run_with_ns3.py

# Run without NS-3 (uses built-in network simulator)
python run_with_ns3.py --no-ns3

# Network stress test only
python run_with_ns3.py --stress-test
```

---

## 7. CloudSim Edge Computing

Simulates edge/cloud offloading for FL aggregation.

```bash
python run_cloudsim.py
```

---

## 8. Distributed FL (Multi-Terminal)

Run FL with separate server and client processes (simulates real deployment).

```bash
# Terminal 1: Start server
python run_fl_server.py --rounds 50 --min-clients 4

# Terminals 2-5: Start one client per intersection
python run_fl_client.py --server localhost:8080 --intersection 0
python run_fl_client.py --server localhost:8080 --intersection 1
python run_fl_client.py --server localhost:8080 --intersection 2
python run_fl_client.py --server localhost:8080 --intersection 3
```

---

## 9. Test Individual Modules

Verify specific components work correctly.

```bash
# GPU/CPU detection
python src/utils/device.py

# Traffic model (training + prediction)
python src/models/traffic_model.py

# Byzantine-robust aggregation
python src/federated_learning/aggregation.py

# Trace-driven network simulation
python ns3_simulation/network_traces.py

# NS-3 bridge client
python ns3_simulation/ns3_bridge_client.py

# Visualization functions
python src/utils/visualization.py
```

---

## Device Selection

GPU is auto-detected by default. Override if needed:

```bash
# Auto-detect (recommended) — will use your RTX 3050
python run_ieee_experiments.py

# Force CPU
python run_ieee_experiments.py --device cpu

# Force CUDA
python run_ieee_experiments.py --device cuda

# Environment variable (works with all scripts)
set RESILNET_DEVICE=cuda
python run_ieee_experiments.py
```

---

## Quick Reference

| Command | What it does | Time | Output |
|---|---|---|---|
| `python run_ieee_experiments.py` | Full IEEE results (5 runs) | ~15-20 min | `results/ieee/` |
| `python run_ieee_experiments.py --runs 1` | Quick single run | ~3 min | `results/ieee/` |
| `python run_comprehensive.py` | All experiments + dashboard | ~10-15 min | `results/comprehensive/` |
| `python run_comprehensive.py --quick` | Fast mode (no scalability) | ~5 min | `results/comprehensive/` |
| `python run_simulation.py` | Basic traffic simulation | ~2 min | `data/` + `results/` |
| `python run_demo.py` | Interactive demo | Interactive | Console |
| `python src/utils/device.py` | Verify GPU detection | ~1 sec | Console |
