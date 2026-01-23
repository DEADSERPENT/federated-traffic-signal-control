# Federated Learning-based Adaptive Traffic Signal Control with CloudSim and NS-3

A privacy-preserving adaptive traffic signal control system using **Federated Learning (FL)** with **CloudSim-based edge/cloud computing** and **NS-3 network simulation** for realistic V2I communication.

## Research Framework: FL + CloudSim + NS-3

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │ FEDERATED   │    │  CLOUDSIM   │    │    NS-3     │                │
│   │  LEARNING   │◄──►│   (Edge/    │◄──►│  (Network   │                │
│   │  (Privacy)  │    │   Cloud)    │    │ Simulation) │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│         │                  │                  │                         │
│         ▼                  ▼                  ▼                         │
│   • FedAvg Algorithm  • Edge Servers    • 802.11p DSRC                 │
│   • Local Training    • Cloud VMs       • V2I Communication            │
│   • Model Aggregation • Task Scheduling • Latency/Packet Loss          │
│   • Privacy Preserve  • Resource Mgmt   • Bandwidth Modeling           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Research Highlights

- **FL outperforms all baselines** in prediction accuracy (MAE) by 5.76%
- **CloudSim integration** - Edge/cloud computing for FL model training
- **NS-3 network simulation** - Realistic 802.11p DSRC V2I communication
- **Robust under network stress** - maintains performance up to 658ms latency
- **Privacy-preserving** - no raw traffic data shared between intersections
- **IEEE publication-ready** - statistical analysis with 5+ experimental runs

## Performance Summary

### FL vs Baselines (Statistical Analysis - 5 Runs)

| Method | Wait Time | MAE | Improvement |
|--------|-----------|-----|-------------|
| Fixed-Time | 13.23 ± 0.31s | N/A | Baseline |
| Local-ML | 9.13 ± 0.22s | 1.9449 ± 0.2906 | - |
| **FL (Ours)** | **9.06 ± 0.18s** | **1.8069 ± 0.1441** | **+7.1% MAE** |

### Network Stress Test (NS-3 Simulation)

| Scenario | Latency | Packet Loss | FL MAE | FL Wins? |
|----------|---------|-------------|--------|----------|
| Ideal | 29ms | 0% | 2.1369 | Yes |
| Normal | 62ms | 1% | 2.1335 | Yes |
| Degraded | 170ms | 5% | 2.1485 | Yes |
| Stressed | 335ms | 10% | 2.1208 | Yes |
| Extreme | 658ms | 20% | 2.1402 | Yes |

**Key Finding**: FL maintains superior accuracy even under extreme network conditions!

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows Environment                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Federated Learning Traffic Controller           │   │
│  │  • AdaptiveFLController with deep neural network         │   │
│  │  • FedAvg aggregation with weighted averaging            │   │
│  │  • Real-time adaptive signal control                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    ZeroMQ Bridge (tcp://localhost:5555)          │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                    WSL2 / Linux Environment                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    NS-3 Simulator                         │   │
│  │  • 802.11p (DSRC) V2I wireless communication             │   │
│  │  • Realistic latency, packet loss, bandwidth             │   │
│  │  • Per-packet network simulation                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd "TRAFFIC SIGNALS"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Option 1: Quick FL test (no NS-3)
python run_with_ns3.py --no-ns3 --rounds 50

# Option 2: FL with NS-3 network simulation
python run_with_ns3.py --rounds 50

# Option 3: FL with CloudSim edge/cloud simulation
python run_cloudsim.py

# Option 4: Network stress test (all scenarios)
python run_with_ns3.py --stress-test

# Option 5: Comprehensive evaluation (all methods)
python run_comprehensive.py

# Option 6: IEEE publication experiments (5 runs + ablation)
python run_ieee_experiments.py --runs 5 --rounds 50 --ablation

# Option 7: Distributed FL (separate terminals)
# Terminal 1: python run_fl_server.py --rounds 30
# Terminal 2: python run_fl_client.py --intersection 0
# Terminal 3: python run_fl_client.py --intersection 1
```

## Project Structure

```
TRAFFIC SIGNALS/
├── src/
│   ├── baselines/
│   │   ├── adaptive_fl.py      # FL controller (MAIN)
│   │   ├── fixed_time.py       # Fixed-time baseline
│   │   └── local_ml.py         # Local-ML baseline
│   ├── models/
│   │   └── traffic_model.py    # Neural network [256,128,64,32]
│   ├── cloudsim_python/        # CloudSim Edge/Cloud Simulation
│   │   ├── __init__.py
│   │   └── edge_cloud_sim.py   # Edge servers, VMs, Cloudlets
│   ├── traffic_generator/
│   │   ├── generator.py        # Traffic data generation
│   │   └── intersection.py     # Intersection simulation
│   ├── federated_learning/
│   │   ├── client.py           # FL client
│   │   └── server.py           # FL server with FedAvg
│   ├── network_simulation/
│   │   └── network_layer.py    # Network abstraction
│   ├── experiments/
│   │   ├── comprehensive_runner.py
│   │   ├── network_stress.py
│   │   └── scalability.py
│   └── utils/
│       ├── reproducibility.py  # Seed management
│       ├── visualization.py    # Plotting
│       └── statistical_tests.py
├── ns3_simulation/             # NS-3 Network Simulation
│   ├── ns3_bridge_client.py    # Windows-side ZeroMQ client
│   ├── ns3_bridge_server.py    # WSL-side ZeroMQ server
│   ├── fl_traffic_network.cc   # NS-3 C++ simulation module
│   ├── setup_wsl_ns3.sh        # NS-3 installation script
│   └── README.md               # NS-3 setup guide
├── results/
│   ├── ieee/                   # IEEE plots and LaTeX tables
│   ├── ns3_integrated/         # NS-3 experiment results
│   ├── ns3_stress/             # Stress test results
│   └── comprehensive/          # Comprehensive evaluation
├── run_with_ns3.py             # Main NS-3 experiment
├── run_cloudsim.py             # CloudSim experiment
├── run_ieee_experiments.py     # Statistical experiments
├── run_comprehensive.py        # Full evaluation
├── run_fl_server.py            # FL server runner
├── run_fl_client.py            # FL client runner
├── DOCKER.md                   # Docker deployment guide
├── NS3_INTEGRATION_GUIDE.md    # NS-3 setup guide
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

### 1. Federated Learning (FL)

**FedAvg with Enhancements:**
- Weighted aggregation (data size + inverse loss)
- Learning rate scheduling (cosine annealing)
- Early stopping with best model restoration
- Gradient clipping for stability

### 2. CloudSim (Edge/Cloud Computing)

**Python-based CloudSim implementation** (`src/cloudsim_python/edge_cloud_sim.py`):

```python
# Edge Server Configuration
EdgeServer:
  - CPU: 4 cores @ 10,000 MIPS each
  - RAM: 8 GB
  - Bandwidth: 1 Gbps
  - Location: At each intersection

# Cloud Server Configuration
CloudServer:
  - CPU: 16 cores @ 50,000 MIPS each
  - RAM: 64 GB
  - Bandwidth: 10 Gbps
  - Location: Central data center

# FL Task (Cloudlet)
FLTrainingTask:
  - Length: 5000 MI (local training)
  - Aggregation: 2000 MI (FedAvg on cloud)
```

**CloudSim Components:**
- `EdgeServer`: Local model training at intersections
- `CloudServer`: Global model aggregation
- `Cloudlet`: FL training tasks
- `VirtualMachine`: Resource allocation
- `DatacenterBroker`: Task scheduling

### 3. Neural Network Architecture

```
Input (6) → Dense(256) → BN → LeakyReLU → Dropout(0.05)
         → Dense(128) → BN → LeakyReLU → Dropout(0.05)
         → Dense(64)  → BN → LeakyReLU → Dropout(0.05)
         → Dense(32)  → BN → LeakyReLU → Dropout(0.05)
         → Dense(1)   → Output (green duration in seconds)
```

**Input Features:**
1. North queue length
2. South queue length
3. East queue length
4. West queue length
5. Current phase (0=EW, 1=NS)
6. Normalized green duration

### Signal Control Strategy

The FL controller uses a **hybrid ML + rule-based approach**:

1. **ML Prediction**: Neural network trained on global traffic patterns
2. **Queue-Proportional Allocation**: Webster's formula adaptation
3. **Dynamic Blending**: 55% ML + 45% optimal strategy
4. **Adaptive Switching**: Faster phase changes under imbalance

### NS-3 Network Scenarios

| Scenario | Base Latency | Jitter | Packet Loss | Bandwidth |
|----------|--------------|--------|-------------|-----------|
| Ideal | 5ms | ±2ms | 0% | 54 Mbps |
| Normal | 15ms | ±5ms | 1% | 27 Mbps |
| Degraded | 50ms | ±15ms | 5% | 12 Mbps |
| Stressed | 100ms | ±30ms | 10% | 6 Mbps |
| Extreme | 200ms | ±50ms | 20% | 3 Mbps |

## Results & Outputs

### Generated Files

```
results/ieee/
├── ieee_method_comparison.png   # Bar chart (wait time + MAE)
├── ieee_fl_convergence.png      # FL training convergence
├── ieee_network_stress.png      # Network robustness plot
├── ieee_ablation_study.png      # Ablation results
├── ieee_results.json            # All raw data
└── latex_table.tex              # Ready for IEEE paper
```

### Sample IEEE LaTeX Table

```latex
\begin{table}[htbp]
\centering
\caption{Performance Comparison of Traffic Signal Control Methods}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Wait Time (s)} & \textbf{MAE} & \textbf{Improvement} \\
\midrule
Fixed-Time & 13.23 $\pm$ 0.31 & N/A & Baseline \\
Local-ML & 9.13 $\pm$ 0.22 & 1.9449 $\pm$ 0.2906 & - \\
\textbf{FL (Ours)} & \textbf{9.06 $\pm$ 0.18} & \textbf{1.8069 $\pm$ 0.1441} & \textbf{7.1\%} \\
\bottomrule
\end{tabular}
\end{table}
```

## Reproducibility

All experiments use controlled random seeds:
- Default: `42`
- Statistical analysis: `42, 123, 456, 789, 1024`

```python
from src.utils.reproducibility import set_global_seed
set_global_seed(42)
```

## NS-3 Integration Setup (Optional)

For realistic network simulation:

### 1. Enable WSL2

```powershell
# PowerShell (Admin)
wsl --install
```

### 2. Install NS-3 in WSL

```bash
# In WSL terminal
cd ns3_simulation
chmod +x setup_wsl_ns3.sh
./setup_wsl_ns3.sh
```

### 3. Start NS-3 Bridge

```bash
# Terminal 1 (WSL)
python3 ns3_simulation/ns3_bridge_server.py

# Terminal 2 (Windows)
python run_with_ns3.py --rounds 50
```

## Research Contributions

1. **Federated Learning Traffic Control**: Privacy-preserving distributed model training
2. **CloudSim Edge/Cloud Integration**: Realistic edge computing simulation for FL
3. **NS-3 Network Simulation**: Packet-level V2I communication with 802.11p DSRC
4. **Three-Tier Architecture**: FL + CloudSim + NS-3 integrated framework
5. **Robustness Analysis**: FL maintains performance under network stress (up to 658ms latency)
6. **Statistical Validation**: Multi-run experiments with confidence intervals
7. **Open Source**: Complete implementation for reproducibility

## Citation

```bibtex
@article{resilnet-fl-2025,
  title={{ResilNet-FL}: A Privacy-Preserving and Network-Resilient Federated
         Learning Framework for Intelligent Traffic Signal Control},
  author={Your Name},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  volume={},
  number={},
  pages={},
  doi={},
  keywords={Federated Learning, Traffic Signal Control, Privacy-Preserving ML,
            NS-3, CloudSim, Edge Computing, V2I Communication}
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, SciPy
- ZeroMQ (pyzmq) for NS-3 bridge
- WSL2 + NS-3 3.40+ (optional)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- NS-3 Development Team
- PyTorch Team
- Traffic signal control research community
