# Federated Learning-based Adaptive Traffic Signal Control System

A privacy-preserving adaptive traffic signal control system using **Federated Learning**, edge computing simulation, and cloud-based model aggregation.

## Key Features

- **Federated Learning** - Privacy-preserving model training (no raw data sharing)
- **Edge Computing** - Each intersection trains locally
- **Baseline Comparisons** - Fixed-Time, Local-ML, and FL methods
- **Network Resilience** - Works under degraded network conditions
- **Docker Support** - One-command reproducible execution
- **Publication-Quality Visualizations** - Ready for thesis/papers

## Quick Start

### Option 1: Docker (Recommended)

```bash
# One command - runs everything!
docker-compose up demo
```

Results saved to `results/comprehensive/`

### Option 2: Python Virtual Environment

```bash
# Setup
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Run comprehensive experiment
python run_comprehensive.py
```

## Project Structure

```
TRAFFIC SIGNALS/
├── config/config.yaml           # Configuration
├── src/
│   ├── traffic_generator/       # Traffic simulation
│   ├── federated_learning/      # FL server & clients
│   ├── network_simulation/      # Network abstraction
│   ├── models/                  # PyTorch ML models
│   ├── baselines/               # Comparison methods
│   ├── experiments/             # Experiment runners
│   └── utils/                   # Visualization & metrics
├── results/comprehensive/       # Output visualizations
├── Dockerfile                   # Docker container
├── docker-compose.yml           # Multi-service orchestration
├── run_comprehensive.py         # Full experiment (one command)
├── run_fl_server.py             # FL server
├── run_fl_client.py             # FL client
└── requirements.txt             # Dependencies
```

## Experiments

### 1. Comprehensive Experiment (Recommended)

```bash
python run_comprehensive.py        # Full (~15 min)
python run_comprehensive.py --quick  # Quick (~5 min)
```

**Includes:**
- Baseline comparison (Fixed-Time vs Local-ML vs FL)
- Network stress tests (latency, packet loss)
- Scalability analysis (2-8 clients)
- Publication-quality figures

### 2. Federated Learning Only

**Terminal 1 - Server:**
```bash
python run_fl_server.py --rounds 30 --min-clients 2
```

**Terminal 2+ - Clients:**
```bash
python run_fl_client.py --intersection 0
python run_fl_client.py --intersection 1
```

### 3. Traffic Simulation

```bash
python run_simulation.py
```

## Results

| Method | MAE | Waiting Time | Privacy |
|--------|-----|--------------|---------|
| Fixed-Time | N/A | 13.35s | N/A |
| Local-ML | 7.27 | 9.72s | No |
| **Federated Learning** | **6.49** | 12.14s | **Yes** |

**FL achieves 10.7% better accuracy than Local-ML while preserving privacy!**

## Generated Visualizations

After running `run_comprehensive.py`:

```
results/comprehensive/
├── fl_convergence.png        # Training convergence
├── method_comparison.png     # Baseline comparison
├── network_stress.png        # Network resilience
├── summary_dashboard.png     # Complete overview
├── experiment_report.txt     # Text report
└── results.json              # Raw data
```

## Docker Commands

```bash
docker-compose up demo              # Quick demo
docker-compose up full-experiment   # Full experiment
docker-compose up fl-server fl-client-0 fl-client-1  # FL training
docker-compose down                 # Cleanup
```

See [DOCKER.md](DOCKER.md) for detailed Docker guide.

## Configuration

Edit `config/config.yaml`:

```yaml
traffic:
  num_intersections: 4
  simulation_duration: 3600

federated_learning:
  num_rounds: 30
  local_epochs: 5
  min_clients: 2

network:
  base_latency: 10
  packet_loss_probability: 0.01
```

## Requirements

- Python 3.8+
- Docker (optional, recommended)
- 4GB RAM minimum

## Citation

If using this project for academic work:

```
Federated Learning-based Adaptive Traffic Signal Control System
Privacy-preserving optimization using edge computing and FedAvg
```

## License

Academic/Educational Use
