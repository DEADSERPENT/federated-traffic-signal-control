# Docker Guide - Federated Learning Traffic Signal Control

## Prerequisites

1. **Docker Desktop** installed and running
   - Download: https://www.docker.com/products/docker-desktop/
   - Ensure Docker Desktop is **running** before executing commands

2. **Minimum Requirements**
   - RAM: 4GB allocated to Docker
   - Disk: 5GB free space

## Quick Start

### Windows (CMD/PowerShell)
```batch
# Build and run demo
docker-run.bat demo

# Or using docker-compose directly
docker-compose up demo
```

### Linux/Mac
```bash
# Make script executable
chmod +x docker-run.sh

# Run demo
./docker-run.sh demo

# Or using docker-compose directly
docker-compose up demo
```

## Available Commands

| Command | Description | Time |
|---------|-------------|------|
| `demo` | Quick comprehensive experiment | ~5 min |
| `full` | Full experiment with scalability | ~15 min |
| `fl` | FL Server + 4 Clients | ~10 min |
| `simulation` | Traffic simulation only | ~2 min |
| `cloudsim` | Edge/Cloud simulation | ~1 min |
| `build` | Build Docker image | ~3 min |
| `clean` | Clean up containers | ~10 sec |

## Detailed Usage

### 1. Run Comprehensive Demo (Recommended for First Run)
```bash
# Windows
docker-run.bat demo

# Linux/Mac
./docker-run.sh demo
```

This runs:
- Baseline comparisons (Fixed-Time, Local-ML, FL)
- Network stress experiments
- Generates visualizations and reports

**Output:** `results/comprehensive/`

### 2. Run Full Federated Learning Training
```bash
# Start FL server and 4 clients
docker-compose up fl-server fl-client-0 fl-client-1 fl-client-2 fl-client-3
```

This simulates:
- 1 FL Server (aggregator)
- 4 FL Clients (traffic intersections)
- 20 rounds of federated averaging

### 3. Run Individual Components

```bash
# Traffic simulation only
docker-compose up simulation

# CloudSim edge/cloud simulation
docker-compose up cloudsim

# Full experiment (includes scalability)
docker-compose up full-experiment
```

### 4. Build Image Manually
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build demo
```

### 5. View Logs
```bash
# View logs for a service
docker-compose logs demo

# Follow logs in real-time
docker-compose logs -f fl-server
```

### 6. Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove all containers and unused images
docker-run.bat clean
# or
./docker-run.sh clean
```

## Output Files

After running experiments, results are saved to:

```
results/
├── comprehensive/
│   ├── fl_convergence.png        # FL training plot
│   ├── method_comparison.png     # Baseline comparison
│   ├── network_stress.png        # Network analysis
│   ├── summary_dashboard.png     # Overview dashboard
│   ├── experiment_report.txt     # Text report
│   └── results.json              # Raw data
├── traffic_metrics.png           # Traffic visualization
└── logs/                         # Experiment logs
```

## Troubleshooting

### Docker Desktop Not Running
```
Error: Cannot connect to Docker daemon
```
**Solution:** Start Docker Desktop application

### Port Already in Use
```
Error: port 8080 already in use
```
**Solution:**
```bash
docker-compose down
# or change port in docker-compose.yml
```

### Out of Memory
```
Error: Container killed (OOM)
```
**Solution:** Increase Docker memory in Docker Desktop Settings > Resources

### Build Fails
```bash
# Clean build cache and retry
docker-compose build --no-cache
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   FL Server                           │   │
│  │              (Model Aggregator)                       │   │
│  │                  Port: 8080                           │   │
│  └──────────────────────────────────────────────────────┘   │
│           ▲              ▲              ▲              ▲     │
│           │              │              │              │     │
│  ┌────────┴───┐  ┌───────┴────┐ ┌──────┴─────┐ ┌─────┴────┐ │
│  │ FL Client 0│  │ FL Client 1│ │ FL Client 2│ │FL Client 3│ │
│  │ (Intersect)│  │ (Intersect)│ │ (Intersect)│ │(Intersect)│ │
│  └────────────┘  └────────────┘ └────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## For Examiners/Presentations

**One-Command Demo:**
```bash
docker-compose up demo
```

This will:
1. Build the container (first time only)
2. Run all experiments
3. Generate publication-quality figures
4. Save results to `results/comprehensive/`

**Time:** ~5 minutes

**No Python/dependencies needed** - everything runs in container!
