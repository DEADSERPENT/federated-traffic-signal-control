# NS-3 Integration for Federated Learning Traffic Control

This directory contains everything needed to integrate NS-3 (Network Simulator 3) with the Federated Learning traffic signal control system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Windows Environment                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Python FL System                             │  │
│  │  ├── Traffic Signal Control                                   │  │
│  │  ├── Federated Learning Training                             │  │
│  │  └── NS-3 Bridge Client (ns3_bridge_client.py)               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│                         ZeroMQ (tcp://localhost:5555)               │
│                                  │                                   │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────┐
│                         WSL2 / Linux                                 │
│                                  │                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              NS-3 Bridge Server (ns3_bridge_server.py)         │  │
│  │  ├── Handles FL simulation requests                           │  │
│  │  ├── Executes NS-3 simulations                               │  │
│  │  └── Returns network metrics                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│                           subprocess                                 │
│                                  │                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    NS-3 Simulator                              │  │
│  │  ├── 802.11p/WAVE (DSRC) V2I Communication                   │  │
│  │  ├── Realistic Urban Network Topology                         │  │
│  │  └── Accurate Latency/Packet Loss Modeling                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: With NS-3 (Full Simulation)

#### Step 1: Setup WSL and NS-3

```bash
# In WSL/Ubuntu terminal
cd /mnt/c/Users/AKSHAY/Music/TRAFFIC\ SIGNALS/ns3_simulation
chmod +x setup_wsl_ns3.sh
./setup_wsl_ns3.sh
```

#### Step 2: Copy NS-3 Module

```bash
# After NS-3 installation
cp fl_traffic_network.cc ~/ns3-fl-traffic/ns-3.3.40/scratch/
cd ~/ns3-fl-traffic/ns-3.3.40
./ns3 build
```

#### Step 3: Start Bridge Server (WSL)

```bash
# Terminal 1 (WSL)
python3 /mnt/c/Users/AKSHAY/Music/TRAFFIC\ SIGNALS/ns3_simulation/ns3_bridge_server.py
```

#### Step 4: Run FL Training (Windows)

```powershell
# Terminal 2 (Windows PowerShell)
cd "C:\Users\AKSHAY\Music\TRAFFIC SIGNALS"
python run_with_ns3.py
```

### Option 2: Without NS-3 (Fallback Simulation)

If you don't want to set up NS-3, the system automatically uses a statistical network simulation:

```powershell
python run_with_ns3.py --no-ns3
```

## Files

| File | Location | Description |
|------|----------|-------------|
| `setup_wsl_ns3.sh` | WSL | Installs NS-3 and dependencies |
| `fl_traffic_network.cc` | NS-3 scratch | C++ V2I network simulation |
| `ns3_bridge_server.py` | WSL | ZeroMQ server handling requests |
| `ns3_bridge_client.py` | Windows | Client for Windows FL system |
| `run_with_ns3.py` | Windows | Integrated FL+NS-3 runner |

## Network Scenarios

The system supports different network conditions:

| Scenario | Latency | Packet Loss | Bandwidth | Use Case |
|----------|---------|-------------|-----------|----------|
| `ideal` | 5ms | 0% | 54 Mbps | Best-case V2I |
| `normal` | 15ms | 1% | 27 Mbps | Typical urban |
| `degraded` | 50ms | 5% | 12 Mbps | Congested |
| `stressed` | 100ms | 10% | 6 Mbps | Heavy load |
| `extreme` | 200ms | 20% | 3 Mbps | Worst-case |

## Commands

### Run with specific network scenario

```powershell
python run_with_ns3.py --scenario degraded
```

### Run network stress test (all scenarios)

```powershell
python run_with_ns3.py --stress-test
```

### Specify FL training rounds

```powershell
python run_with_ns3.py --rounds 100
```

## Troubleshooting

### "Could not connect to bridge server"

1. Ensure WSL is running: `wsl --list --running`
2. Start the bridge server in WSL first
3. Check firewall settings allow localhost:5555

### NS-3 build errors

```bash
cd ~/ns3-fl-traffic/ns-3.3.40
./ns3 clean
./ns3 configure --enable-examples
./ns3 build
```

### ZeroMQ import error

```bash
pip3 install pyzmq  # WSL
pip install pyzmq   # Windows
```

## For Different Devices (Linux Machine)

If running NS-3 on a separate Linux machine:

1. **On Linux machine:**
   ```bash
   # Install NS-3
   ./setup_wsl_ns3.sh

   # Start server with public IP
   python3 ns3_bridge_server.py --port 5555
   ```

2. **On Windows machine:**
   ```python
   # In your code, specify the Linux machine's IP
   from ns3_simulation.ns3_bridge_client import NS3Client
   client = NS3Client(host="192.168.1.100", port=5555)
   ```

3. **Configure firewall on Linux:**
   ```bash
   sudo ufw allow 5555/tcp
   ```

## Publication Notes

When citing the NS-3 integration in your paper:

```bibtex
@inproceedings{riley2010ns,
  title={The ns-3 network simulator},
  author={Riley, George F and Henderson, Thomas R},
  booktitle={Modeling and tools for network simulation},
  pages={15--34},
  year={2010},
  publisher={Springer}
}
```

Key metrics to report:
- V2I communication latency distribution
- Packet delivery ratio under various traffic loads
- FL convergence with realistic network delays
- Communication overhead (bytes transferred per FL round)
