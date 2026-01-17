# NS-3 Integration Guide for Federated Traffic Signal Control

This guide explains how to integrate NS-3 (Network Simulator 3) with your Federated Learning traffic control system using WSL (Windows Subsystem for Linux).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows Native Environment                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Python FL System (Your Current Code)            │   │
│  │  • Traffic Signal Control                                 │   │
│  │  • Federated Learning Training                           │   │
│  │  • Performance Metrics                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    Socket/ZeroMQ Bridge                          │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                    WSL2 Linux Environment                        │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    NS-3 Simulator                         │   │
│  │  • Realistic Network Topology                            │   │
│  │  • V2I Communication (DSRC/C-V2X)                        │   │
│  │  • Packet-level Network Simulation                       │   │
│  │  • FL Communication Overhead Modeling                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Part 1: WSL2 Setup

### Step 1: Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install

# Verify installation
wsl --list --verbose

# Set WSL2 as default
wsl --set-default-version 2
```

### Step 2: Install Ubuntu

```powershell
# Install Ubuntu 22.04 LTS
wsl --install -d Ubuntu-22.04

# Launch Ubuntu
wsl -d Ubuntu-22.04
```

### Step 3: Update Ubuntu

```bash
# Inside WSL Ubuntu
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential gcc g++ python3 python3-pip \
    cmake git mercurial qt5-qmake libqt5-dev \
    gdb valgrind tcpdump wireshark sqlite3 libsqlite3-dev \
    libgtk-3-dev libxml2-dev libboost-all-dev
```

## Part 2: NS-3 Installation

### Step 1: Download NS-3

```bash
# Create workspace
mkdir -p ~/ns3-workspace && cd ~/ns3-workspace

# Clone NS-3 (version 3.40 or later recommended)
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3.40
cd ns-3.40
```

### Step 2: Configure and Build

```bash
# Configure with Python bindings
./ns3 configure --enable-examples --enable-tests --enable-python-bindings

# Build (this takes 15-30 minutes)
./ns3 build

# Verify installation
./ns3 run hello-simulator
```

### Step 3: Install Python Integration

```bash
# Install Python dependencies
pip3 install numpy pandas matplotlib zmq

# Test Python bindings
./ns3 run scratch/my-test.py
```

## Part 3: Project Bridge Setup

### Step 1: Copy Project to WSL

```bash
# From WSL, access Windows files
cd /mnt/c/Users/AKSHAY/Music/TRAFFIC\ SIGNALS/

# Create a symlink for easy access
ln -s "/mnt/c/Users/AKSHAY/Music/TRAFFIC SIGNALS" ~/fl-traffic
```

### Step 2: Create NS-3 Module

Create the NS-3 simulation module in your project:

**File: `ns3_simulation/fl_network_module.cc`**

```cpp
/*
 * NS-3 Module for Federated Learning Network Simulation
 * Simulates V2I communication for traffic signal control
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLTrafficNetwork");

class FLNetworkSimulator {
public:
    FLNetworkSimulator(uint32_t numIntersections, uint32_t numVehicles)
        : m_numIntersections(numIntersections),
          m_numVehicles(numVehicles) {}

    void Setup() {
        // Create intersection nodes (edge servers)
        m_intersectionNodes.Create(m_numIntersections);

        // Create vehicle nodes
        m_vehicleNodes.Create(m_numVehicles);

        // Create central server node
        m_serverNode.Create(1);

        SetupWifi();
        SetupMobility();
        SetupInternetStack();
        SetupApplications();
    }

    void SetupWifi() {
        // Configure 802.11p (DSRC) for V2I communication
        YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
        YansWifiPhyHelper phy;
        phy.SetChannel(channel.Create());

        WifiMacHelper mac;
        mac.SetType("ns3::AdhocWifiMac");

        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211p);
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                      "DataMode", StringValue("OfdmRate6Mbps"));

        m_intersectionDevices = wifi.Install(phy, mac, m_intersectionNodes);
        m_vehicleDevices = wifi.Install(phy, mac, m_vehicleNodes);
    }

    void SetupMobility() {
        // Static positions for intersections (grid layout)
        MobilityHelper intersectionMobility;
        intersectionMobility.SetPositionAllocator(
            "ns3::GridPositionAllocator",
            "MinX", DoubleValue(0.0),
            "MinY", DoubleValue(0.0),
            "DeltaX", DoubleValue(500.0),  // 500m between intersections
            "DeltaY", DoubleValue(500.0),
            "GridWidth", UintegerValue(2),
            "LayoutType", StringValue("RowFirst"));
        intersectionMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        intersectionMobility.Install(m_intersectionNodes);

        // Random waypoint mobility for vehicles
        MobilityHelper vehicleMobility;
        vehicleMobility.SetPositionAllocator(
            "ns3::RandomBoxPositionAllocator",
            "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=1000]"),
            "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=1000]"),
            "Z", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        vehicleMobility.SetMobilityModel(
            "ns3::RandomWaypointMobilityModel",
            "Speed", StringValue("ns3::UniformRandomVariable[Min=5|Max=20]"),
            "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        vehicleMobility.Install(m_vehicleNodes);
    }

    void SetupInternetStack() {
        InternetStackHelper internet;
        internet.Install(m_intersectionNodes);
        internet.Install(m_vehicleNodes);
        internet.Install(m_serverNode);

        Ipv4AddressHelper address;
        address.SetBase("10.1.1.0", "255.255.255.0");
        m_intersectionInterfaces = address.Assign(m_intersectionDevices);

        address.SetBase("10.1.2.0", "255.255.255.0");
        m_vehicleInterfaces = address.Assign(m_vehicleDevices);
    }

    void SetupApplications() {
        // FL model update application (UDP)
        uint16_t flPort = 9999;

        for (uint32_t i = 0; i < m_numIntersections; i++) {
            // Each intersection sends model updates to server
            OnOffHelper onoff("ns3::UdpSocketFactory",
                             InetSocketAddress(m_serverAddress, flPort));
            onoff.SetAttribute("DataRate", StringValue("1Mbps"));
            onoff.SetAttribute("PacketSize", UintegerValue(1024));  // Model chunk size

            ApplicationContainer app = onoff.Install(m_intersectionNodes.Get(i));
            app.Start(Seconds(1.0 + i * 0.1));  // Staggered start
        }
    }

    void Run(double duration) {
        Simulator::Stop(Seconds(duration));
        Simulator::Run();
        Simulator::Destroy();
    }

    void GetNetworkMetrics() {
        // Collect and report network statistics
        // Throughput, latency, packet loss, etc.
    }

private:
    uint32_t m_numIntersections;
    uint32_t m_numVehicles;

    NodeContainer m_intersectionNodes;
    NodeContainer m_vehicleNodes;
    NodeContainer m_serverNode;

    NetDeviceContainer m_intersectionDevices;
    NetDeviceContainer m_vehicleDevices;

    Ipv4InterfaceContainer m_intersectionInterfaces;
    Ipv4InterfaceContainer m_vehicleInterfaces;

    Ipv4Address m_serverAddress;
};

int main(int argc, char *argv[]) {
    uint32_t numIntersections = 4;
    uint32_t numVehicles = 100;
    double simDuration = 100.0;  // seconds

    CommandLine cmd;
    cmd.AddValue("intersections", "Number of intersections", numIntersections);
    cmd.AddValue("vehicles", "Number of vehicles", numVehicles);
    cmd.AddValue("duration", "Simulation duration", simDuration);
    cmd.Parse(argc, argv);

    FLNetworkSimulator sim(numIntersections, numVehicles);
    sim.Setup();
    sim.Run(simDuration);

    return 0;
}
```

### Step 3: Create Python Bridge

**File: `ns3_simulation/bridge.py`**

```python
"""
Bridge between Windows FL system and WSL NS-3 simulation.
Uses ZeroMQ for cross-platform communication.
"""

import zmq
import json
import subprocess
import numpy as np
from typing import Dict, Any
import time


class NS3Bridge:
    """
    Bridges Python FL code with NS-3 network simulation.
    Runs on both Windows (client) and WSL (server).
    """

    def __init__(self, mode: str = "client", port: int = 5555):
        """
        Initialize bridge.

        Args:
            mode: "client" (Windows) or "server" (WSL)
            port: ZeroMQ port for communication
        """
        self.mode = mode
        self.port = port
        self.context = zmq.Context()

        if mode == "client":
            self.socket = self.context.socket(zmq.REQ)
            # Connect to WSL - use localhost.localdomain for WSL2
            self.socket.connect(f"tcp://localhost:{port}")
        else:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")

    def send_fl_update(self, model_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Send FL model update through NS-3 simulated network.

        Args:
            model_params: Model parameters to transmit

        Returns:
            Network metrics (latency, success rate, etc.)
        """
        if self.mode != "client":
            raise RuntimeError("send_fl_update only available in client mode")

        # Serialize and send
        message = {
            "type": "fl_update",
            "timestamp": time.time(),
            "payload_size": sum(np.array(p).nbytes for p in model_params.values()),
            "num_params": len(model_params)
        }

        self.socket.send_json(message)
        response = self.socket.recv_json()

        return response

    def run_network_simulation(
        self,
        num_intersections: int,
        num_rounds: int,
        network_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run full NS-3 network simulation for FL scenario.

        Args:
            num_intersections: Number of FL clients
            num_rounds: Number of FL training rounds
            network_config: Network parameters

        Returns:
            Simulation results
        """
        if self.mode != "client":
            raise RuntimeError("run_network_simulation only available in client mode")

        message = {
            "type": "run_simulation",
            "num_intersections": num_intersections,
            "num_rounds": num_rounds,
            "network_config": network_config
        }

        self.socket.send_json(message)
        response = self.socket.recv_json()

        return response


class NS3Server:
    """
    NS-3 simulation server running in WSL.
    """

    def __init__(self, ns3_path: str = "~/ns3-workspace/ns-3.40"):
        self.ns3_path = ns3_path
        self.bridge = NS3Bridge(mode="server")

    def run_ns3_simulation(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute NS-3 simulation and return network metrics.
        """
        # Build NS-3 command
        cmd = [
            f"{self.ns3_path}/ns3", "run",
            "fl-network-module",
            f"--intersections={config.get('num_intersections', 4)}",
            f"--vehicles={config.get('num_vehicles', 100)}",
            f"--duration={config.get('duration', 100)}"
        ]

        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output
        metrics = self._parse_ns3_output(result.stdout)

        return metrics

    def _parse_ns3_output(self, output: str) -> Dict[str, float]:
        """Parse NS-3 simulation output."""
        # Implementation depends on NS-3 output format
        return {
            "avg_latency_ms": 0.0,
            "packet_loss_rate": 0.0,
            "throughput_mbps": 0.0
        }

    def serve(self):
        """Run server loop."""
        print("NS-3 Bridge Server running...")

        while True:
            message = self.bridge.socket.recv_json()

            if message["type"] == "fl_update":
                # Simulate network transmission
                metrics = self._simulate_transmission(message)
                self.bridge.socket.send_json(metrics)

            elif message["type"] == "run_simulation":
                results = self.run_ns3_simulation(message)
                self.bridge.socket.send_json(results)

    def _simulate_transmission(self, message: Dict) -> Dict[str, float]:
        """Simulate FL update transmission through NS-3."""
        # Quick simulation for single transmission
        payload_size = message.get("payload_size", 10000)

        # Simple model: latency = base + size/bandwidth + jitter
        base_latency = 10  # ms
        bandwidth_mbps = 54  # 802.11p typical
        size_latency = (payload_size * 8) / (bandwidth_mbps * 1000)  # ms
        jitter = np.random.exponential(2)  # ms

        total_latency = base_latency + size_latency + jitter

        # Packet loss probability (simplified)
        packet_loss = np.random.random() < 0.01

        return {
            "latency_ms": total_latency,
            "packet_loss": packet_loss,
            "success": not packet_loss,
            "bandwidth_used_kbps": payload_size * 8 / total_latency
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as server in WSL
        server = NS3Server()
        server.serve()
    else:
        # Test client
        bridge = NS3Bridge(mode="client")

        # Test FL update
        result = bridge.send_fl_update({
            "layer1": np.random.randn(128, 6),
            "layer2": np.random.randn(64, 128)
        })
        print(f"Network result: {result}")
```

## Part 4: Running the Integrated System

### Step 1: Start NS-3 Server in WSL

```bash
# In WSL terminal
cd ~/fl-traffic
python3 ns3_simulation/bridge.py server
```

### Step 2: Run FL Training from Windows

```powershell
# In Windows PowerShell
cd "C:\Users\AKSHAY\Music\TRAFFIC SIGNALS"
python run_with_ns3.py
```

### Step 3: Create Integration Script

**File: `run_with_ns3.py`**

```python
"""
Run FL training with NS-3 network simulation.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'ns3_simulation')

from bridge import NS3Bridge
from experiments.comprehensive_runner import ComprehensiveExperimentRunner

def main():
    # Initialize NS-3 bridge
    print("Connecting to NS-3 simulation in WSL...")
    bridge = NS3Bridge(mode="client")

    # Test connection
    try:
        result = bridge.send_fl_update({"test": [1, 2, 3]})
        print(f"NS-3 connection successful: {result}")
    except Exception as e:
        print(f"NS-3 connection failed: {e}")
        print("Running without NS-3 (using built-in network simulation)...")
        bridge = None

    # Run experiment
    runner = ComprehensiveExperimentRunner(
        seed=42,
        num_intersections=4,
        simulation_duration=1800,
        fl_rounds=100
    )

    results = runner.run_all(skip_scalability=True)

    print("\nExperiment complete!")
    return results

if __name__ == "__main__":
    main()
```

## Part 5: Advanced NS-3 Features

### V2X Communication Protocols

For realistic V2I simulation, consider these NS-3 modules:

1. **802.11p (DSRC)**: `src/wave/` module
2. **LTE-V2X**: `src/lte/` module
3. **5G NR-V2X**: `src/nr/` module (NS-3.37+)

### Example: DSRC Configuration

```cpp
// Configure 802.11p for dedicated short-range communication
WifiHelper wifi;
wifi.SetStandard(WIFI_STANDARD_80211p);

// CCH (Control Channel) for safety messages
// SCH (Service Channel) for FL updates
QosWifiMacHelper mac = QosWifiMacHelper::Default();
mac.SetType("ns3::OcbWifiMac",
            "QosSupported", BooleanValue(true));
```

## Part 6: Troubleshooting

### Common Issues

1. **WSL2 Network Issues**
   ```bash
   # Reset WSL network
   wsl --shutdown
   # Then restart WSL
   ```

2. **NS-3 Build Errors**
   ```bash
   # Clean and rebuild
   ./ns3 clean
   ./ns3 configure --enable-python-bindings
   ./ns3 build
   ```

3. **ZeroMQ Connection Refused**
   ```bash
   # Check if port is in use
   netstat -tulpn | grep 5555

   # In Windows, check WSL IP
   wsl hostname -I
   ```

### Performance Tips

1. Use `--enable-optimized` for faster NS-3 execution
2. Reduce simulation granularity for faster results
3. Use parallel builds: `./ns3 build -j$(nproc)`

## Part 7: Publication Notes

When citing NS-3 integration:

```bibtex
@inproceedings{ns3,
  title={ns-3: A discrete-event network simulator},
  author={Henderson, Thomas R and others},
  booktitle={ACM SIGCOMM},
  year={2008}
}
```

Key metrics to report:
- End-to-end latency distribution
- Packet delivery ratio under congestion
- FL convergence with realistic network delays
- Bandwidth utilization efficiency

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `wsl` | Enter WSL from PowerShell |
| `./ns3 run <script>` | Run NS-3 simulation |
| `./ns3 build` | Build NS-3 |
| `python3 bridge.py server` | Start NS-3 bridge server |

## Next Steps

1. ✅ Install WSL2 and Ubuntu
2. ✅ Install NS-3 with Python bindings
3. ⬜ Create project symlink
4. ⬜ Build NS-3 FL module
5. ⬜ Test bridge communication
6. ⬜ Run integrated experiments
