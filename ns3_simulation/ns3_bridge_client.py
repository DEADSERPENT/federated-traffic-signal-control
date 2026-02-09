#!/usr/bin/env python3
"""
NS-3 Bridge Client (Runs on Windows)

This client connects to the NS-3 bridge server running in WSL
and integrates realistic network simulation with FL training.

Features:
- Connects to NS-3 bridge server when available
- Falls back to trace-driven simulation using IEEE 802.11p/3GPP V2X parameters
- Provides scientifically-grounded network characteristics

References:
- IEEE 802.11p DSRC: 6Mbps bandwidth, Rayleigh fading channel model
- 3GPP LTE-V2X/NR-V2X specifications for V2I communications
- Kenney, J.B. "Dedicated Short-Range Communications" (IEEE 2011)

Usage (in Windows):
    from ns3_simulation.ns3_bridge_client import NS3Client
    client = NS3Client()
    metrics = client.simulate_fl_round(model_params, num_clients=4)
"""

import os
import sys
import time
import json
import socket
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import zmq
except ImportError:
    print("Installing pyzmq...")
    os.system("pip install pyzmq")
    import zmq

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np

# Import trace-driven simulation
try:
    from ns3_simulation.network_traces import NetworkTraceGenerator, NetworkType
except ImportError:
    # Fallback if running as main
    from network_traces import NetworkTraceGenerator, NetworkType


@dataclass
class NetworkMetrics:
    """Metrics from NS-3 network simulation."""
    avg_latency_ms: float
    packet_loss_rate: float
    throughput_mbps: float
    successful_clients: int
    total_clients: int
    per_client_latencies: List[float] = field(default_factory=list)
    simulation_source: str = "simulated"

    def get_success_rate(self) -> float:
        return self.successful_clients / max(self.total_clients, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": self.avg_latency_ms,
            "packet_loss_rate": self.packet_loss_rate,
            "throughput_mbps": self.throughput_mbps,
            "successful_clients": self.successful_clients,
            "total_clients": self.total_clients,
            "success_rate": self.get_success_rate(),
            "source": self.simulation_source
        }


class NS3Client:
    """
    Client for communicating with NS-3 Bridge Server.

    Automatically handles connection to WSL and provides
    realistic network metrics for FL training.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 3000,  # 3 seconds for faster fallback
        auto_fallback: bool = True
    ):
        """
        Initialize NS-3 client.

        Args:
            host: Bridge server host (localhost for WSL on same machine)
            port: ZeroMQ port
            timeout_ms: Request timeout in milliseconds
            auto_fallback: Use local simulation if server unavailable
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.auto_fallback = auto_fallback
        self.connected = False

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)

        # Statistics
        self.stats = {
            "requests_sent": 0,
            "requests_failed": 0,
            "fallbacks_used": 0,
            "total_latency_ms": 0.0
        }

        # Initialize trace-driven simulator for fallback
        # Uses IEEE 802.11p DSRC / 3GPP V2X parameters
        self.trace_generator = NetworkTraceGenerator(
            network_type=NetworkType.HYBRID,
            seed=42
        )

        # Try to connect
        self._connect()

    def _connect(self) -> bool:
        """Attempt to connect to bridge server."""
        try:
            address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(address)

            # Test connection with ping
            self.socket.send_json({"type": "ping"})
            response = self.socket.recv_json()

            if response.get("status") == "ok":
                self.connected = True
                print(f"[NS3Client] Connected to bridge server at {address}")
                print(f"[NS3Client] NS-3 available: {response.get('ns3_available', False)}")
                return True

        except zmq.ZMQError as e:
            print(f"[NS3Client] Could not connect to bridge server: {e}")
            if self.auto_fallback:
                print("[NS3Client] Will use local simulation fallback")
            self.connected = False

        return False

    def reconnect(self) -> bool:
        """Attempt to reconnect to server."""
        # Close existing socket
        self.socket.close()

        # Create new socket
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)

        return self._connect()

    def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to bridge server."""
        if not self.connected:
            if not self.reconnect():
                return None

        try:
            start_time = time.time()
            self.socket.send_json(request)
            response = self.socket.recv_json()
            elapsed = (time.time() - start_time) * 1000

            self.stats["requests_sent"] += 1
            self.stats["total_latency_ms"] += elapsed

            return response

        except zmq.ZMQError as e:
            self.stats["requests_failed"] += 1
            self.connected = False
            print(f"[NS3Client] Request failed: {e}")
            return None

    def simulate_fl_round(
        self,
        model_params: List[np.ndarray],
        num_clients: int = 4,
        network_scenario: str = "normal"
    ) -> NetworkMetrics:
        """
        Simulate one FL training round through the network.

        Args:
            model_params: Model parameters being transmitted
            num_clients: Number of FL clients
            network_scenario: Network condition scenario

        Returns:
            NetworkMetrics with simulation results
        """
        # Calculate payload size
        payload_size = sum(p.nbytes for p in model_params)

        request = {
            "type": "fl_update",
            "payload_size": payload_size,
            "num_clients": num_clients,
            "network_scenario": network_scenario,
            "timestamp": time.time()
        }

        response = self._send_request(request)

        if response is None:
            # Fallback to local simulation
            self.stats["fallbacks_used"] += 1
            return self._local_simulation(payload_size, num_clients, network_scenario)

        return NetworkMetrics(
            avg_latency_ms=response.get("avg_latency_ms", 10.0),
            packet_loss_rate=response.get("packet_loss_rate", 0.01),
            throughput_mbps=response.get("bandwidth_used_kbps", 1000) / 1000,
            successful_clients=response.get("successful_clients", num_clients),
            total_clients=num_clients,
            per_client_latencies=[
                c["latency_ms"] for c in response.get("client_results", [])
            ],
            simulation_source="ns3_bridge" if response.get("simulated") else "ns3"
        )

    def run_full_simulation(
        self,
        num_intersections: int = 4,
        num_vehicles: int = 50,
        duration: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run full NS-3 network simulation.

        Args:
            num_intersections: Number of RSU nodes
            num_vehicles: Number of vehicle nodes
            duration: Simulation duration in seconds

        Returns:
            Complete simulation results
        """
        request = {
            "type": "run_simulation",
            "num_intersections": num_intersections,
            "num_vehicles": num_vehicles,
            "duration": duration
        }

        response = self._send_request(request)

        if response is None:
            self.stats["fallbacks_used"] += 1
            return self._local_full_simulation(num_intersections, num_vehicles, duration)

        return response

    def run_network_stress_test(
        self,
        num_intersections: int = 4,
        scenarios: List[str] = None
    ) -> Dict[str, NetworkMetrics]:
        """
        Run network stress test across different scenarios.

        Args:
            num_intersections: Number of clients
            scenarios: List of scenario names

        Returns:
            Dictionary mapping scenario name to metrics
        """
        if scenarios is None:
            scenarios = ["ideal", "normal", "degraded", "stressed", "extreme"]

        results = {}
        dummy_params = [np.random.randn(128, 6).astype(np.float32)]

        for scenario in scenarios:
            metrics = self.simulate_fl_round(
                dummy_params,
                num_clients=num_intersections,
                network_scenario=scenario
            )
            results[scenario] = metrics
            print(f"  {scenario}: latency={metrics.avg_latency_ms:.1f}ms, "
                  f"loss={metrics.packet_loss_rate*100:.1f}%")

        return results

    def _local_simulation(
        self,
        payload_size: int,
        num_clients: int,
        scenario: str
    ) -> NetworkMetrics:
        """
        Trace-driven fallback simulation when bridge is unavailable.

        Uses scientifically-grounded network models based on:
        - IEEE 802.11p DSRC: 6Mbps bandwidth, 2-20ms latency
        - 3GPP LTE-V2X/NR-V2X specifications
        - Rayleigh fading channel model for V2I communications

        Reference: Kenney, J.B. "Dedicated Short-Range Communications
                   (DSRC) Standards in the United States" IEEE 2011
        """
        # Use trace-driven generator for realistic simulation
        result = self.trace_generator.simulate_fl_round(
            payload_size_bytes=payload_size,
            num_clients=num_clients,
            scenario=scenario
        )

        # Extract per-client latencies
        latencies = [
            c["latency_ms"] for c in result["client_results"]
            if not c["packet_lost"]
        ]

        return NetworkMetrics(
            avg_latency_ms=result["avg_latency_ms"],
            packet_loss_rate=result["packet_loss_rate"],
            throughput_mbps=result["throughput_mbps"],
            successful_clients=result["successful_clients"],
            total_clients=result["total_clients"],
            per_client_latencies=latencies if latencies else [result["avg_latency_ms"]],
            simulation_source="trace_driven_802.11p_3GPP"
        )

    def _local_full_simulation(
        self,
        num_intersections: int,
        num_vehicles: int,
        duration: float
    ) -> Dict[str, Any]:
        """
        Trace-driven full simulation fallback.

        Models 802.11p DSRC characteristics:
        - 6Mbps effective throughput
        - Rayleigh fading channel model
        - CSMA/CA MAC layer behavior
        """
        # Congestion increases with vehicle density
        congestion = min(0.9, num_vehicles / 100)

        # Generate multiple FL rounds worth of traffic
        num_rounds = int(duration / 5)  # One round per 5 seconds
        all_latencies = []
        total_lost = 0

        for _ in range(num_rounds):
            result = self.trace_generator.simulate_fl_round(
                payload_size_bytes=50000,  # Typical model update
                num_clients=num_intersections,
                scenario="normal" if congestion < 0.5 else "degraded"
            )
            all_latencies.extend([
                c["latency_ms"] for c in result["client_results"]
            ])
            total_lost += num_intersections - result["successful_clients"]

        packets_sent = num_intersections * num_rounds
        packets_received = packets_sent - total_lost

        return {
            "status": "ok",
            "source": "trace_driven_802.11p_3GPP",
            "reference": "IEEE 802.11p DSRC / 3GPP V2X",
            "intersections": num_intersections,
            "vehicles": num_vehicles,
            "duration_s": duration,
            "packets_sent": packets_sent,
            "packets_received": packets_received,
            "packet_loss_rate": total_lost / max(packets_sent, 1),
            "avg_latency_ms": float(np.mean(all_latencies)) if all_latencies else 15.0,
            "throughput_mbps": (packets_received * 50000 * 8) / (duration * 1e6)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            **self.stats
        }

    def close(self):
        """Close connection."""
        self.socket.close()
        self.context.term()


def test_client():
    """Test the NS-3 client."""
    print("=" * 60)
    print("  NS-3 Bridge Client Test")
    print("=" * 60)

    client = NS3Client(auto_fallback=True)

    print("\n1. Testing FL round simulation...")
    dummy_params = [np.random.randn(128, 6).astype(np.float32)]
    metrics = client.simulate_fl_round(dummy_params, num_clients=4)
    print(f"   Latency: {metrics.avg_latency_ms:.2f}ms")
    print(f"   Success rate: {metrics.get_success_rate()*100:.1f}%")
    print(f"   Source: {metrics.simulation_source}")

    print("\n2. Testing network stress...")
    stress_results = client.run_network_stress_test(num_intersections=4)

    print("\n3. Client statistics:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    client.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_client()
