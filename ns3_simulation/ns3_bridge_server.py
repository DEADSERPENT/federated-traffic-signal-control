#!/usr/bin/env python3
"""
NS-3 Bridge Server (Runs on WSL/Linux)

This server runs in WSL and handles requests from the Windows FL system.
It executes NS-3 simulations and returns network metrics.

Usage (in WSL):
    python3 ns3_bridge_server.py [--port PORT] [--ns3-dir NS3_DIR]

Architecture:
    Windows (FL Training) <--ZeroMQ--> WSL (NS-3 Bridge) <--subprocess--> NS-3
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import zmq
except ImportError:
    print("Installing pyzmq...")
    os.system("pip3 install pyzmq")
    import zmq

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip3 install numpy")
    import numpy as np


class NS3BridgeServer:
    """
    Bridge server that connects Windows FL system with NS-3 on WSL.
    """

    def __init__(
        self,
        port: int = 5555,
        ns3_dir: str = None,
        verbose: bool = True
    ):
        """
        Initialize the bridge server.

        Args:
            port: ZeroMQ port for communication
            ns3_dir: NS-3 installation directory
            verbose: Enable verbose output
        """
        self.port = port
        self.verbose = verbose

        # Find NS-3 directory
        if ns3_dir:
            self.ns3_dir = Path(ns3_dir)
        else:
            # Default locations
            possible_dirs = [
                Path.home() / "ns3-fl-traffic" / "ns-3.3.40",
                Path.home() / "ns-3.40",
                Path.home() / "ns3" / "ns-3.40",
                Path("/opt/ns-3.40"),
            ]
            self.ns3_dir = None
            for d in possible_dirs:
                if d.exists():
                    self.ns3_dir = d
                    break

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Statistics
        self.stats = {
            "requests_handled": 0,
            "simulations_run": 0,
            "total_time_s": 0.0
        }

        # Cache for simulation results
        self.result_cache: Dict[str, Any] = {}

    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def start(self):
        """Start the bridge server."""
        bind_address = f"tcp://*:{self.port}"
        self.socket.bind(bind_address)

        print("=" * 60)
        print("  NS-3 Bridge Server")
        print("=" * 60)
        print(f"  Listening on: {bind_address}")
        print(f"  NS-3 Directory: {self.ns3_dir or 'Not found (simulation mode)'}")
        print("=" * 60)
        print("\nWaiting for connections from Windows FL system...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Wait for request
                message = self.socket.recv_json()
                self.log(f"Received: {message.get('type', 'unknown')}")

                # Process request
                start_time = time.time()
                response = self.handle_request(message)
                elapsed = time.time() - start_time

                self.stats["requests_handled"] += 1
                self.stats["total_time_s"] += elapsed

                # Send response
                self.socket.send_json(response)
                self.log(f"Response sent ({elapsed:.3f}s)")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.print_stats()

        finally:
            self.socket.close()
            self.context.term()

    def handle_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming request from Windows FL system.

        Args:
            message: Request message

        Returns:
            Response dictionary
        """
        request_type = message.get("type", "")

        if request_type == "ping":
            return self.handle_ping()

        elif request_type == "fl_update":
            return self.handle_fl_update(message)

        elif request_type == "run_simulation":
            return self.handle_run_simulation(message)

        elif request_type == "get_network_metrics":
            return self.handle_get_metrics(message)

        elif request_type == "batch_simulation":
            return self.handle_batch_simulation(message)

        else:
            return {"error": f"Unknown request type: {request_type}"}

    def handle_ping(self) -> Dict[str, Any]:
        """Handle ping request for connection testing."""
        return {
            "status": "ok",
            "server": "NS-3 Bridge Server",
            "ns3_available": self.ns3_dir is not None,
            "timestamp": time.time()
        }

    def handle_fl_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate FL model update transmission through network.

        This provides realistic network metrics without running full NS-3
        simulation (for faster response during training).
        """
        payload_size = message.get("payload_size", 100000)  # bytes
        num_clients = message.get("num_clients", 4)
        network_scenario = message.get("network_scenario", "normal")

        # Network parameters based on scenario
        scenarios = {
            "ideal": {"latency_base": 5, "loss_prob": 0.0, "bandwidth_mbps": 54},
            "normal": {"latency_base": 15, "loss_prob": 0.01, "bandwidth_mbps": 27},
            "degraded": {"latency_base": 50, "loss_prob": 0.05, "bandwidth_mbps": 12},
            "stressed": {"latency_base": 100, "loss_prob": 0.10, "bandwidth_mbps": 6},
            "extreme": {"latency_base": 200, "loss_prob": 0.20, "bandwidth_mbps": 3}
        }

        params = scenarios.get(network_scenario, scenarios["normal"])

        # Calculate latency
        # Components: base + transmission + queueing + jitter
        transmission_time = (payload_size * 8) / (params["bandwidth_mbps"] * 1e6) * 1000  # ms
        queueing_delay = np.random.exponential(2)  # ms
        jitter = np.random.uniform(-params["latency_base"] * 0.1,
                                    params["latency_base"] * 0.1)

        total_latency = params["latency_base"] + transmission_time + queueing_delay + jitter

        # Packet loss
        packet_lost = np.random.random() < params["loss_prob"]

        # Per-client results
        client_results = []
        for i in range(num_clients):
            client_latency = total_latency + np.random.normal(0, 5)
            client_lost = np.random.random() < params["loss_prob"]
            client_results.append({
                "client_id": i,
                "latency_ms": max(1, client_latency),
                "packet_lost": client_lost,
                "success": not client_lost
            })

        # Aggregate results
        successful = sum(1 for c in client_results if c["success"])
        avg_latency = np.mean([c["latency_ms"] for c in client_results])

        return {
            "status": "ok",
            "scenario": network_scenario,
            "total_clients": num_clients,
            "successful_clients": successful,
            "failed_clients": num_clients - successful,
            "avg_latency_ms": float(avg_latency),
            "packet_loss_rate": 1 - (successful / num_clients),
            "bandwidth_used_kbps": payload_size * 8 / avg_latency,
            "client_results": client_results,
            "simulated": True  # Indicates this was quick simulation
        }

    def handle_run_simulation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full NS-3 simulation.

        This executes the actual NS-3 binary for accurate results.
        """
        num_intersections = message.get("num_intersections", 4)
        num_vehicles = message.get("num_vehicles", 50)
        duration = message.get("duration", 60)

        if self.ns3_dir is None:
            # Fallback to simulation mode
            self.log("NS-3 not found, using simulation mode")
            return self.simulate_ns3_results(num_intersections, num_vehicles, duration)

        try:
            # Create output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                             delete=False) as f:
                output_file = f.name

            # Build NS-3 command
            cmd = [
                str(self.ns3_dir / "ns3"), "run",
                f"scratch/fl-traffic/fl_traffic_network",
                "--",
                f"--intersections={num_intersections}",
                f"--vehicles={num_vehicles}",
                f"--duration={duration}",
                f"--output={output_file}"
            ]

            self.log(f"Running NS-3: {' '.join(cmd)}")

            # Execute NS-3
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.ns3_dir),
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                self.log(f"NS-3 error: {result.stderr}")
                return self.simulate_ns3_results(num_intersections, num_vehicles, duration)

            # Read results
            with open(output_file, 'r') as f:
                ns3_results = json.load(f)

            os.unlink(output_file)

            self.stats["simulations_run"] += 1

            return {
                "status": "ok",
                "source": "ns3",
                **ns3_results
            }

        except subprocess.TimeoutExpired:
            self.log("NS-3 simulation timed out")
            return self.simulate_ns3_results(num_intersections, num_vehicles, duration)

        except Exception as e:
            self.log(f"NS-3 error: {str(e)}")
            return self.simulate_ns3_results(num_intersections, num_vehicles, duration)

    def simulate_ns3_results(
        self,
        num_intersections: int,
        num_vehicles: int,
        duration: float
    ) -> Dict[str, Any]:
        """
        Simulate NS-3 results when actual NS-3 is not available.
        Uses statistical models based on typical 802.11p performance.
        """
        # Base metrics for 802.11p V2I communication
        base_latency = 10 + num_vehicles * 0.1  # Increases with vehicle density
        packet_loss = 0.01 + num_vehicles * 0.0002  # Increases with congestion

        # Generate realistic latency distribution
        latencies = []
        for _ in range(num_intersections * 10):  # 10 updates per intersection
            latency = base_latency + np.random.exponential(5)
            if np.random.random() < 0.1:  # 10% chance of higher latency
                latency += np.random.exponential(20)
            latencies.append(latency)

        # Calculate statistics
        packets_sent = num_intersections * 10
        packets_lost = int(packets_sent * packet_loss)
        packets_received = packets_sent - packets_lost

        return {
            "status": "ok",
            "source": "simulated",
            "intersections": num_intersections,
            "vehicles": num_vehicles,
            "duration_s": duration,
            "packets_sent": packets_sent,
            "packets_received": packets_received,
            "packet_loss_rate": packet_loss,
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "throughput_mbps": (packets_received * 100 * 1024 * 8) / (duration * 1e6),
            "latencies": latencies[:100]  # First 100 for analysis
        }

    def handle_get_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get cached network metrics."""
        return {
            "status": "ok",
            "server_stats": self.stats,
            "cache_size": len(self.result_cache)
        }

    def handle_batch_simulation(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run batch of simulations with different parameters.
        Useful for network stress testing.
        """
        scenarios = message.get("scenarios", [])
        results = []

        for scenario in scenarios:
            result = self.handle_run_simulation(scenario)
            results.append({
                "scenario": scenario,
                "result": result
            })

        return {
            "status": "ok",
            "batch_size": len(scenarios),
            "results": results
        }

    def print_stats(self):
        """Print server statistics."""
        print("\n" + "=" * 40)
        print("Server Statistics:")
        print(f"  Requests handled: {self.stats['requests_handled']}")
        print(f"  Simulations run: {self.stats['simulations_run']}")
        print(f"  Total time: {self.stats['total_time_s']:.2f}s")
        print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="NS-3 Bridge Server for FL Traffic")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port")
    parser.add_argument("--ns3-dir", type=str, help="NS-3 installation directory")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    args = parser.parse_args()

    server = NS3BridgeServer(
        port=args.port,
        ns3_dir=args.ns3_dir,
        verbose=not args.quiet
    )
    server.start()


if __name__ == "__main__":
    main()
