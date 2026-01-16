"""
Network Abstraction Layer
Simulates network conditions for federated learning communication.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import threading
import queue


@dataclass
class NetworkConditions:
    """Network condition parameters."""
    base_latency: float = 10.0  # milliseconds
    bandwidth: float = 100.0  # Mbps
    packet_loss_probability: float = 0.01
    jitter_range: float = 5.0  # milliseconds
    congestion_factor: float = 1.0  # 1.0 = no congestion, higher = more congestion


class NetworkSimulator:
    """
    Simulates network behavior for federated learning communication.
    Models latency, bandwidth, packet loss, and jitter.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the network simulator.

        Args:
            config: Configuration dictionary with network parameters
        """
        if config is None:
            config = {}

        network_config = config.get("network", config)

        self.conditions = NetworkConditions(
            base_latency=network_config.get("base_latency", 10.0),
            bandwidth=network_config.get("bandwidth", 100.0),
            packet_loss_probability=network_config.get("packet_loss_probability", 0.01),
            jitter_range=network_config.get("jitter_range", 5.0)
        )

        # Metrics tracking
        self.total_bytes_sent = 0
        self.total_packets_sent = 0
        self.total_packets_lost = 0
        self.latency_history: List[float] = []

        # Lock for thread safety
        self._lock = threading.Lock()

    def simulate_transmission(
        self,
        data_size_bytes: int,
        source: str = "client",
        destination: str = "server"
    ) -> Tuple[float, bool]:
        """
        Simulate data transmission with network effects.

        Args:
            data_size_bytes: Size of data to transmit in bytes
            source: Source node identifier
            destination: Destination node identifier

        Returns:
            Tuple of (transmission_time_ms, success)
        """
        with self._lock:
            # Calculate base transmission time based on bandwidth
            # bandwidth is in Mbps, convert to bytes per ms
            bytes_per_ms = (self.conditions.bandwidth * 1000000) / (8 * 1000)
            transmission_time = data_size_bytes / bytes_per_ms

            # Add base latency
            total_latency = self.conditions.base_latency

            # Add jitter (random variation)
            jitter = np.random.uniform(
                -self.conditions.jitter_range,
                self.conditions.jitter_range
            )
            total_latency += abs(jitter)

            # Apply congestion factor
            total_latency *= self.conditions.congestion_factor
            transmission_time *= self.conditions.congestion_factor

            # Total time = latency + transmission time
            total_time = total_latency + transmission_time

            # Check for packet loss
            success = np.random.random() > self.conditions.packet_loss_probability

            # Update metrics
            self.total_packets_sent += 1
            if success:
                self.total_bytes_sent += data_size_bytes
            else:
                self.total_packets_lost += 1

            self.latency_history.append(total_time)

            return total_time, success

    def simulate_model_update(
        self,
        model_parameters: List[np.ndarray],
        client_id: int
    ) -> Tuple[float, bool, List[np.ndarray]]:
        """
        Simulate sending model parameters from client to server.

        Args:
            model_parameters: List of numpy arrays (model weights)
            client_id: Client identifier

        Returns:
            Tuple of (latency_ms, success, parameters)
        """
        # Calculate total size of model parameters
        total_size = sum(param.nbytes for param in model_parameters)

        # Simulate transmission
        latency, success = self.simulate_transmission(
            total_size,
            source=f"client_{client_id}",
            destination="server"
        )

        if success:
            # Simulate actual delay
            time.sleep(latency / 1000.0)  # Convert ms to seconds
            return latency, True, model_parameters
        else:
            # Packet lost - parameters not received
            return latency, False, None

    def simulate_global_model_broadcast(
        self,
        model_parameters: List[np.ndarray],
        num_clients: int
    ) -> Dict[int, Tuple[float, bool]]:
        """
        Simulate broadcasting global model to all clients.

        Args:
            model_parameters: Global model parameters
            num_clients: Number of clients to broadcast to

        Returns:
            Dictionary mapping client_id to (latency, success) tuples
        """
        total_size = sum(param.nbytes for param in model_parameters)
        results = {}

        for client_id in range(num_clients):
            latency, success = self.simulate_transmission(
                total_size,
                source="server",
                destination=f"client_{client_id}"
            )
            results[client_id] = (latency, success)

        return results

    def set_congestion(self, factor: float):
        """
        Set network congestion factor.

        Args:
            factor: Congestion factor (1.0 = normal, higher = more congestion)
        """
        with self._lock:
            self.conditions.congestion_factor = max(1.0, factor)

    def simulate_time_varying_conditions(self, time_step: int):
        """
        Simulate time-varying network conditions.
        Models realistic patterns like peak hours.

        Args:
            time_step: Current simulation time step
        """
        # Simulate daily traffic pattern (simplified)
        # Higher congestion during "peak hours"
        hour_of_day = (time_step % 24)

        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            # Peak hours - higher congestion
            self.set_congestion(2.0)
        elif 0 <= hour_of_day <= 5:
            # Night - low congestion
            self.set_congestion(1.0)
        else:
            # Normal hours
            self.set_congestion(1.5)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get network simulation metrics.

        Returns:
            Dictionary of network metrics
        """
        with self._lock:
            if self.latency_history:
                avg_latency = np.mean(self.latency_history)
                max_latency = np.max(self.latency_history)
                min_latency = np.min(self.latency_history)
            else:
                avg_latency = max_latency = min_latency = 0.0

            packet_loss_rate = (
                self.total_packets_lost / self.total_packets_sent
                if self.total_packets_sent > 0 else 0.0
            )

            return {
                "total_bytes_sent": self.total_bytes_sent,
                "total_packets_sent": self.total_packets_sent,
                "total_packets_lost": self.total_packets_lost,
                "packet_loss_rate": packet_loss_rate,
                "average_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "min_latency_ms": min_latency,
                "current_congestion": self.conditions.congestion_factor
            }

    def reset_metrics(self):
        """Reset all metrics to zero."""
        with self._lock:
            self.total_bytes_sent = 0
            self.total_packets_sent = 0
            self.total_packets_lost = 0
            self.latency_history = []


class NetworkAwareClient:
    """
    Wrapper for FL client that applies network simulation.
    """

    def __init__(self, network_simulator: NetworkSimulator, client_id: int):
        self.network = network_simulator
        self.client_id = client_id

    def send_parameters(
        self,
        parameters: List[np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Send model parameters with simulated network effects.

        Args:
            parameters: Model parameters to send

        Returns:
            Tuple of (success, latency_ms)
        """
        latency, success, _ = self.network.simulate_model_update(
            parameters, self.client_id
        )
        return success, latency

    def receive_parameters(
        self,
        parameters: List[np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Receive model parameters with simulated network effects.

        Args:
            parameters: Model parameters being received

        Returns:
            Tuple of (success, latency_ms)
        """
        total_size = sum(param.nbytes for param in parameters)
        latency, success = self.network.simulate_transmission(
            total_size,
            source="server",
            destination=f"client_{self.client_id}"
        )
        return success, latency


def main():
    """Test the network simulator."""
    print("Testing Network Simulator...")

    # Create simulator with default config
    simulator = NetworkSimulator({
        "network": {
            "base_latency": 10,
            "bandwidth": 100,
            "packet_loss_probability": 0.05,
            "jitter_range": 5
        }
    })

    # Simulate some transmissions
    print("\nSimulating 100 transmissions...")
    for i in range(100):
        data_size = np.random.randint(1000, 100000)
        latency, success = simulator.simulate_transmission(data_size)
        if not success:
            print(f"  Packet {i} lost!")

    # Get metrics
    metrics = simulator.get_metrics()
    print("\nNetwork Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test model update simulation
    print("\nSimulating model parameter update...")
    fake_params = [np.random.rand(64, 6), np.random.rand(64), np.random.rand(32, 64)]
    latency, success, params = simulator.simulate_model_update(fake_params, client_id=0)
    print(f"  Latency: {latency:.2f} ms, Success: {success}")


if __name__ == "__main__":
    main()
