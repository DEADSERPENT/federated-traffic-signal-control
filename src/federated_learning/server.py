"""
Federated Learning Server
Central aggregator using Flower framework with FedAvg algorithm.
"""

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_model import create_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples

    Returns:
        Aggregated metrics dictionary
    """
    # Collect all metric values
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Weighted average for common metrics
    aggregated = {}

    # Get all metric names from the first client
    if metrics:
        metric_names = metrics[0][1].keys()

        for name in metric_names:
            weighted_sum = sum(
                num_examples * m.get(name, 0)
                for num_examples, m in metrics
            )
            aggregated[name] = weighted_sum / total_examples

    return aggregated


class FederatedServer:
    """
    Federated Learning Server for Traffic Signal Optimization.
    Uses FedAvg algorithm to aggregate model updates from edge nodes.
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:8080",
        num_rounds: int = 10,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        model_config: Dict = None
    ):
        """
        Initialize the federated server.

        Args:
            server_address: Address to bind the server
            num_rounds: Number of federated learning rounds
            min_clients: Minimum clients required to start training
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            model_config: Model configuration dictionary
        """
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.model_config = model_config or {}

        # Initialize global model
        self.model = create_model(
            model_type=self.model_config.get("type", "neural_network"),
            hidden_layers=self.model_config.get("hidden_layers", [64, 32])
        )

        # Get initial parameters
        self.initial_parameters = ndarrays_to_parameters(
            self.model.get_parameters()
        )

        # Create strategy
        self.strategy = self._create_strategy()

    def _create_strategy(self) -> FedAvg:
        """Create the FedAvg strategy for aggregation."""
        return FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            initial_parameters=self.initial_parameters,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    def start(self):
        """Start the federated learning server."""
        print(f"Starting Federated Learning Server on {self.server_address}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Minimum clients: {self.min_clients}")
        print("-" * 50)

        fl.server.start_server(
            server_address=self.server_address,
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    min_clients: int = 2,
    config_path: str = None
):
    """
    Convenience function to start the federated server.

    Args:
        server_address: Address to bind the server
        num_rounds: Number of federated learning rounds
        min_clients: Minimum clients required
        config_path: Path to configuration file
    """
    import yaml

    model_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_config = config.get("model", {})
            fl_config = config.get("federated_learning", {})
            num_rounds = fl_config.get("num_rounds", num_rounds)
            min_clients = fl_config.get("min_clients", min_clients)
            server_address = fl_config.get("server_address", server_address)

    server = FederatedServer(
        server_address=server_address,
        num_rounds=num_rounds,
        min_clients=min_clients,
        model_config=model_config
    )
    server.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        config_path=args.config
    )
