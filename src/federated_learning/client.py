"""
Federated Learning Client
Edge node client representing a traffic intersection.
"""

import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np
import torch
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_model import create_model, train_model, evaluate_model
from traffic_generator.generator import TrafficDataGenerator


class TrafficClient(fl.client.NumPyClient):
    """
    Federated Learning Client for a traffic intersection.
    Acts as an edge node that trains locally and shares model parameters.
    """

    def __init__(
        self,
        intersection_id: int,
        config: Dict = None
    ):
        """
        Initialize the traffic client.

        Args:
            intersection_id: ID of the traffic intersection
            config: Configuration dictionary
        """
        self.intersection_id = intersection_id
        self.config = config or {}

        # Model configuration
        model_config = self.config.get("model", {})
        self.model = create_model(
            model_type=model_config.get("type", "neural_network"),
            hidden_layers=model_config.get("hidden_layers", [64, 32])
        )

        # Training configuration
        fl_config = self.config.get("federated_learning", {})
        self.local_epochs = fl_config.get("local_epochs", 5)
        self.batch_size = fl_config.get("batch_size", 32)
        self.learning_rate = fl_config.get("learning_rate", 0.01)

        # Initialize traffic data generator
        self.data_generator = TrafficDataGenerator(config=self.config)

        # Generate local training and test data
        self._prepare_data()

        print(f"Client {intersection_id} initialized with {len(self.train_features)} training samples")

    def _prepare_data(self):
        """Prepare local training and test data."""
        # Generate data for this intersection
        features, labels = self.data_generator.get_intersection_data(
            self.intersection_id
        )

        # Split into train/test (80/20)
        split_idx = int(len(features) * 0.8)
        self.train_features = features[:split_idx]
        self.train_labels = labels[:split_idx]
        self.test_features = features[split_idx:]
        self.test_labels = labels[split_idx:]

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current model parameters."""
        return self.model.get_parameters()

    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from server."""
        self.model.set_parameters(parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.

        Args:
            parameters: Model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated parameters, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)

        # Get training configuration
        epochs = int(config.get("local_epochs", self.local_epochs))
        batch_size = int(config.get("batch_size", self.batch_size))
        learning_rate = float(config.get("learning_rate", self.learning_rate))

        # Train locally
        self.model, loss_history = train_model(
            self.model,
            (self.train_features, self.train_labels),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Return updated parameters
        metrics = {
            "train_loss": float(loss_history[-1]),
            "intersection_id": self.intersection_id
        }

        return self.model.get_parameters(), len(self.train_features), metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local test data.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)

        # Evaluate on local test data
        mse, mae = evaluate_model(
            self.model,
            (self.test_features, self.test_labels)
        )

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "intersection_id": self.intersection_id
        }

        return float(mse), len(self.test_features), metrics


def start_client(
    server_address: str = "localhost:8080",
    intersection_id: int = 0,
    config_path: str = None
):
    """
    Start a federated learning client.

    Args:
        server_address: Address of the FL server
        intersection_id: ID of the traffic intersection
        config_path: Path to configuration file
    """
    import yaml

    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Update server address from config if available
    fl_config = config.get("federated_learning", {})
    server_address = fl_config.get("server_address", server_address)

    print(f"Starting client for intersection {intersection_id}")
    print(f"Connecting to server at {server_address}")
    print("-" * 50)

    # Create and start client
    client = TrafficClient(intersection_id=intersection_id, config=config)

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="Server address"
    )
    parser.add_argument(
        "--intersection",
        type=int,
        default=0,
        help="Intersection ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    start_client(
        server_address=args.server,
        intersection_id=args.intersection,
        config_path=args.config
    )
