"""
Adaptive Federated Learning Traffic Signal Controller
Uses FL to train a global model shared across intersections.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_model import create_model, train_model, evaluate_model


class AdaptiveFLController:
    """
    Federated Learning-based traffic signal controller.
    Trains local models and aggregates using FedAvg.
    """

    def __init__(
        self,
        num_intersections: int = 4,
        hidden_layers: List[int] = None,
        num_rounds: int = 10,
        local_epochs: int = 5,
        learning_rate: float = 0.01
    ):
        self.num_intersections = num_intersections
        self.hidden_layers = hidden_layers or [64, 32]
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Create local models
        self.local_models = {}
        for i in range(num_intersections):
            self.local_models[i] = create_model(
                "neural_network",
                hidden_layers=self.hidden_layers
            )

        # Global model
        self.global_model = create_model(
            "neural_network",
            hidden_layers=self.hidden_layers
        )

        self.round_metrics = []
        self.is_trained = False

    def federated_averaging(self, model_params: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Perform FedAvg aggregation.

        Args:
            model_params: List of model parameters from each client

        Returns:
            Averaged parameters
        """
        avg_params = []
        for i in range(len(model_params[0])):
            layer_params = [params[i] for params in model_params]
            avg_params.append(np.mean(layer_params, axis=0))
        return avg_params

    def train_federated(
        self,
        training_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """
        Train using federated learning.

        Args:
            training_data: Dict mapping intersection_id to (features, labels)

        Returns:
            Training metrics per round
        """
        print(f"\nStarting Federated Learning ({self.num_rounds} rounds)...")

        for round_num in range(self.num_rounds):
            round_losses = []
            model_params = []

            # Distribute global model to all clients
            global_params = self.global_model.get_parameters()
            for i in range(self.num_intersections):
                self.local_models[i].set_parameters(global_params)

            # Local training at each client
            for intersection_id, (features, labels) in training_data.items():
                model = self.local_models[intersection_id]

                # Train locally
                model, loss_history = train_model(
                    model,
                    (features, labels),
                    epochs=self.local_epochs,
                    batch_size=32,
                    learning_rate=self.learning_rate
                )

                self.local_models[intersection_id] = model
                round_losses.append(loss_history[-1])
                model_params.append(model.get_parameters())

            # Aggregate using FedAvg
            avg_params = self.federated_averaging(model_params)
            self.global_model.set_parameters(avg_params)

            # Evaluate global model
            total_mse = 0
            total_mae = 0
            for intersection_id, (features, labels) in training_data.items():
                # Use last 20% as test
                test_idx = int(len(features) * 0.8)
                mse, mae = evaluate_model(
                    self.global_model,
                    (features[test_idx:], labels[test_idx:])
                )
                total_mse += mse
                total_mae += mae

            avg_mse = total_mse / len(training_data)
            avg_mae = total_mae / len(training_data)

            self.round_metrics.append({
                "round": round_num + 1,
                "avg_local_loss": np.mean(round_losses),
                "global_mse": avg_mse,
                "global_mae": avg_mae
            })

            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"  Round {round_num + 1}: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")

        self.is_trained = True
        return self.round_metrics

    def get_green_duration(self, features: np.ndarray) -> float:
        """
        Predict optimal green duration using global model.

        Args:
            features: Current intersection state features

        Returns:
            Predicted green duration
        """
        if not self.is_trained:
            return 30.0

        prediction = self.global_model.predict(features)
        return float(np.clip(prediction[0], 10, 90))

    def run_simulation(
        self,
        intersections: List,
        generator,
        duration: int = 3600,
        time_step: int = 5
    ) -> Dict:
        """
        Run simulation with FL-trained global model.

        Args:
            intersections: List of Intersection objects
            generator: TrafficDataGenerator for training data
            duration: Simulation duration
            time_step: Time step

        Returns:
            Simulation results
        """
        # Train FL first
        if not self.is_trained:
            training_data = generator.get_all_intersections_data()
            self.train_federated(training_data)

        num_steps = duration // time_step

        # Reset intersections
        for intersection in intersections:
            intersection.reset()

        # Metrics tracking
        total_waiting_time = 0
        total_queue_length = 0
        step_metrics = []

        for step in range(num_steps):
            step_waiting = 0
            step_queue = 0

            for intersection in intersections:
                # Get features
                features = intersection.get_feature_vector()

                # Predict using global model
                green_duration = self.get_green_duration(features)

                # Update signal
                intersection.update_signal(green_duration)

                # Step simulation
                metrics = intersection.step(time_step, "poisson")

                step_waiting += metrics["average_waiting_time"]
                step_queue += metrics["total_queue_length"]

            step_metrics.append({
                "step": step,
                "time": step * time_step,
                "avg_waiting_time": step_waiting / len(intersections),
                "avg_queue_length": step_queue / len(intersections)
            })

            total_waiting_time += step_waiting
            total_queue_length += step_queue

        final_throughput = sum(i.total_throughput for i in intersections)

        # Final evaluation
        test_data = generator.get_all_intersections_data()
        total_mse = 0
        total_mae = 0
        for intersection_id, (features, labels) in test_data.items():
            test_idx = int(len(features) * 0.8)
            mse, mae = evaluate_model(
                self.global_model,
                (features[test_idx:], labels[test_idx:])
            )
            total_mse += mse
            total_mae += mae

        return {
            "method": "Federated-Learning",
            "avg_waiting_time": total_waiting_time / (num_steps * len(intersections)),
            "avg_queue_length": total_queue_length / (num_steps * len(intersections)),
            "total_throughput": final_throughput,
            "throughput_per_hour": final_throughput * (3600 / duration),
            "mse": total_mse / len(test_data),
            "mae": total_mae / len(test_data),
            "step_metrics": step_metrics,
            "round_metrics": self.round_metrics,
            "num_rounds": self.num_rounds
        }


if __name__ == "__main__":
    from traffic_generator import TrafficDataGenerator

    print("Testing Adaptive FL Controller...")

    generator = TrafficDataGenerator()
    controller = AdaptiveFLController(num_intersections=4, num_rounds=10)

    results = controller.run_simulation(
        generator.intersections,
        generator,
        duration=300
    )

    print(f"\nFederated Learning Results (5 min simulation):")
    print(f"  Average Waiting Time: {results['avg_waiting_time']:.2f}s")
    print(f"  Average Queue Length: {results['avg_queue_length']:.2f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
