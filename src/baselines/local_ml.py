"""
Local ML Traffic Signal Controller
Each intersection trains its own model without federated learning.
Used as baseline to show FL benefits.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_model import create_model, train_model, evaluate_model


class LocalMLController:
    """
    Local ML-based traffic signal controller.
    Each intersection has its own model trained only on local data.
    No knowledge sharing between intersections.
    """

    def __init__(
        self,
        num_intersections: int = 4,
        hidden_layers: List[int] = None,
        local_epochs: int = 10,
        learning_rate: float = 0.01
    ):
        self.num_intersections = num_intersections
        self.hidden_layers = hidden_layers or [64, 32]
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Create separate model for each intersection
        self.models = {}
        for i in range(num_intersections):
            self.models[i] = create_model(
                "neural_network",
                hidden_layers=self.hidden_layers
            )

        self.training_history = {}
        self.is_trained = False

    def train_local_models(
        self,
        training_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        epochs: int = None
    ) -> Dict[int, List[float]]:
        """
        Train each local model independently.

        Args:
            training_data: Dict mapping intersection_id to (features, labels)
            epochs: Number of training epochs

        Returns:
            Training loss history for each intersection
        """
        epochs = epochs or self.local_epochs

        for intersection_id, (features, labels) in training_data.items():
            print(f"  Training Local Model for Intersection {intersection_id}...")

            model = self.models[intersection_id]
            model, loss_history = train_model(
                model,
                (features, labels),
                epochs=epochs,
                batch_size=32,
                learning_rate=self.learning_rate
            )

            self.models[intersection_id] = model
            self.training_history[intersection_id] = loss_history

            print(f"    Initial Loss: {loss_history[0]:.4f}")
            print(f"    Final Loss: {loss_history[-1]:.4f}")

        self.is_trained = True
        return self.training_history

    def get_green_duration(
        self,
        intersection_id: int,
        features: np.ndarray
    ) -> float:
        """
        Predict optimal green duration using local model.

        Args:
            intersection_id: ID of the intersection
            features: Current intersection state features

        Returns:
            Predicted green duration
        """
        if not self.is_trained:
            return 30.0  # Default

        model = self.models[intersection_id]
        prediction = model.predict(features)
        return float(np.clip(prediction[0], 10, 90))

    def evaluate_models(
        self,
        test_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Evaluate all local models.

        Args:
            test_data: Dict mapping intersection_id to (features, labels)

        Returns:
            Evaluation metrics
        """
        total_mse = 0
        total_mae = 0
        count = 0

        for intersection_id, (features, labels) in test_data.items():
            model = self.models[intersection_id]
            mse, mae = evaluate_model(model, (features, labels))
            total_mse += mse
            total_mae += mae
            count += 1

        return {
            "method": "Local-ML",
            "avg_mse": total_mse / count,
            "avg_mae": total_mae / count
        }

    def run_simulation(
        self,
        intersections: List,
        generator,
        duration: int = 3600,
        time_step: int = 5
    ) -> Dict:
        """
        Run simulation with local ML models.

        Args:
            intersections: List of Intersection objects
            generator: TrafficDataGenerator for training data
            duration: Simulation duration
            time_step: Time step

        Returns:
            Simulation results
        """
        # Train models first
        if not self.is_trained:
            print("\nTraining Local ML Models...")
            training_data = generator.get_all_intersections_data()
            self.train_local_models(training_data)

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

                # Predict green duration
                green_duration = self.get_green_duration(
                    intersection.intersection_id,
                    features
                )

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

        # Get evaluation metrics
        test_data = generator.get_all_intersections_data()
        eval_metrics = self.evaluate_models(test_data)

        return {
            "method": "Local-ML",
            "avg_waiting_time": total_waiting_time / (num_steps * len(intersections)),
            "avg_queue_length": total_queue_length / (num_steps * len(intersections)),
            "total_throughput": final_throughput,
            "throughput_per_hour": final_throughput * (3600 / duration),
            "mse": eval_metrics["avg_mse"],
            "mae": eval_metrics["avg_mae"],
            "step_metrics": step_metrics,
            "training_history": self.training_history
        }


if __name__ == "__main__":
    from traffic_generator import TrafficDataGenerator

    print("Testing Local ML Controller...")

    generator = TrafficDataGenerator()
    controller = LocalMLController(num_intersections=4)

    results = controller.run_simulation(
        generator.intersections,
        generator,
        duration=300
    )

    print(f"\nLocal ML Results (5 min simulation):")
    print(f"  Average Waiting Time: {results['avg_waiting_time']:.2f}s")
    print(f"  Average Queue Length: {results['avg_queue_length']:.2f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
