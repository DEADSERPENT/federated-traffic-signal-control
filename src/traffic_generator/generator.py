"""
Traffic Data Generator
Generates synthetic traffic data for multiple intersections.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import yaml

from .intersection import Intersection


class TrafficDataGenerator:
    """
    Generates real-time traffic data for federated learning training.
    Each intersection acts as an edge node with local data.
    """

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        Initialize the traffic data generator.

        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (overrides config_path)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "traffic": {
                    "num_intersections": 4,
                    "simulation_duration": 3600,
                    "time_step": 5,
                    "arrival_distribution": "poisson",
                    "min_arrival_rate": 5,
                    "max_arrival_rate": 30,
                    "max_queue_length": 50,
                    "min_green_duration": 10,
                    "max_green_duration": 90,
                    "yellow_duration": 3
                }
            }

        self.traffic_config = self.config.get("traffic", self.config)
        self.intersections: List[Intersection] = []
        self._initialize_intersections()

    def _initialize_intersections(self):
        """Initialize all intersection objects."""
        num_intersections = self.traffic_config.get("num_intersections", 4)
        min_rate = self.traffic_config.get("min_arrival_rate", 5)
        max_rate = self.traffic_config.get("max_arrival_rate", 30)

        for i in range(num_intersections):
            # Each intersection has a slightly different arrival rate
            arrival_rate = np.random.uniform(min_rate, max_rate)

            intersection = Intersection(
                intersection_id=i,
                arrival_rate=arrival_rate,
                max_queue_length=self.traffic_config.get("max_queue_length", 50),
                min_green=self.traffic_config.get("min_green_duration", 10),
                max_green=self.traffic_config.get("max_green_duration", 90),
                yellow_duration=self.traffic_config.get("yellow_duration", 3)
            )
            self.intersections.append(intersection)

    def generate_training_data(
        self,
        num_samples: int = 1000,
        intersection_id: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for a specific intersection or all intersections.

        Args:
            num_samples: Number of training samples to generate
            intersection_id: Specific intersection ID (None for all)

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        if intersection_id is not None:
            intersections = [self.intersections[intersection_id]]
        else:
            intersections = self.intersections

        all_features = []
        all_labels = []

        time_step = self.traffic_config.get("time_step", 5)
        distribution = self.traffic_config.get("arrival_distribution", "poisson")

        for intersection in intersections:
            intersection.reset()

            for _ in range(num_samples):
                # Get current state features
                features = intersection.get_feature_vector()

                # Simulate one step
                metrics = intersection.step(time_step, distribution)

                # Calculate optimal green duration based on queue lengths
                # This is a simplified heuristic - in practice, this would be
                # based on actual traffic optimization algorithms
                optimal_green = self._calculate_optimal_green(intersection)

                all_features.append(features)
                all_labels.append(optimal_green)

        return np.array(all_features), np.array(all_labels, dtype=np.float32)

    def _calculate_optimal_green(self, intersection: Intersection) -> float:
        """
        Optimal green duration calculation optimized for MINIMUM WAITING TIME.
        This trains the FL model to aggressively minimize vehicle waiting.

        Key principles:
        - Switch faster when waiting queue is building up
        - Clear active queue efficiently but don't over-stay
        - Prefer shorter cycles for better responsiveness

        Args:
            intersection: Intersection object

        Returns:
            Optimal green duration in seconds
        """
        # Get queue lengths for current green phase
        if intersection.current_phase == "north_south":
            active_queue = (intersection.lanes["north"].queue_length +
                           intersection.lanes["south"].queue_length)
            waiting_queue = (intersection.lanes["east"].queue_length +
                            intersection.lanes["west"].queue_length)
        else:
            active_queue = (intersection.lanes["east"].queue_length +
                           intersection.lanes["west"].queue_length)
            waiting_queue = (intersection.lanes["north"].queue_length +
                            intersection.lanes["south"].queue_length)

        total_queue = active_queue + waiting_queue
        min_green = intersection.min_green
        max_green = min(intersection.max_green, 70)  # Cap at 70 for responsiveness

        if total_queue == 0:
            # No traffic - use minimum
            optimal_green = min_green
        else:
            # Time to clear active queue (2 vehicles/sec throughput)
            time_to_clear = active_queue / 2.0

            # WAITING TIME MINIMIZATION STRATEGY:
            # The key insight is that waiting time accumulates faster when
            # vehicles wait longer. Short, responsive cycles reduce total wait.

            if waiting_queue > active_queue * 1.5 and waiting_queue > 5:
                # Heavy imbalance: waiting vehicles are accumulating wait time fast
                # Give just enough time to clear some of active, then switch
                optimal_green = max(time_to_clear * 0.6, min_green)
                optimal_green = min(optimal_green, 25)  # Cap to switch quickly
            elif active_queue > waiting_queue * 1.5 and active_queue > 5:
                # Active queue is heavy - clear it efficiently
                optimal_green = time_to_clear + 3
                optimal_green = min(optimal_green, 45)  # But don't over-stay
            elif total_queue < 10:
                # Light traffic - use short cycles for responsiveness
                optimal_green = min_green + 5
            else:
                # Balanced traffic - proportional allocation
                queue_ratio = active_queue / (total_queue + 1)
                # Base: proportional to active queue's share
                optimal_green = 15 + queue_ratio * 30

                # Urgency penalty: high waiting queue = cut current phase shorter
                if waiting_queue > 10:
                    urgency = min(waiting_queue / 10, 2)
                    optimal_green -= urgency * 5

        # Minimal noise for stable learning
        optimal_green += np.random.normal(0, 0.3)

        # Clip to valid range with lower cap for faster response
        optimal_green = np.clip(optimal_green, min_green, max_green)

        return float(optimal_green)

    def run_simulation(
        self,
        duration: int = None,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Run traffic simulation and collect data.

        Args:
            duration: Simulation duration in seconds
            save_path: Path to save the data CSV

        Returns:
            DataFrame with simulation data
        """
        if duration is None:
            duration = self.traffic_config.get("simulation_duration", 3600)

        time_step = self.traffic_config.get("time_step", 5)
        distribution = self.traffic_config.get("arrival_distribution", "poisson")
        num_steps = int(duration / time_step)

        # Reset all intersections
        for intersection in self.intersections:
            intersection.reset()

        # Collect data
        data_records = []

        for step in range(num_steps):
            for intersection in self.intersections:
                metrics = intersection.step(time_step, distribution)

                record = {
                    "step": step,
                    "time": step * time_step,
                    "intersection_id": intersection.intersection_id,
                    "total_queue_length": metrics["total_queue_length"],
                    "average_waiting_time": metrics["average_waiting_time"],
                    "north_queue": metrics["queue_lengths"]["north"],
                    "south_queue": metrics["queue_lengths"]["south"],
                    "east_queue": metrics["queue_lengths"]["east"],
                    "west_queue": metrics["queue_lengths"]["west"],
                    "current_phase": metrics["current_phase"],
                    "signal_state": metrics["signal_state"],
                    "green_duration": metrics["green_duration"],
                    "throughput": metrics["total_throughput"]
                }
                data_records.append(record)

        df = pd.DataFrame(data_records)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Simulation data saved to {save_path}")

        return df

    def get_intersection_data(self, intersection_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data for a specific intersection (edge node).
        Used by federated learning clients.

        Enhanced with more training samples for better FL generalization.

        Args:
            intersection_id: ID of the intersection

        Returns:
            Tuple of (features, labels) for local training
        """
        return self.generate_training_data(
            num_samples=1000,  # Increased from 500 for better FL performance
            intersection_id=intersection_id
        )

    def get_all_intersections_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get training data for all intersections.

        Returns:
            Dictionary mapping intersection_id to (features, labels) tuples
        """
        data = {}
        for i in range(len(self.intersections)):
            data[i] = self.get_intersection_data(i)
        return data


def main():
    """Test the traffic data generator."""
    print("Initializing Traffic Data Generator...")

    # Create generator with default config
    generator = TrafficDataGenerator()

    print(f"Created {len(generator.intersections)} intersections")

    # Generate some training data
    print("\nGenerating training data for intersection 0...")
    features, labels = generator.get_intersection_data(0)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample features: {features[0]}")
    print(f"Sample label: {labels[0]}")

    # Run simulation
    print("\nRunning simulation for 5 minutes...")
    df = generator.run_simulation(duration=300, save_path="data/simulation_test.csv")
    print(f"Collected {len(df)} records")
    print(df.head())


if __name__ == "__main__":
    main()
