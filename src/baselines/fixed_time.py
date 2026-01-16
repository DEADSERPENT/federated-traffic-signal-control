"""
Fixed-Time Traffic Signal Controller
Traditional approach with pre-defined signal timings.
Used as baseline for comparison with FL-based approach.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FixedTimeConfig:
    """Configuration for fixed-time signal control."""
    green_duration_ns: int = 30  # North-South green duration
    green_duration_ew: int = 30  # East-West green duration
    yellow_duration: int = 3
    cycle_length: int = 66  # Total cycle = 2*(green + yellow)


class FixedTimeController:
    """
    Fixed-time traffic signal controller.
    Uses predetermined signal timings regardless of traffic conditions.
    """

    def __init__(self, config: FixedTimeConfig = None):
        self.config = config or FixedTimeConfig()
        self.current_time = 0
        self.metrics_history = []

    def get_green_duration(self, intersection_state: Dict) -> float:
        """
        Return fixed green duration (ignores traffic state).

        Args:
            intersection_state: Current intersection state (ignored)

        Returns:
            Fixed green duration based on current phase
        """
        phase = intersection_state.get("current_phase", "north_south")
        if phase == "north_south":
            return float(self.config.green_duration_ns)
        else:
            return float(self.config.green_duration_ew)

    def run_simulation(
        self,
        intersections: List,
        duration: int = 3600,
        time_step: int = 5
    ) -> Dict:
        """
        Run fixed-time simulation.

        Args:
            intersections: List of Intersection objects
            duration: Simulation duration in seconds
            time_step: Time step in seconds

        Returns:
            Simulation results dictionary
        """
        num_steps = duration // time_step

        # Reset intersections
        for intersection in intersections:
            intersection.reset()

        # Metrics tracking
        total_waiting_time = 0
        total_queue_length = 0
        total_throughput = 0
        step_metrics = []

        for step in range(num_steps):
            step_waiting = 0
            step_queue = 0
            step_throughput = 0

            for intersection in intersections:
                # Get fixed green duration
                state = intersection.get_current_metrics()
                green_duration = self.get_green_duration(state)

                # Update signal with fixed duration
                intersection.update_signal(green_duration)

                # Step simulation
                metrics = intersection.step(time_step, "poisson")

                # Accumulate metrics
                step_waiting += metrics["average_waiting_time"]
                step_queue += metrics["total_queue_length"]
                step_throughput += sum(metrics["vehicles_passed"].values())

            step_metrics.append({
                "step": step,
                "time": step * time_step,
                "avg_waiting_time": step_waiting / len(intersections),
                "avg_queue_length": step_queue / len(intersections),
                "throughput": step_throughput
            })

            total_waiting_time += step_waiting
            total_queue_length += step_queue
            total_throughput += step_throughput

        # Final metrics
        final_throughput = sum(i.total_throughput for i in intersections)

        return {
            "method": "Fixed-Time",
            "avg_waiting_time": total_waiting_time / (num_steps * len(intersections)),
            "avg_queue_length": total_queue_length / (num_steps * len(intersections)),
            "total_throughput": final_throughput,
            "throughput_per_hour": final_throughput * (3600 / duration),
            "step_metrics": step_metrics,
            "config": {
                "green_ns": self.config.green_duration_ns,
                "green_ew": self.config.green_duration_ew,
                "cycle_length": self.config.cycle_length
            }
        }


def evaluate_fixed_time(
    intersections: List,
    duration: int = 3600,
    configs: List[FixedTimeConfig] = None
) -> List[Dict]:
    """
    Evaluate multiple fixed-time configurations.

    Args:
        intersections: List of Intersection objects
        duration: Simulation duration
        configs: List of configurations to test

    Returns:
        List of results for each configuration
    """
    if configs is None:
        # Test different cycle lengths
        configs = [
            FixedTimeConfig(green_duration_ns=20, green_duration_ew=20),
            FixedTimeConfig(green_duration_ns=30, green_duration_ew=30),
            FixedTimeConfig(green_duration_ns=45, green_duration_ew=45),
            FixedTimeConfig(green_duration_ns=60, green_duration_ew=60),
        ]

    results = []
    for config in configs:
        controller = FixedTimeController(config)
        result = controller.run_simulation(intersections, duration)
        results.append(result)

    return results


if __name__ == "__main__":
    # Test fixed-time controller
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from traffic_generator import TrafficDataGenerator

    print("Testing Fixed-Time Controller...")

    generator = TrafficDataGenerator()
    controller = FixedTimeController()

    results = controller.run_simulation(generator.intersections, duration=300)

    print(f"\nFixed-Time Results (5 min simulation):")
    print(f"  Average Waiting Time: {results['avg_waiting_time']:.2f}s")
    print(f"  Average Queue Length: {results['avg_queue_length']:.2f}")
    print(f"  Total Throughput: {results['total_throughput']} vehicles")
