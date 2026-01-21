"""
Actuated Traffic Signal Controller
Industry-standard sensor-based approach that extends green time based on vehicle detection.
Stronger baseline than Fixed-Time - represents real-world adaptive signals.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ActuatedConfig:
    """Configuration for actuated signal control."""
    min_green: int = 10          # Minimum green time (safety)
    max_green: int = 50          # Maximum green time (prevent starvation)
    extension_time: int = 3      # Time extension per vehicle detected
    gap_out_time: float = 3.0    # Switch if no vehicle for this duration
    yellow_duration: int = 3
    vehicle_detection_threshold: int = 2  # Min vehicles to extend


class ActuatedController:
    """
    Actuated traffic signal controller.

    Uses sensor-based detection to extend green phases when vehicles are present.
    This is the INDUSTRY STANDARD for adaptive traffic control.

    Logic:
    1. Start with minimum green time
    2. Extend green if vehicles detected in current phase
    3. Switch phase if:
       - No vehicles detected for gap_out_time, OR
       - Maximum green reached, OR
       - Waiting queue in other direction exceeds threshold
    """

    def __init__(self, config: ActuatedConfig = None):
        self.config = config or ActuatedConfig()
        self.phase_timers = {}  # Track green time per intersection
        self.gap_timers = {}    # Track time since last vehicle
        self.metrics_history = []

    def get_green_duration(
        self,
        intersection_id: int,
        features: np.ndarray,
        current_green_time: float = 0
    ) -> float:
        """
        Determine green duration using actuated logic.

        Args:
            intersection_id: ID of the intersection
            features: [north_queue, south_queue, east_queue, west_queue, phase, normalized_green]
            current_green_time: How long current phase has been green

        Returns:
            Recommended green duration
        """
        # Extract queue information
        north_queue = features[0]
        south_queue = features[1]
        east_queue = features[2]
        west_queue = features[3]
        current_phase = features[4]

        ns_queue = north_queue + south_queue
        ew_queue = east_queue + west_queue

        # Determine active/waiting queues based on phase
        if current_phase > 0.5:  # NS phase active
            active_queue = ns_queue
            waiting_queue = ew_queue
        else:  # EW phase active
            active_queue = ew_queue
            waiting_queue = ns_queue

        # === ACTUATED LOGIC ===

        # Base: start with minimum green
        green_duration = self.config.min_green

        # Extension logic: add time based on active queue
        if active_queue > self.config.vehicle_detection_threshold:
            # Vehicles present - extend green
            extension = min(
                active_queue * self.config.extension_time / 5,  # ~0.6s per vehicle
                self.config.max_green - self.config.min_green
            )
            green_duration += extension

        # Gap-out logic: if waiting queue is large and active is clearing, reduce
        if waiting_queue > active_queue * 1.5 and active_queue < 5:
            # Waiting vehicles are starving - gap out sooner
            green_duration = min(green_duration, self.config.min_green + 5)

        # Starvation prevention: cap at max if waiting queue is very large
        if waiting_queue > 15:
            max_allowed = self.config.max_green * (1 - waiting_queue / 50)
            green_duration = min(green_duration, max(max_allowed, self.config.min_green))

        # Ensure bounds
        green_duration = float(np.clip(
            green_duration,
            self.config.min_green,
            self.config.max_green
        ))

        return green_duration

    def run_simulation(
        self,
        intersections: List,
        duration: int = 3600,
        time_step: int = 5
    ) -> Dict:
        """
        Run actuated signal simulation.

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
            self.phase_timers[intersection.intersection_id] = 0

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

                # Get actuated green duration
                current_green = self.phase_timers.get(intersection.intersection_id, 0)
                green_duration = self.get_green_duration(
                    intersection.intersection_id,
                    features,
                    current_green
                )

                # Update signal
                intersection.update_signal(green_duration)

                # Step simulation
                metrics = intersection.step(time_step, "poisson")

                # Update phase timer
                self.phase_timers[intersection.intersection_id] += time_step

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

        return {
            "method": "Actuated",
            "avg_waiting_time": total_waiting_time / (num_steps * len(intersections)),
            "avg_queue_length": total_queue_length / (num_steps * len(intersections)),
            "total_throughput": final_throughput,
            "throughput_per_hour": final_throughput * (3600 / duration),
            "step_metrics": step_metrics,
            "config": {
                "min_green": self.config.min_green,
                "max_green": self.config.max_green,
                "extension_time": self.config.extension_time,
                "gap_out_time": self.config.gap_out_time
            }
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from traffic_generator import TrafficDataGenerator

    print("Testing Actuated Controller...")

    generator = TrafficDataGenerator()
    controller = ActuatedController()

    results = controller.run_simulation(generator.intersections, duration=300)

    print(f"\nActuated Results (5 min simulation):")
    print(f"  Average Waiting Time: {results['avg_waiting_time']:.2f}s")
    print(f"  Average Queue Length: {results['avg_queue_length']:.2f}")
    print(f"  Total Throughput: {results['total_throughput']} vehicles")
