"""
Intersection Module
Simulates a single traffic intersection with multiple lanes and signals.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class SignalPhase(Enum):
    """Traffic signal phases"""
    RED = 0
    YELLOW = 1
    GREEN = 2


@dataclass
class Lane:
    """Represents a single lane at an intersection"""
    lane_id: str
    direction: str  # "north", "south", "east", "west"
    queue_length: int = 0
    vehicles_waiting: List[float] = field(default_factory=list)  # arrival times
    total_vehicles_passed: int = 0
    total_waiting_time: float = 0.0

    def add_vehicle(self, arrival_time: float):
        """Add a vehicle to the queue"""
        self.vehicles_waiting.append(arrival_time)
        self.queue_length = len(self.vehicles_waiting)

    def remove_vehicles(self, count: int, current_time: float):
        """Remove vehicles that passed through the intersection"""
        vehicles_to_remove = min(count, self.queue_length)
        for _ in range(vehicles_to_remove):
            if self.vehicles_waiting:
                arrival_time = self.vehicles_waiting.pop(0)
                self.total_waiting_time += (current_time - arrival_time)
                self.total_vehicles_passed += 1
        self.queue_length = len(self.vehicles_waiting)

    def get_average_waiting_time(self) -> float:
        """Calculate average waiting time for vehicles that passed"""
        if self.total_vehicles_passed == 0:
            return 0.0
        return self.total_waiting_time / self.total_vehicles_passed


class Intersection:
    """
    Simulates a 4-way traffic intersection.
    Each intersection has 4 approaches (N, S, E, W) with lanes.
    """

    def __init__(
        self,
        intersection_id: int,
        arrival_rate: float = 10.0,
        max_queue_length: int = 50,
        min_green: int = 10,
        max_green: int = 90,
        yellow_duration: int = 3
    ):
        self.intersection_id = intersection_id
        self.arrival_rate = arrival_rate  # vehicles per minute
        self.max_queue_length = max_queue_length
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration

        # Initialize lanes for each direction
        self.lanes: Dict[str, Lane] = {
            "north": Lane(f"int{intersection_id}_north", "north"),
            "south": Lane(f"int{intersection_id}_south", "south"),
            "east": Lane(f"int{intersection_id}_east", "east"),
            "west": Lane(f"int{intersection_id}_west", "west")
        }

        # Signal state
        self.current_phase = "north_south"  # "north_south" or "east_west"
        self.signal_state = SignalPhase.GREEN
        self.green_duration = 30  # current green signal duration
        self.phase_timer = 0

        # Metrics
        self.total_throughput = 0
        self.simulation_time = 0

    def generate_arrivals(self, time_step: float, distribution: str = "poisson") -> Dict[str, int]:
        """
        Generate vehicle arrivals for each lane based on arrival rate.

        Args:
            time_step: Time interval in seconds
            distribution: "poisson" or "uniform"

        Returns:
            Dictionary with arrivals per lane
        """
        arrivals = {}
        rate_per_step = (self.arrival_rate / 60.0) * time_step  # Convert to per-step rate

        for direction, lane in self.lanes.items():
            if distribution == "poisson":
                num_arrivals = np.random.poisson(rate_per_step)
            else:  # uniform
                num_arrivals = int(np.random.uniform(0, rate_per_step * 2))

            # Add vehicles to queue (respecting max queue length)
            actual_arrivals = 0
            for _ in range(num_arrivals):
                if lane.queue_length < self.max_queue_length:
                    lane.add_vehicle(self.simulation_time)
                    actual_arrivals += 1

            arrivals[direction] = actual_arrivals

        return arrivals

    def process_signal(self, time_step: float) -> Dict[str, int]:
        """
        Process signal phase and allow vehicles to pass.

        Args:
            time_step: Time interval in seconds

        Returns:
            Dictionary with vehicles passed per lane
        """
        vehicles_passed = {direction: 0 for direction in self.lanes}

        # Determine which lanes have green signal
        if self.current_phase == "north_south":
            green_lanes = ["north", "south"]
        else:
            green_lanes = ["east", "west"]

        # Process vehicles for green lanes
        if self.signal_state == SignalPhase.GREEN:
            # Assume ~2 vehicles can pass per second during green
            vehicles_per_step = int(2 * time_step)

            for direction in green_lanes:
                lane = self.lanes[direction]
                passed = min(vehicles_per_step, lane.queue_length)
                lane.remove_vehicles(passed, self.simulation_time)
                vehicles_passed[direction] = passed
                self.total_throughput += passed

        return vehicles_passed

    def update_signal(self, new_green_duration: float = None):
        """
        Update signal timing based on predicted optimal duration.

        Args:
            new_green_duration: Predicted optimal green duration (optional)
        """
        if new_green_duration is not None:
            self.green_duration = np.clip(
                new_green_duration,
                self.min_green,
                self.max_green
            )

    def step(self, time_step: float, distribution: str = "poisson") -> Dict:
        """
        Execute one simulation step.

        Args:
            time_step: Time interval in seconds
            distribution: Arrival distribution type

        Returns:
            Dictionary with step metrics
        """
        self.simulation_time += time_step
        self.phase_timer += time_step

        # Generate new arrivals
        arrivals = self.generate_arrivals(time_step, distribution)

        # Process signal and vehicles
        vehicles_passed = self.process_signal(time_step)

        # Check for phase transition
        if self.signal_state == SignalPhase.GREEN:
            if self.phase_timer >= self.green_duration:
                self.signal_state = SignalPhase.YELLOW
                self.phase_timer = 0
        elif self.signal_state == SignalPhase.YELLOW:
            if self.phase_timer >= self.yellow_duration:
                self.signal_state = SignalPhase.GREEN
                self.phase_timer = 0
                # Switch phase
                self.current_phase = "east_west" if self.current_phase == "north_south" else "north_south"

        # Collect metrics
        metrics = self.get_current_metrics()
        metrics["arrivals"] = arrivals
        metrics["vehicles_passed"] = vehicles_passed

        return metrics

    def get_current_metrics(self) -> Dict:
        """Get current intersection metrics for ML model input."""
        total_queue = sum(lane.queue_length for lane in self.lanes.values())
        avg_waiting = np.mean([
            lane.get_average_waiting_time()
            for lane in self.lanes.values()
        ])

        # Per-direction metrics
        queue_lengths = {d: lane.queue_length for d, lane in self.lanes.items()}

        return {
            "intersection_id": self.intersection_id,
            "simulation_time": self.simulation_time,
            "total_queue_length": total_queue,
            "average_waiting_time": avg_waiting,
            "queue_lengths": queue_lengths,
            "current_phase": self.current_phase,
            "signal_state": self.signal_state.name,
            "green_duration": self.green_duration,
            "total_throughput": self.total_throughput
        }

    def get_feature_vector(self) -> np.ndarray:
        """
        Get feature vector for ML model input.

        Features:
        - Queue length for each direction (4 features)
        - Current phase encoded (1 feature)
        - Current green duration (1 feature)

        Returns:
            numpy array of features
        """
        features = [
            self.lanes["north"].queue_length,
            self.lanes["south"].queue_length,
            self.lanes["east"].queue_length,
            self.lanes["west"].queue_length,
            1.0 if self.current_phase == "north_south" else 0.0,
            self.green_duration / self.max_green  # Normalized
        ]
        return np.array(features, dtype=np.float32)

    def reset(self):
        """Reset intersection to initial state."""
        for lane in self.lanes.values():
            lane.queue_length = 0
            lane.vehicles_waiting = []
            lane.total_vehicles_passed = 0
            lane.total_waiting_time = 0.0

        self.current_phase = "north_south"
        self.signal_state = SignalPhase.GREEN
        self.green_duration = 30
        self.phase_timer = 0
        self.total_throughput = 0
        self.simulation_time = 0
