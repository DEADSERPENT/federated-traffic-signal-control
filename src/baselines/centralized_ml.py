"""
Centralized-ML Traffic Signal Controller
==========================================
Trains a single model on the pooled data from ALL intersections.

This represents the theoretical upper-bound for prediction accuracy:
every intersection's data is collected at a central server and used to
train one large model.

Purpose in paper:
- Shows FL achieves *near-centralized* accuracy (privacy-accuracy gap)
- Demonstrates FL's advantage over Local-ML while paying only a small
  accuracy penalty vs. full data pooling
- Quantifies the "privacy tax": Δ MAE (Centralized → FL) << Δ MAE (FL → Local)

Privacy context:
- Centralized training requires transmitting raw sensor records
  (queue lengths, timings) from every RSU to the cloud server.
- Under GDPR/CCPA, this data can be used to infer mobility patterns
  and daily routines of individual vehicles.
- FL avoids this by sharing only model weights (no raw data).

GPU-Agnostic: uses the same DeviceManager as other controllers.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_model import create_model, train_model, evaluate_model
from utils.device import get_device, is_gpu_available


class CentralizedMLController:
    """
    Centralized ML baseline: trains one model on all intersection data.

    Represents the gold-standard accuracy achievable when privacy is *not*
    a concern and all raw data is pooled at a central server.

    Key difference from FL:
    - Raw (features, labels) from all intersections are concatenated
    - Single model trained for more epochs on the full dataset
    - No communication constraint (all data already at server)
    """

    def __init__(
        self,
        num_intersections: int = 9,
        hidden_layers: List[int] = None,
        epochs: int = 25,
        learning_rate: float = 0.002,
        weight_decay: float = 5e-5,
        device: Optional[torch.device] = None,
    ):
        self.num_intersections = num_intersections
        self.hidden_layers = hidden_layers or [256, 128, 64, 32]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device if device is not None else get_device()

        self.model = create_model(
            "neural_network",
            hidden_layers=self.hidden_layers,
            use_batch_norm=True,
            dropout_rate=0.05,
            device=self.device,
        )

        self.is_trained = False
        self.mae = None
        self.training_history: List[float] = []

    # ─── training ─────────────────────────────────────────────────────────────
    def train(
        self,
        training_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> List[float]:
        """
        Pool all intersection data and train the centralized model.

        Data from all K intersections is concatenated:
            X_pool = [X_0; X_1; … ; X_{K-1}]    (N_total × 6)
            y_pool = [y_0; y_1; … ; y_{K-1}]    (N_total,)

        Args:
            training_data: Dict mapping intersection_id → (X, y).

        Returns:
            Loss history across epochs.
        """
        print(f"\nTraining Centralized-ML (pooled data from "
              f"{len(training_data)} intersections)...")
        print(f"  Device: {self.device} ({'GPU' if is_gpu_available() else 'CPU'})")
        print(f"  Architecture: {self.hidden_layers}")

        # Pool data
        X_all = np.concatenate([v[0] for v in training_data.values()], axis=0)
        y_all = np.concatenate([v[1] for v in training_data.values()], axis=0)
        print(f"  Total pooled samples: {len(X_all):,}  "
              f"(= {len(training_data)} × {len(X_all)//len(training_data)})")

        self.model, self.training_history = train_model(
            self.model,
            (X_all, y_all),
            epochs=self.epochs,
            batch_size=64,         # larger batch for pooled data
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            use_scheduler=True,
            gradient_clip=1.0,
        )

        print(f"  Initial loss: {self.training_history[0]:.4f}")
        print(f"  Final loss:   {self.training_history[-1]:.4f}")
        self.is_trained = True
        return self.training_history

    # ─── prediction ──────────────────────────────────────────────────────────
    def get_green_duration(self, features: np.ndarray) -> float:
        """
        Predict optimal green duration using the centralized model.

        Args:
            features: [N_queue, S_queue, E_queue, W_queue, phase, green_norm]

        Returns:
            Green duration in seconds, clipped to [10, 40].
        """
        if not self.is_trained:
            return 20.0

        prediction = self.model.predict(features)
        ml_duration = float(prediction[0])

        # Apply the same queue-clearing heuristic as AdaptiveFLController
        north_q, south_q, east_q, west_q = features[0], features[1], features[2], features[3]
        phase = features[4]
        ns_q = north_q + south_q
        ew_q = east_q + west_q
        total_q = ns_q + ew_q + 0.1

        active_q  = ns_q if phase > 0.5 else ew_q
        waiting_q = ew_q if phase > 0.5 else ns_q

        if total_q < 3:
            optimal = 10.0
        elif active_q < 1:
            optimal = 10.0
        else:
            queue_ratio = active_q / (active_q + waiting_q + 0.1)
            base_cycle  = 30 if total_q < 15 else (40 if total_q < 30 else 50)
            effective   = 0.30 + 0.40 * queue_ratio
            optimal     = base_cycle * effective
            if waiting_q > active_q * 1.5:
                optimal = min(optimal, active_q / 3.0 + 5)

        final = 0.45 * ml_duration + 0.55 * optimal
        return float(np.clip(final, 10, 40))

    # ─── evaluation ──────────────────────────────────────────────────────────
    def evaluate(
        self,
        test_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """
        Evaluate the centralized model per intersection and averaged.

        Returns:
            avg_mse, avg_mae, per_intersection metrics.
        """
        total_mse = total_mae = 0.0
        per_int = {}
        for iid, (X, y) in test_data.items():
            idx = int(len(X) * 0.8)
            mse, mae = evaluate_model(self.model, (X[idx:], y[idx:]))
            total_mse += mse
            total_mae += mae
            per_int[iid] = {"mse": mse, "mae": mae}

        n = len(test_data)
        self.mae = total_mae / n
        return {
            "method": "Centralized-ML",
            "avg_mse": total_mse / n,
            "avg_mae": self.mae,
            "per_intersection": per_int,
        }

    # ─── full simulation run ─────────────────────────────────────────────────
    def run_simulation(
        self,
        intersections: List,
        generator,
        duration: int = 1800,
        time_step: int = 5,
    ) -> Dict:
        """
        Train on pooled data then run the traffic simulation.

        Args:
            intersections: List of Intersection objects.
            generator:     TrafficDataGenerator instance.
            duration:      Simulation duration in seconds.
            time_step:     Step size in seconds.

        Returns:
            Result dict compatible with other controllers.
        """
        if not self.is_trained:
            training_data = generator.get_all_intersections_data()
            self.train(training_data)

        num_steps = duration // time_step

        for inter in intersections:
            inter.reset()

        total_waiting   = 0.0
        total_queue     = 0.0
        step_metrics    = []

        for step in range(num_steps):
            sw = sq = 0.0
            for inter in intersections:
                feat = inter.get_feature_vector()
                green = self.get_green_duration(feat)
                inter.update_signal(green)
                m = inter.step(time_step, "poisson")
                sw += m["average_waiting_time"]
                sq += m["total_queue_length"]

            step_metrics.append({
                "step": step,
                "time": step * time_step,
                "avg_waiting_time": sw / len(intersections),
                "avg_queue_length": sq / len(intersections),
            })
            total_waiting += sw
            total_queue   += sq

        throughput = sum(i.total_throughput for i in intersections)

        # Evaluation MAE
        test_data = generator.get_all_intersections_data()
        eval_result = self.evaluate(test_data)

        return {
            "method": "Centralized-ML",
            "avg_waiting_time": total_waiting / (num_steps * len(intersections)),
            "avg_queue_length": total_queue   / (num_steps * len(intersections)),
            "total_throughput": throughput,
            "throughput_per_hour": throughput * (3600 / duration),
            "mse": eval_result["avg_mse"],
            "mae": eval_result["avg_mae"],
            "step_metrics": step_metrics,
            "training_history": self.training_history,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from traffic_generator import TrafficDataGenerator

    print("Testing Centralized-ML Controller...")
    generator = TrafficDataGenerator()
    controller = CentralizedMLController(num_intersections=4, epochs=10)

    results = controller.run_simulation(
        generator.intersections, generator, duration=300
    )

    print(f"\nCentralized-ML Results (5-min simulation):")
    print(f"  Average Waiting Time : {results['avg_waiting_time']:.2f}s")
    print(f"  Average Queue Length : {results['avg_queue_length']:.2f}")
    print(f"  MAE                  : {results['mae']:.4f}")
