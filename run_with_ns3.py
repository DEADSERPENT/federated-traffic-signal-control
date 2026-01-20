#!/usr/bin/env python3
"""
Integrated FL Training with NS-3 Network Simulation
====================================================

This script runs the complete FL traffic signal control system with
realistic network simulation using NS-3 (via WSL bridge).

Features:
- Realistic V2I (Vehicle-to-Infrastructure) communication simulation
- Accurate latency and packet loss modeling
- Integration with NS-3 802.11p/DSRC simulation
- Graceful fallback when NS-3 is unavailable

Usage:
    python run_with_ns3.py                    # Run with NS-3 integration
    python run_with_ns3.py --no-ns3          # Run without NS-3 (built-in sim)
    python run_with_ns3.py --stress-test     # Run network stress test only

Prerequisites:
    1. Start NS-3 bridge server in WSL:
       wsl python3 ns3_simulation/ns3_bridge_server.py

    2. Or run without NS-3 using --no-ns3 flag

Author: FL Traffic Signal Control Project
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from utils.reproducibility import set_global_seed, ExperimentLogger
from utils.metrics import compare_methods, calculate_convergence_metrics
from traffic_generator import TrafficDataGenerator
from baselines.fixed_time import FixedTimeController
from baselines.local_ml import LocalMLController
from baselines.adaptive_fl import AdaptiveFLController
from models.traffic_model import create_model, train_model, evaluate_model

# Import NS-3 client
try:
    from ns3_simulation.ns3_bridge_client import NS3Client, NetworkMetrics
    NS3_AVAILABLE = True
except ImportError:
    NS3_AVAILABLE = False
    print("[Warning] NS-3 bridge client not available, using built-in simulation")


class NS3IntegratedFLController(AdaptiveFLController):
    """
    FL Controller with NS-3 network simulation integration.

    Extends AdaptiveFLController to include realistic network effects
    from NS-3 simulation.
    """

    def __init__(
        self,
        ns3_client: 'NS3Client' = None,
        network_scenario: str = "normal",
        **kwargs
    ):
        """
        Initialize NS-3 integrated FL controller.

        Args:
            ns3_client: NS-3 bridge client (None for built-in simulation)
            network_scenario: Network condition scenario
            **kwargs: Arguments for AdaptiveFLController
        """
        super().__init__(**kwargs)
        self.ns3_client = ns3_client
        self.network_scenario = network_scenario
        self.network_metrics_history = []

    def train_federated_with_ns3(
        self,
        training_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """
        Train using federated learning with NS-3 network simulation.

        This method extends the base train_federated to include realistic
        network effects (latency, packet loss) from NS-3.
        """
        print(f"\n{'='*60}")
        print("  FL Training with NS-3 Network Simulation")
        print(f"  Network Scenario: {self.network_scenario}")
        print(f"{'='*60}")

        self.current_lr = self.learning_rate
        patience_counter = 0
        patience = 15

        for round_num in range(self.num_rounds):
            round_losses = []
            model_params = []
            data_sizes = []

            # === Phase 1: Distribute global model (with network simulation) ===
            global_params = self.global_model.get_parameters()

            # Simulate model broadcast through network
            if self.ns3_client:
                broadcast_metrics = self.ns3_client.simulate_fl_round(
                    global_params,
                    num_clients=self.num_intersections,
                    network_scenario=self.network_scenario
                )
                broadcast_latency = broadcast_metrics.avg_latency_ms
            else:
                broadcast_latency = 10.0  # Default

            # Set parameters for clients that received the broadcast
            for i in range(self.num_intersections):
                self.local_models[i].set_parameters(global_params)

            # === Phase 2: Local training at each client ===
            for intersection_id, (features, labels) in training_data.items():
                model = self.local_models[intersection_id]
                data_sizes.append(len(features))

                model, loss_history = train_model(
                    model,
                    (features, labels),
                    epochs=self.local_epochs,
                    batch_size=32,
                    learning_rate=self.current_lr,
                    weight_decay=self.weight_decay,
                    use_scheduler=True,
                    gradient_clip=1.0
                )

                self.local_models[intersection_id] = model
                round_losses.append(loss_history[-1])
                model_params.append(model.get_parameters())

            # === Phase 3: Upload model updates (with network simulation) ===
            if self.ns3_client:
                upload_metrics = self.ns3_client.simulate_fl_round(
                    model_params[0],  # All models same size
                    num_clients=self.num_intersections,
                    network_scenario=self.network_scenario
                )
                upload_latency = upload_metrics.avg_latency_ms
                successful_clients = upload_metrics.successful_clients

                # Only aggregate from successful uploads
                if successful_clients < len(model_params):
                    # Simulate some clients failing
                    successful_indices = np.random.choice(
                        len(model_params),
                        size=successful_clients,
                        replace=False
                    )
                    model_params = [model_params[i] for i in successful_indices]
                    round_losses = [round_losses[i] for i in successful_indices]
                    data_sizes = [data_sizes[i] for i in successful_indices]
            else:
                upload_latency = 10.0
                successful_clients = len(model_params)

            # === Phase 4: Federated Averaging ===
            if model_params:
                inv_losses = [1.0 / (loss + 1e-6) for loss in round_losses]
                combined_weights = [size * inv_loss for size, inv_loss in zip(data_sizes, inv_losses)]
                avg_params = self.federated_averaging(model_params, combined_weights)
                self.global_model.set_parameters(avg_params)

            # === Phase 5: Evaluate ===
            total_mse = 0
            total_mae = 0
            for intersection_id, (features, labels) in training_data.items():
                test_idx = int(len(features) * 0.8)
                mse, mae = evaluate_model(
                    self.global_model,
                    (features[test_idx:], labels[test_idx:])
                )
                total_mse += mse
                total_mae += mae

            avg_mse = total_mse / len(training_data)
            avg_mae = total_mae / len(training_data)

            # Track best model
            if avg_mae < self.best_mae:
                self.best_mae = avg_mae
                self.best_model_params = self.global_model.get_parameters()
                patience_counter = 0
            else:
                patience_counter += 1

            # Record metrics
            round_metrics = {
                "round": round_num + 1,
                "avg_local_loss": float(np.mean(round_losses)),
                "global_mse": avg_mse,
                "global_mae": avg_mae,
                "learning_rate": self.current_lr,
                "broadcast_latency_ms": broadcast_latency,
                "upload_latency_ms": upload_latency,
                "successful_clients": successful_clients,
                "total_round_latency_ms": broadcast_latency + upload_latency
            }
            self.round_metrics.append(round_metrics)

            # Store network metrics
            self.network_metrics_history.append({
                "round": round_num + 1,
                "broadcast_latency_ms": broadcast_latency,
                "upload_latency_ms": upload_latency,
                "total_latency_ms": broadcast_latency + upload_latency,
                "successful_clients": successful_clients
            })

            if (round_num + 1) % 10 == 0 or round_num == 0:
                print(f"  Round {round_num + 1}: MAE={avg_mae:.4f}, "
                      f"Latency={broadcast_latency + upload_latency:.1f}ms, "
                      f"Clients={successful_clients}/{self.num_intersections}")

            # Learning rate decay
            self.current_lr = max(self.current_lr * self.lr_decay, self.min_lr)

            # Early stopping
            if patience_counter >= patience and round_num >= 30:
                print(f"  Early stopping at round {round_num + 1}")
                break

        # Restore best model
        if self.best_model_params is not None:
            self.global_model.set_parameters(self.best_model_params)
            print(f"  Restored best model with MAE: {self.best_mae:.4f}")

        self.is_trained = True
        return self.round_metrics


def run_with_ns3_integration(
    use_ns3: bool = True,
    network_scenario: str = "normal",
    num_rounds: int = 50,
    output_dir: str = "results/ns3_integrated"
):
    """
    Run FL training with NS-3 network simulation.

    Args:
        use_ns3: Whether to use NS-3 bridge (False for built-in simulation)
        network_scenario: Network condition scenario
        num_rounds: Number of FL training rounds
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FEDERATED LEARNING WITH NS-3 NETWORK SIMULATION")
    print("=" * 70)

    # Initialize NS-3 client
    ns3_client = None
    if use_ns3 and NS3_AVAILABLE:
        print("\nConnecting to NS-3 bridge server...")
        try:
            ns3_client = NS3Client(auto_fallback=True)
            if ns3_client.connected:
                print("  Connected to NS-3 bridge server")
            else:
                print("  Using local fallback simulation")
        except Exception as e:
            print(f"  NS-3 connection failed: {e}")
            print("  Using local fallback simulation")
    else:
        print("\nUsing built-in network simulation (NS-3 disabled)")

    # Set seed
    set_global_seed(42)

    # Initialize traffic generator
    print("\n[1/4] Initializing traffic simulation...")
    generator = TrafficDataGenerator()
    training_data = generator.get_all_intersections_data()
    print(f"  Generated {sum(len(d[0]) for d in training_data.values())} training samples")

    # Run baselines
    print("\n[2/4] Running baseline experiments...")

    # Fixed-Time
    set_global_seed(42)
    fixed_controller = FixedTimeController()
    fixed_results = fixed_controller.run_simulation(
        generator.intersections, duration=1800
    )
    print(f"  Fixed-Time: Wait={fixed_results['avg_waiting_time']:.2f}s")

    # Local-ML
    set_global_seed(42)
    local_controller = LocalMLController(num_intersections=4)
    local_results = local_controller.run_simulation(
        generator.intersections, generator, duration=1800
    )
    print(f"  Local-ML: Wait={local_results['avg_waiting_time']:.2f}s, MAE={local_results['mae']:.4f}")

    # Run FL with NS-3
    print(f"\n[3/4] Running FL with network scenario: {network_scenario}...")
    set_global_seed(42)

    fl_controller = NS3IntegratedFLController(
        ns3_client=ns3_client,
        network_scenario=network_scenario,
        num_intersections=4,
        num_rounds=num_rounds,
        local_epochs=15,  # More local training for better model
        hidden_layers=[256, 128, 64, 32],  # Deeper network
        learning_rate=0.002,  # Higher initial LR
        lr_decay=0.99,
        weight_decay=5e-5  # Less regularization
    )

    fl_results = fl_controller.run_simulation_with_ns3(
        generator.intersections, generator, training_data, duration=1800
    )

    # Generate report
    print("\n[4/4] Generating report...")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "ns3_enabled": use_ns3,
            "network_scenario": network_scenario,
            "num_rounds": num_rounds
        },
        "fixed_time": {
            "avg_waiting_time": fixed_results["avg_waiting_time"],
            "avg_queue_length": fixed_results["avg_queue_length"]
        },
        "local_ml": {
            "avg_waiting_time": local_results["avg_waiting_time"],
            "mae": local_results["mae"]
        },
        "federated_learning": {
            "avg_waiting_time": fl_results["avg_waiting_time"],
            "mae": fl_results["mae"],
            "network_metrics": fl_controller.network_metrics_history
        }
    }

    # Save results
    with open(output_path / "ns3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Wait Time':<15} {'MAE':<15}")
    print("-" * 50)
    print(f"{'Fixed-Time':<20} {fixed_results['avg_waiting_time']:.2f}s{'':<10} N/A")
    print(f"{'Local-ML':<20} {local_results['avg_waiting_time']:.2f}s{'':<10} {local_results['mae']:.4f}")
    print(f"{'FL (NS-3 sim)':<20} {fl_results['avg_waiting_time']:.2f}s{'':<10} {fl_results['mae']:.4f}")

    if fl_controller.network_metrics_history:
        avg_latency = np.mean([m["total_latency_ms"] for m in fl_controller.network_metrics_history])
        print(f"\nNetwork Statistics ({network_scenario}):")
        print(f"  Average round-trip latency: {avg_latency:.1f}ms")

    print("\n" + "=" * 70)
    print(f"  Results saved to: {output_path}")
    print("=" * 70)

    if ns3_client:
        ns3_client.close()

    return results


def run_network_stress_test(output_dir: str = "results/ns3_stress"):
    """
    Run comprehensive network stress test using NS-3.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  NS-3 NETWORK STRESS TEST")
    print("=" * 70)

    scenarios = ["ideal", "normal", "degraded", "stressed", "extreme"]
    results = {}

    for scenario in scenarios:
        print(f"\n--- Testing scenario: {scenario} ---")
        scenario_results = run_with_ns3_integration(
            use_ns3=True,
            network_scenario=scenario,
            num_rounds=30,
            output_dir=str(output_path / scenario)
        )
        results[scenario] = scenario_results

    # Generate comparison report
    print("\n" + "=" * 70)
    print("  NETWORK STRESS TEST COMPARISON")
    print("=" * 70)
    print(f"\n{'Scenario':<12} {'FL MAE':<12} {'FL Wait':<12} {'Avg Latency':<15}")
    print("-" * 55)

    for scenario, result in results.items():
        fl = result["federated_learning"]
        avg_latency = np.mean([m["total_latency_ms"] for m in fl.get("network_metrics", [{"total_latency_ms": 0}])])
        print(f"{scenario:<12} {fl['mae']:.4f}{'':<6} {fl['avg_waiting_time']:.2f}s{'':<6} {avg_latency:.1f}ms")

    # Save combined results
    with open(output_path / "stress_test_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FL Training with NS-3 Network Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_with_ns3.py                    # Run with NS-3 (normal network)
    python run_with_ns3.py --scenario degraded  # Run with degraded network
    python run_with_ns3.py --stress-test      # Run all network scenarios
    python run_with_ns3.py --no-ns3           # Run without NS-3

Prerequisites:
    Start NS-3 bridge server in WSL first:
    wsl python3 ns3_simulation/ns3_bridge_server.py
        """
    )

    parser.add_argument("--no-ns3", action="store_true",
                        help="Disable NS-3 integration")
    parser.add_argument("--scenario", type=str, default="normal",
                        choices=["ideal", "normal", "degraded", "stressed", "extreme"],
                        help="Network scenario")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of FL rounds")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run network stress test")

    args = parser.parse_args()

    if args.stress_test:
        run_network_stress_test()
    else:
        run_with_ns3_integration(
            use_ns3=not args.no_ns3,
            network_scenario=args.scenario,
            num_rounds=args.rounds
        )


# Add method to NS3IntegratedFLController
def run_simulation_with_ns3(self, intersections, generator, training_data, duration):
    """Run FL simulation with NS-3 network effects - OPTIMIZED VERSION."""
    # Train with NS-3
    self.train_federated_with_ns3(training_data)

    # Run traffic simulation with trained model
    time_step = 5
    num_steps = int(duration / time_step)

    total_waiting_time = 0
    total_queue_length = 0
    total_throughput = 0
    step_count = 0

    for intersection in intersections:
        intersection.reset()

    for step in range(num_steps):
        for intersection in intersections:
            # Get features and predict
            features = intersection.get_feature_vector()

            if self.is_trained:
                green_duration = self.get_green_duration(features)
            else:
                green_duration = 30.0

            # CRITICAL: Apply the predicted green duration to the signal!
            intersection.update_signal(green_duration)

            # Step simulation
            metrics = intersection.step(time_step, "poisson")

            total_waiting_time += metrics["average_waiting_time"]
            total_queue_length += metrics["total_queue_length"]
            total_throughput += metrics["total_throughput"]
            step_count += 1

    # Evaluate final model
    total_mse = 0
    total_mae = 0
    for intersection_id, (features, labels) in training_data.items():
        test_idx = int(len(features) * 0.8)
        mse, mae = evaluate_model(
            self.global_model,
            (features[test_idx:], labels[test_idx:])
        )
        total_mse += mse
        total_mae += mae

    return {
        "avg_waiting_time": total_waiting_time / max(step_count, 1),
        "avg_queue_length": total_queue_length / max(step_count, 1),
        "total_throughput": total_throughput,
        "mse": total_mse / len(training_data),
        "mae": total_mae / len(training_data),
        "num_rounds": len(self.round_metrics),
        "round_metrics": self.round_metrics
    }

# Monkey-patch the method
NS3IntegratedFLController.run_simulation_with_ns3 = run_simulation_with_ns3


if __name__ == "__main__":
    main()
