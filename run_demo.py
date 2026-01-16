"""
Complete Demo Runner for Traffic Signal Control System.
Runs traffic simulation, local training, and demonstrates the full system.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from traffic_generator import TrafficDataGenerator
from models.traffic_model import create_model, train_model, evaluate_model
from network_simulation import NetworkSimulator
from utils.visualization import plot_training_metrics, plot_traffic_metrics
from utils.config_loader import load_config


def run_demo():
    print("=" * 70)
    print("  FEDERATED LEARNING-BASED ADAPTIVE TRAFFIC SIGNAL CONTROL SYSTEM")
    print("  Complete System Demonstration")
    print("=" * 70)

    # Load configuration
    config = load_config("config/config.yaml")
    print("\n[1/6] Configuration loaded successfully")

    # Step 1: Initialize traffic simulation
    print("\n" + "-" * 50)
    print("[2/6] Initializing Traffic Simulation...")
    print("-" * 50)

    generator = TrafficDataGenerator(config=config)
    print(f"  - Created {len(generator.intersections)} intersections")

    # Run short simulation
    print("  - Running traffic simulation (5 minutes)...")
    df = generator.run_simulation(duration=300, save_path="data/demo_simulation.csv")
    print(f"  - Collected {len(df)} simulation records")

    # Step 2: Generate training data for each edge node
    print("\n" + "-" * 50)
    print("[3/6] Generating Training Data for Edge Nodes...")
    print("-" * 50)

    edge_data = {}
    for i in range(len(generator.intersections)):
        features, labels = generator.get_intersection_data(i)
        edge_data[i] = (features, labels)
        print(f"  - Edge Node {i}: {len(features)} samples generated")

    # Step 3: Local model training (simulating FL clients)
    print("\n" + "-" * 50)
    print("[4/6] Local Model Training (Simulating Federated Learning)...")
    print("-" * 50)

    models = {}
    all_losses = {}

    for edge_id, (features, labels) in edge_data.items():
        print(f"\n  Edge Node {edge_id}:")

        # Create local model
        model = create_model("neural_network", hidden_layers=[64, 32])

        # Train locally
        model, losses = train_model(
            model,
            (features, labels),
            epochs=10,
            batch_size=32,
            learning_rate=0.01
        )

        models[edge_id] = model
        all_losses[edge_id] = losses

        print(f"    - Initial Loss: {losses[0]:.4f}")
        print(f"    - Final Loss: {losses[-1]:.4f}")
        print(f"    - Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    # Step 4: Simulate FedAvg aggregation
    print("\n" + "-" * 50)
    print("[5/6] Simulating Federated Averaging (FedAvg)...")
    print("-" * 50)

    # Get parameters from all models
    all_params = [models[i].get_parameters() for i in range(len(models))]

    # Average parameters (FedAvg)
    avg_params = []
    for i in range(len(all_params[0])):
        layer_params = [params[i] for params in all_params]
        avg_params.append(np.mean(layer_params, axis=0))

    # Create global model and set averaged parameters
    global_model = create_model("neural_network", hidden_layers=[64, 32])
    global_model.set_parameters(avg_params)
    print("  - Aggregated model parameters from all edge nodes")

    # Evaluate global model on combined test data
    combined_features = np.vstack([edge_data[i][0][-100:] for i in range(len(edge_data))])
    combined_labels = np.concatenate([edge_data[i][1][-100:] for i in range(len(edge_data))])

    mse, mae = evaluate_model(global_model, (combined_features, combined_labels))
    print(f"  - Global Model MSE: {mse:.4f}")
    print(f"  - Global Model MAE: {mae:.4f}")

    # Step 5: Network simulation
    print("\n" + "-" * 50)
    print("[6/6] Network Communication Simulation...")
    print("-" * 50)

    network = NetworkSimulator(config)

    # Simulate model parameter transmission
    for edge_id in range(len(models)):
        params = models[edge_id].get_parameters()
        latency, success, _ = network.simulate_model_update(params, edge_id)
        status = "SUCCESS" if success else "FAILED (packet loss)"
        print(f"  - Edge Node {edge_id} -> Server: {latency:.2f}ms [{status}]")

    network_metrics = network.get_metrics()
    print(f"\n  Network Summary:")
    print(f"    - Total packets sent: {network_metrics['total_packets_sent']}")
    print(f"    - Packet loss rate: {network_metrics['packet_loss_rate']*100:.2f}%")
    print(f"    - Average latency: {network_metrics['average_latency_ms']:.2f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nSystem Components Demonstrated:")
    print("  [x] Traffic Data Generator - Real-time traffic simulation")
    print("  [x] ML Model - Neural network for signal optimization")
    print("  [x] Local Training - Edge node model training")
    print("  [x] FedAvg Aggregation - Federated model averaging")
    print("  [x] Network Simulation - Communication latency modeling")
    print("\nOutput Files:")
    print("  - data/demo_simulation.csv - Traffic simulation data")
    print("\nTo run full federated learning:")
    print("  1. Start server: python run_fl_server.py")
    print("  2. Start clients: python run_fl_client.py --intersection 0")
    print("                    python run_fl_client.py --intersection 1")
    print("=" * 70)

    return df, models, global_model


if __name__ == "__main__":
    run_demo()
