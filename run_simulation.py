"""
Main simulation runner for Traffic Signal Control System.
Runs traffic simulation and generates training data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from traffic_generator import TrafficDataGenerator
from utils.visualization import plot_traffic_metrics
from utils.config_loader import load_config


def main():
    print("=" * 60)
    print("Federated Learning-based Adaptive Traffic Signal Control")
    print("Traffic Simulation Module")
    print("=" * 60)

    # Load configuration
    config = load_config("config/config.yaml")
    print("\nConfiguration loaded successfully")

    # Initialize traffic data generator
    print("\nInitializing traffic data generator...")
    generator = TrafficDataGenerator(config=config)
    print(f"Created {len(generator.intersections)} intersections")

    # Run simulation
    duration = config.get("traffic", {}).get("simulation_duration", 3600)
    print(f"\nRunning simulation for {duration} seconds...")

    df = generator.run_simulation(
        duration=duration,
        save_path="data/traffic_simulation.csv"
    )

    print(f"\nSimulation complete!")
    print(f"Total records collected: {len(df)}")

    # Print summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics:")
    print("-" * 40)

    for int_id in df['intersection_id'].unique():
        int_df = df[df['intersection_id'] == int_id]
        print(f"\nIntersection {int_id}:")
        print(f"  Average queue length: {int_df['total_queue_length'].mean():.2f}")
        print(f"  Max queue length: {int_df['total_queue_length'].max()}")
        print(f"  Average waiting time: {int_df['average_waiting_time'].mean():.2f}s")
        print(f"  Final throughput: {int_df['throughput'].iloc[-1]}")

    # Generate training data
    print("\n" + "-" * 40)
    print("Generating training data for each intersection...")
    print("-" * 40)

    for i in range(len(generator.intersections)):
        features, labels = generator.get_intersection_data(i)
        print(f"Intersection {i}: {len(features)} samples, "
              f"avg optimal green: {labels.mean():.2f}s")

    # Visualize results
    if config.get("output", {}).get("visualization", True):
        print("\nGenerating visualization...")
        plot_traffic_metrics(df, save_path="results/traffic_metrics.png")

    print("\n" + "=" * 60)
    print("Simulation complete! Data saved to 'data/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
