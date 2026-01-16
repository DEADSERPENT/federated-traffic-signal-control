"""
CloudSim Simulation Runner
Runs the Python-based edge/cloud simulation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cloudsim_python.edge_cloud_sim import EdgeCloudSimulator


def main():
    print("=" * 70)
    print("  FEDERATED LEARNING-BASED ADAPTIVE TRAFFIC SIGNAL CONTROL SYSTEM")
    print("  CloudSim Edge/Cloud Computing Simulation")
    print("=" * 70)

    # Create simulator
    simulator = EdgeCloudSimulator()

    # Run simulation with 10 FL rounds
    results = simulator.run_simulation(num_fl_rounds=10)

    # Additional analysis
    print("\n" + "=" * 70)
    print("  PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Calculate speedup
    edge_time = results["total_edge_time"]
    cloud_time = results["total_cloud_time"]

    print(f"\nTime Distribution:")
    print(f"  - Edge Computing (Local Training): {edge_time:.2f}s ({edge_time/(edge_time+cloud_time)*100:.1f}%)")
    print(f"  - Cloud Computing (Aggregation): {cloud_time:.2f}s ({cloud_time/(edge_time+cloud_time)*100:.1f}%)")

    # Edge vs Cloud comparison
    edge_mips = 1000  # Per node
    cloud_mips = 10000
    speedup = cloud_mips / edge_mips
    print(f"\n  Cloud vs Edge Speedup: {speedup:.1f}x (based on MIPS ratio)")

    # Parallelism benefit
    num_edge_nodes = 4
    print(f"  Edge Parallelism: {num_edge_nodes} concurrent nodes")

    print("\n" + "=" * 70)
    print("  Simulation completed successfully!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
