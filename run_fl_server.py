"""
Federated Learning Server Runner.
Starts the FL server for traffic signal optimization.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from federated_learning.server import start_server


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Server for Traffic Signal Control"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated rounds (default: 10)"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients (default: 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Federated Learning Server")
    print("Traffic Signal Control System")
    print("=" * 60)
    print(f"\nServer Address: {args.address}")
    print(f"Number of Rounds: {args.rounds}")
    print(f"Minimum Clients: {args.min_clients}")
    print("-" * 60)

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
