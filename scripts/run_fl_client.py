"""
Federated Learning Client Runner.
Starts an FL client representing a traffic intersection.
"""

import sys
import os
import argparse

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from federated_learning.client import start_client


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Client for Traffic Signal Control"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)"
    )
    parser.add_argument(
        "--intersection",
        type=int,
        default=0,
        help="Intersection ID (default: 0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to configuration file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Federated Learning Client - Intersection {args.intersection}")
    print("Traffic Signal Control System")
    print("=" * 60)
    print(f"\nServer Address: {args.server}")
    print(f"Intersection ID: {args.intersection}")
    print("-" * 60)

    start_client(
        server_address=args.server,
        intersection_id=args.intersection,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
