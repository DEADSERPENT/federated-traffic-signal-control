"""
Configuration loader utility.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return get_default_config()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "traffic": {
            "num_intersections": 4,
            "simulation_duration": 3600,
            "time_step": 5,
            "arrival_distribution": "poisson",
            "min_arrival_rate": 5,
            "max_arrival_rate": 30,
            "max_queue_length": 50,
            "min_green_duration": 10,
            "max_green_duration": 90,
            "yellow_duration": 3
        },
        "federated_learning": {
            "num_rounds": 10,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "server_address": "localhost:8080",
            "min_clients": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0
        },
        "model": {
            "type": "neural_network",
            "hidden_layers": [64, 32],
            "activation": "relu",
            "output_activation": "linear"
        },
        "network": {
            "base_latency": 10,
            "bandwidth": 100,
            "packet_loss_probability": 0.01,
            "jitter_range": 5
        },
        "output": {
            "save_metrics": True,
            "save_models": True,
            "visualization": True,
            "results_dir": "results"
        }
    }


if __name__ == "__main__":
    config = load_config()
    print("Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))
