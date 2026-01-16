"""
Reproducibility utilities for consistent experimental results.
Fixes random seeds across all libraries.
"""

import random
import numpy as np
import torch
import os


def set_global_seed(seed: int = 42):
    """
    Set random seed for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[Reproducibility] Global seed set to {seed}")


def get_experiment_id() -> str:
    """Generate unique experiment ID based on timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ExperimentLogger:
    """Logger for experiment tracking."""

    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_id = get_experiment_id()
        self.logs = []

        os.makedirs(log_dir, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(entry)
        print(entry)

    def save(self):
        """Save logs to file."""
        log_file = os.path.join(
            self.log_dir,
            f"{self.experiment_name}_{self.experiment_id}.log"
        )
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.logs))
        print(f"[Logger] Logs saved to {log_file}")
        return log_file
