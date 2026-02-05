"""
Reproducibility utilities for consistent experimental results.
Fixes random seeds across all libraries.
Integrates with device management for GPU/CPU portability.
"""

import random
import numpy as np
import torch
import os

from utils.device import get_device, is_gpu_available


def set_global_seed(seed: int = 42, verbose: bool = True):
    """
    Set random seed for all libraries to ensure reproducibility.
    Works on both GPU and CPU systems.

    Args:
        seed: Random seed value
        verbose: Whether to print status messages
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PyTorch MPS (Apple Silicon) - set seed if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS uses the same manual_seed
        pass  # Already set via torch.manual_seed

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    if verbose:
        device = get_device()
        print(f"[Reproducibility] Global seed set to {seed} (device: {device})")


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
