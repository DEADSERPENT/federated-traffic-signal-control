"""
Device Utility Module for GPU/CPU Automatic Selection.
Provides portable device management across different systems.
"""

import torch
import os
from typing import Optional
import warnings


class DeviceManager:
    """
    Centralized device management for the ResilNet-FL project.
    Automatically selects the best available device (GPU or CPU).
    """

    _instance = None
    _device = None

    def __new__(cls):
        """Singleton pattern to ensure consistent device usage across the project."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_device()
        return cls._instance

    def _initialize_device(self):
        """Initialize the best available device."""
        self._device = self._select_device()
        self._print_device_info()

    def _select_device(self) -> torch.device:
        """
        Automatically select the best available device.

        Priority:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU (fallback)

        Returns:
            torch.device: The selected device
        """
        # Check for user-specified device via environment variable
        env_device = os.environ.get('RESILNET_DEVICE', '').lower()
        if env_device:
            if env_device == 'cpu':
                return torch.device('cpu')
            elif env_device == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif env_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')

        # Automatic detection
        if torch.cuda.is_available():
            # Check if CUDA is actually functional
            try:
                # Test CUDA functionality
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                return torch.device('cuda')
            except Exception as e:
                warnings.warn(f"CUDA available but not functional: {e}. Falling back to CPU.")

        # Check for Apple Silicon (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS functionality
                test_tensor = torch.zeros(1, device='mps')
                del test_tensor
                return torch.device('mps')
            except Exception as e:
                warnings.warn(f"MPS available but not functional: {e}. Falling back to CPU.")

        # Default to CPU
        return torch.device('cpu')

    def _print_device_info(self):
        """Print device information for logging purposes."""
        device_type = self._device.type

        if device_type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Device] Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif device_type == 'mps':
            print("[Device] Using Apple Silicon GPU (MPS)")
        else:
            print("[Device] Using CPU")

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    def to_device(self, tensor_or_model):
        """
        Move a tensor or model to the current device.

        Args:
            tensor_or_model: PyTorch tensor or model

        Returns:
            The tensor/model on the current device
        """
        return tensor_or_model.to(self._device)

    def is_gpu(self) -> bool:
        """Check if currently using GPU."""
        return self._device.type in ('cuda', 'mps')

    def get_device_type(self) -> str:
        """Get the device type as a string."""
        return self._device.type

    def empty_cache(self):
        """Clear GPU memory cache if using GPU."""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self._device.type == 'mps':
            # MPS doesn't have a direct cache clearing method yet
            pass


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device() -> torch.device:
    """
    Get the current device for tensor/model operations.
    This is the main function to use throughout the project.

    Usage:
        from utils.device import get_device
        device = get_device()
        model = model.to(device)
        tensor = tensor.to(device)

    Returns:
        torch.device: The best available device
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager.device


def to_device(tensor_or_model):
    """
    Convenience function to move tensor/model to the best device.

    Usage:
        from utils.device import to_device
        model = to_device(model)
        tensor = to_device(tensor)

    Args:
        tensor_or_model: PyTorch tensor or model

    Returns:
        The tensor/model on the current device
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager.to_device(tensor_or_model)


def is_gpu_available() -> bool:
    """
    Check if GPU is being used.

    Returns:
        bool: True if using GPU (CUDA or MPS)
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager.is_gpu()


def empty_gpu_cache():
    """Clear GPU memory cache if available."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    _device_manager.empty_cache()


def reset_device_manager():
    """Reset the device manager (useful for testing)."""
    global _device_manager
    _device_manager = None


# Convenience function to create tensors on the correct device
def tensor(data, dtype=None) -> torch.Tensor:
    """
    Create a tensor on the current device.

    Usage:
        from utils.device import tensor
        x = tensor([1, 2, 3])  # Automatically on GPU if available

    Args:
        data: Data to create tensor from
        dtype: Optional dtype for the tensor

    Returns:
        torch.Tensor on the current device
    """
    device = get_device()
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(data, dtype=dtype, device=device)


if __name__ == "__main__":
    # Test the device module
    print("\n" + "="*50)
    print("Testing Device Module")
    print("="*50)

    device = get_device()
    print(f"\nSelected device: {device}")
    print(f"Is GPU: {is_gpu_available()}")

    # Test tensor creation
    test_tensor = tensor([1.0, 2.0, 3.0])
    print(f"Test tensor device: {test_tensor.device}")

    # Test model movement
    import torch.nn as nn
    test_model = nn.Linear(10, 5)
    test_model = to_device(test_model)
    print(f"Model on device: {next(test_model.parameters()).device}")

    print("\nDevice module test complete!")
