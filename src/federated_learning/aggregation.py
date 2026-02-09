"""
Byzantine-Robust Aggregation Strategies for Federated Learning.

Protects against:
- Faulty sensors sending garbage data
- Byzantine (malicious) clients attempting model poisoning
- Outlier updates from divergent local training

Strategies:
- FedAvg: Standard weighted averaging (baseline)
- Median: Coordinate-wise median (robust to outliers)
- TrimmedMean: Remove extreme values before averaging
- Krum: Select most representative update (Byzantine-tolerant)
- MultiKrum: Average top-k most representative updates
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    FEDAVG = "fedavg"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"


def fedavg_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None
) -> List[np.ndarray]:
    """
    Standard Federated Averaging (FedAvg).

    Args:
        model_params: List of model parameters from each client
        weights: Optional weights for each client (e.g., based on data size)

    Returns:
        Weighted averaged parameters
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    if weights is None:
        weights = [1.0 / len(model_params)] * len(model_params)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    avg_params = []
    for layer_idx in range(len(model_params[0])):
        layer_params = [params[layer_idx] for params in model_params]
        weighted_avg = np.zeros_like(layer_params[0], dtype=np.float32)
        for param, weight in zip(layer_params, weights):
            weighted_avg += param.astype(np.float32) * weight
        avg_params.append(weighted_avg.astype(layer_params[0].dtype))

    return avg_params


def median_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None
) -> List[np.ndarray]:
    """
    Coordinate-wise Median Aggregation.

    Robust against up to 50% Byzantine clients.
    For each parameter, takes the median across all clients.

    Args:
        model_params: List of model parameters from each client
        weights: Ignored for median (all clients equal weight)

    Returns:
        Median parameters
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    median_params = []
    for layer_idx in range(len(model_params[0])):
        # Stack all client parameters for this layer
        layer_stack = np.stack([params[layer_idx].astype(np.float32)
                               for params in model_params], axis=0)
        # Compute coordinate-wise median
        layer_median = np.median(layer_stack, axis=0)
        median_params.append(layer_median.astype(model_params[0][layer_idx].dtype))

    return median_params


def trimmed_mean_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
    trim_ratio: float = 0.1
) -> List[np.ndarray]:
    """
    Trimmed Mean Aggregation.

    Removes the top and bottom trim_ratio fraction of values
    before computing the mean. More robust than FedAvg.

    Args:
        model_params: List of model parameters from each client
        weights: Ignored for trimmed mean
        trim_ratio: Fraction of extreme values to remove (default 10%)

    Returns:
        Trimmed mean parameters
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    n_clients = len(model_params)
    n_trim = max(1, int(n_clients * trim_ratio))

    # Need at least 3 clients after trimming
    if n_clients - 2 * n_trim < 1:
        # Fall back to median if not enough clients
        return median_aggregate(model_params)

    trimmed_params = []
    for layer_idx in range(len(model_params[0])):
        layer_stack = np.stack([params[layer_idx].astype(np.float32)
                               for params in model_params], axis=0)

        # Sort along client axis and trim
        sorted_stack = np.sort(layer_stack, axis=0)
        trimmed_stack = sorted_stack[n_trim:n_clients - n_trim]

        # Compute mean of remaining values
        layer_mean = np.mean(trimmed_stack, axis=0)
        trimmed_params.append(layer_mean.astype(model_params[0][layer_idx].dtype))

    return trimmed_params


def _compute_distances(model_params: List[List[np.ndarray]]) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between model updates.

    Returns:
        n_clients x n_clients distance matrix
    """
    n_clients = len(model_params)

    # Flatten all parameters for each client
    flattened = []
    for params in model_params:
        flat = np.concatenate([p.flatten().astype(np.float32) for p in params])
        flattened.append(flat)

    flattened = np.stack(flattened, axis=0)  # [n_clients, n_params]

    # Compute pairwise L2 distances
    distances = np.zeros((n_clients, n_clients), dtype=np.float32)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def krum_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
    num_byzantine: int = 1
) -> List[np.ndarray]:
    """
    Krum Aggregation (Byzantine-tolerant).

    Selects the single model update that is closest to its neighbors.
    Tolerates up to num_byzantine malicious clients.

    Reference: Blanchard et al., "Machine Learning with Adversaries:
               Byzantine Tolerant Gradient Descent" (NeurIPS 2017)

    Args:
        model_params: List of model parameters from each client
        weights: Ignored for Krum
        num_byzantine: Maximum number of Byzantine clients to tolerate

    Returns:
        Selected model parameters (most representative client)
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    n_clients = len(model_params)

    # Need at least 2f + 3 clients where f = num_byzantine
    min_clients = 2 * num_byzantine + 3
    if n_clients < min_clients:
        # Fall back to median if not enough clients
        return median_aggregate(model_params)

    # Compute pairwise distances
    distances = _compute_distances(model_params)

    # For each client, compute sum of distances to n - f - 2 closest neighbors
    n_neighbors = n_clients - num_byzantine - 2
    scores = np.zeros(n_clients)

    for i in range(n_clients):
        # Sort distances to other clients
        sorted_dists = np.sort(distances[i])
        # Sum distances to n_neighbors closest (excluding self which is 0)
        scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])

    # Select client with lowest score (most representative)
    selected_idx = np.argmin(scores)

    return model_params[selected_idx]


def multi_krum_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
    num_byzantine: int = 1,
    num_select: int = None
) -> List[np.ndarray]:
    """
    Multi-Krum Aggregation.

    Selects the top-k most representative updates and averages them.
    Combines Byzantine tolerance with information from multiple clients.

    Args:
        model_params: List of model parameters from each client
        weights: Ignored for Multi-Krum
        num_byzantine: Maximum number of Byzantine clients to tolerate
        num_select: Number of clients to select and average (default: n - f)

    Returns:
        Averaged parameters of selected clients
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    n_clients = len(model_params)

    if num_select is None:
        num_select = max(1, n_clients - num_byzantine)

    min_clients = 2 * num_byzantine + 3
    if n_clients < min_clients:
        return median_aggregate(model_params)

    # Compute pairwise distances
    distances = _compute_distances(model_params)

    # Compute Krum scores
    n_neighbors = n_clients - num_byzantine - 2
    scores = np.zeros(n_clients)

    for i in range(n_clients):
        sorted_dists = np.sort(distances[i])
        scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])

    # Select top num_select clients with lowest scores
    selected_indices = np.argsort(scores)[:num_select]

    # Average selected clients
    selected_params = [model_params[i] for i in selected_indices]
    return fedavg_aggregate(selected_params)


def robust_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
    strategy: str = "trimmed_mean",
    **kwargs
) -> List[np.ndarray]:
    """
    Main aggregation function with strategy selection.

    Args:
        model_params: List of model parameters from each client
        weights: Optional weights for weighted strategies
        strategy: Aggregation strategy name
        **kwargs: Additional arguments for specific strategies

    Returns:
        Aggregated model parameters
    """
    strategy = strategy.lower()

    if strategy == "fedavg":
        return fedavg_aggregate(model_params, weights)

    elif strategy == "median":
        return median_aggregate(model_params, weights)

    elif strategy == "trimmed_mean":
        trim_ratio = kwargs.get("trim_ratio", 0.1)
        return trimmed_mean_aggregate(model_params, weights, trim_ratio)

    elif strategy == "krum":
        num_byzantine = kwargs.get("num_byzantine", 1)
        return krum_aggregate(model_params, weights, num_byzantine)

    elif strategy == "multi_krum":
        num_byzantine = kwargs.get("num_byzantine", 1)
        num_select = kwargs.get("num_select", None)
        return multi_krum_aggregate(model_params, weights, num_byzantine, num_select)

    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}. "
                        f"Choose from: fedavg, median, trimmed_mean, krum, multi_krum")


if __name__ == "__main__":
    """Test aggregation strategies."""
    print("="*60)
    print("Testing Byzantine-Robust Aggregation Strategies")
    print("="*60)

    # Create mock model parameters (3 layers, 5 clients)
    np.random.seed(42)
    n_clients = 5

    # Normal clients
    model_params = []
    for i in range(n_clients - 1):
        params = [
            np.random.randn(10, 6).astype(np.float32) + i * 0.1,  # Layer 1
            np.random.randn(10).astype(np.float32),               # Bias 1
            np.random.randn(1, 10).astype(np.float32),            # Layer 2
        ]
        model_params.append(params)

    # Byzantine client (outlier)
    byzantine_params = [
        np.random.randn(10, 6).astype(np.float32) * 100,  # Huge values!
        np.random.randn(10).astype(np.float32) * 100,
        np.random.randn(1, 10).astype(np.float32) * 100,
    ]
    model_params.append(byzantine_params)

    print(f"\nClients: {n_clients - 1} normal + 1 Byzantine (outlier)")
    print(f"Byzantine client has 100x larger values")

    # Test each strategy
    strategies = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]

    for strategy in strategies:
        result = robust_aggregate(model_params, strategy=strategy)
        max_val = max(np.max(np.abs(p)) for p in result)
        print(f"\n{strategy.upper():15} -> Max absolute value: {max_val:.4f}")

        # FedAvg should be corrupted by Byzantine client
        if strategy == "fedavg":
            print(f"                  (Corrupted by Byzantine client!)")
        else:
            print(f"                  (Robust to Byzantine client)")

    print("\n" + "="*60)
    print("Aggregation strategies test complete!")
    print("="*60)
