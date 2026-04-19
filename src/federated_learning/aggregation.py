"""
Byzantine-Robust Aggregation Strategies for Federated Learning.

Protects against:
- Faulty sensors sending garbage data
- Byzantine (malicious) clients attempting model poisoning
- Outlier updates from divergent local training
- Directional (ALIE-style) scaling attacks

Strategies:
- FedAvg:       Standard weighted averaging (baseline)
- Median:       Coordinate-wise median (robust to outliers)
- TrimmedMean:  Remove extreme values before averaging
- Krum:         Select most representative update (Byzantine-tolerant)
- MultiKrum:    Average top-k most representative updates
- ResilAgg:     Novel two-stage MAD-filtered quality-aware aggregation
                Designed for adversarial, highly Non-IID ITS deployments
"""

import numpy as np
import warnings
from typing import List, Optional, Tuple
from enum import Enum

# Import GPU-accelerated distance backend.
# Falls back gracefully to torch.cdist (GPU) or numpy loops (CPU).
try:
    from federated_learning.cuda_krum import compute_model_distances as _gpu_distances
    _USE_GPU_DISTANCES = True
except ImportError:
    try:
        from cuda_krum import compute_model_distances as _gpu_distances
        _USE_GPU_DISTANCES = True
    except ImportError:
        _USE_GPU_DISTANCES = False


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    FEDAVG = "fedavg"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    RESIL_AGG = "resil_agg"
    H_FL = "h_fl"             # Hierarchical Byzantine-Robust FL


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_params(model_params: List[List[np.ndarray]]) -> np.ndarray:
    """Flatten each client's parameter list into a single 1-D vector."""
    flattened = []
    for params in model_params:
        flat = np.concatenate([p.flatten().astype(np.float32) for p in params])
        flattened.append(flat)
    return np.stack(flattened, axis=0)  # [n_clients, n_params]


def _compute_distances(model_params: List[List[np.ndarray]]) -> np.ndarray:
    """
    Pairwise Euclidean distance matrix (used by Krum / Multi-Krum).

    Dispatches to the GPU-accelerated backend (custom CUDA kernel or
    torch.cdist) when available, falling back to a NumPy loop otherwise.

    Returns:
        n_clients × n_clients float32 distance matrix
    """
    if _USE_GPU_DISTANCES:
        return _gpu_distances(model_params)

    # CPU fallback
    flattened = _flatten_params(model_params)
    n = flattened.shape[0]
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def _compute_hybrid_distances(flattened: np.ndarray) -> np.ndarray:
    """
    Hybrid distance matrix fusing L2 magnitude and cosine direction.

    Motivation
    ----------
    Standard L2-only aggregators (Krum, Multi-Krum) are vulnerable to
    "A Little Is Enough" (ALIE) attacks [Baruch et al., 2019] where
    adversaries scale their updates to sit just inside the Euclidean
    clipping radius while inverting the gradient direction.

    The hybrid metric

        d_hybrid(i, j) = ||u_i - u_j||_2 * (1 + (1 - cos(u_i, u_j)))

    simultaneously penalises both magnitude divergence (L2 term) and
    directional inversion (cosine term), making it significantly harder
    to craft updates that evade detection on both axes at once.

    Args:
        flattened: [n_clients, n_params] float32 array

    Returns:
        n_clients × n_clients float32 hybrid distance matrix
    """
    n = flattened.shape[0]
    distances = np.zeros((n, n), dtype=np.float32)

    norms = np.linalg.norm(flattened, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)  # guard against zero-norm updates

    for i in range(n):
        for j in range(i + 1, n):
            l2_dist = np.linalg.norm(flattened[i] - flattened[j])
            cos_sim = np.dot(flattened[i], flattened[j]) / (norms[i] * norms[j])
            cos_dist = 1.0 - float(cos_sim)
            hybrid = l2_dist * (1.0 + cos_dist)
            distances[i, j] = hybrid
            distances[j, i] = hybrid

    return distances


# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD BASELINES
# ─────────────────────────────────────────────────────────────────────────────

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
        layer_stack = np.stack([params[layer_idx].astype(np.float32)
                               for params in model_params], axis=0)
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

    if n_clients - 2 * n_trim < 1:
        return median_aggregate(model_params)

    trimmed_params = []
    for layer_idx in range(len(model_params[0])):
        layer_stack = np.stack([params[layer_idx].astype(np.float32)
                               for params in model_params], axis=0)
        sorted_stack = np.sort(layer_stack, axis=0)
        trimmed_stack = sorted_stack[n_trim:n_clients - n_trim]
        layer_mean = np.mean(trimmed_stack, axis=0)
        trimmed_params.append(layer_mean.astype(model_params[0][layer_idx].dtype))

    return trimmed_params


def krum_aggregate(
    model_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
    num_byzantine: int = 1
) -> List[np.ndarray]:
    """
    Krum Aggregation (Byzantine-tolerant).

    Selects the single model update closest to its n - f - 2 neighbours.

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
    if n_clients < 2 * num_byzantine + 3:
        return median_aggregate(model_params)

    distances = _compute_distances(model_params)
    n_neighbors = n_clients - num_byzantine - 2
    scores = np.zeros(n_clients)
    for i in range(n_clients):
        sorted_dists = np.sort(distances[i])
        scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])

    return model_params[int(np.argmin(scores))]


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

    if n_clients < 2 * num_byzantine + 3:
        return median_aggregate(model_params)

    distances = _compute_distances(model_params)
    n_neighbors = n_clients - num_byzantine - 2
    scores = np.zeros(n_clients)
    for i in range(n_clients):
        sorted_dists = np.sort(distances[i])
        scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])

    selected_indices = np.argsort(scores)[:num_select]
    return fedavg_aggregate([model_params[i] for i in selected_indices])


# ─────────────────────────────────────────────────────────────────────────────
#  NOVEL: ResilAgg
# ─────────────────────────────────────────────────────────────────────────────

def resil_agg_aggregate(
    model_params: List[List[np.ndarray]],
    losses: List[float],
    data_sizes: List[int],
    mad_threshold: float = 3.0,
    epsilon: float = 1e-5,
) -> List[np.ndarray]:
    """
    ResilAgg: Dynamic MAD-Filtered Quality-Aware Aggregation.

    Two-stage design for adversarial, heterogeneous ITS deployments:

    Stage 1 — Dynamic Byzantine Filtering (no need to know f a priori)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Classical Krum / Multi-Krum require the operator to hard-code the number
    of Byzantine nodes f before training begins.  In real urban deployments,
    f is unknown and varies as sensors break or recover.

    ResilAgg instead computes each client's *neighbourhood score*
    (sum of hybrid distances to all peers) and applies a Modified Z-score
    filter based on the Median Absolute Deviation (MAD):

        z_i = 0.6745 * (score_i - median_score) / MAD(scores)

    Clients with z_i > mad_threshold (default 3.0, equivalent to ~3σ in a
    Gaussian world and recommended by [Iglewicz & Hoaglin, 1993]) are
    classified as outliers and dropped.  Because the threshold is derived
    from the data each round rather than a fixed f, the method adapts
    automatically to the current attack intensity.

    The hybrid distance metric (L2 × (1 + cosine_distance)) catches both
    magnitude-scaled attacks and direction-inverting (ALIE-style) attacks
    that pure-L2 aggregators miss [Baruch et al., 2019].

    Stage 2 — Quality-Aware Aggregation on Honest Survivors
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    After Byzantine clients are removed, surviving clients are aggregated
    using Quality-Aware (inverse-loss × data-size) weights:

        α_k = n_k / (L_k + ε)    (unnormalised)

    This ensures that honest-but-statistically-distant clients (e.g. a CBD
    intersection with extreme Non-IID traffic) still contribute in proportion
    to their data quality, addressing the Non-IID vs. Byzantine dilemma
    identified in recent ITS-FL literature.

    Fallback behaviour
    ------------------
    - < 3 clients:           falls back to FedAvg.
    - All clients filtered:  falls back to strict Krum (argmin score).

    Args:
        model_params:   List of per-client model parameter lists (numpy arrays).
        losses:         Local training loss for each client (lower = better).
        data_sizes:     Number of local training samples per client.
        mad_threshold:  Modified Z-score threshold for outlier rejection (default 3.0).
        epsilon:        Numerical stability constant for inverse-loss weights.

    Returns:
        Aggregated model parameters as a list of numpy arrays.

    References
    ----------
    - Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
      Gradient Descent." NeurIPS 2017.
    - Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed
      Learning." NeurIPS 2019.
    - Iglewicz & Hoaglin, "How to Detect and Handle Outliers." ASQ Press, 1993.
    - Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal
      Statistical Rates." ICML 2018.
    """
    if not model_params:
        raise ValueError("No model parameters provided")

    n_clients = len(model_params)

    if n_clients < 3:
        warnings.warn(
            "ResilAgg requires >= 3 clients for MAD filtering. "
            "Falling back to FedAvg.",
            stacklevel=2,
        )
        return fedavg_aggregate(model_params)

    # ── Stage 1: Dynamic Byzantine filtering via MAD ─────────────────────────
    flattened = _flatten_params(model_params)
    distances = _compute_hybrid_distances(flattened)

    # Neighbourhood score: sum of hybrid distances to all other clients.
    # Byzantine clients (far from the honest cluster) get high scores.
    client_scores = np.sum(distances, axis=1)

    median_score = np.median(client_scores)
    mad = np.median(np.abs(client_scores - median_score))
    mad = max(mad, 1e-8)  # prevent division by zero when all updates are identical

    # Modified Z-score (Iglewicz & Hoaglin, 1993)
    z_scores = 0.6745 * (client_scores - median_score) / mad

    # Survivors: clients whose scores are not anomalously high
    survivor_indices = [i for i, z in enumerate(z_scores) if z <= mad_threshold]

    # Fallback when attack is so massive (>50% colluding) that MAD breaks
    if len(survivor_indices) == 0:
        survivor_indices = [int(np.argmin(client_scores))]  # strict Krum fallback

    # ── Stage 2: Quality-Aware aggregation on surviving honest clients ────────
    survivor_params = [model_params[i] for i in survivor_indices]
    survivor_losses = [losses[i] for i in survivor_indices]
    survivor_sizes  = [data_sizes[i] for i in survivor_indices]

    # α_k = n_k / (L_k + ε)  — reward large, low-loss nodes
    raw_weights = [
        sz * (1.0 / (lk + epsilon))
        for sz, lk in zip(survivor_sizes, survivor_losses)
    ]
    total_weight = sum(raw_weights)
    norm_weights = [w / total_weight for w in raw_weights]

    avg_params = []
    for layer_idx in range(len(survivor_params[0])):
        layer_tensors = [p[layer_idx] for p in survivor_params]
        weighted_avg = np.zeros_like(layer_tensors[0], dtype=np.float32)
        for param, w in zip(layer_tensors, norm_weights):
            weighted_avg += param.astype(np.float32) * w
        avg_params.append(weighted_avg.astype(survivor_params[0][layer_idx].dtype))

    return avg_params


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ROUTING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

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
        weights: Optional weights for weighted strategies (used by fedavg)
        strategy: Aggregation strategy name — one of:
                  fedavg | median | trimmed_mean | krum | multi_krum | resil_agg
        **kwargs: Strategy-specific options:
                  - trim_ratio (float):    for trimmed_mean
                  - num_byzantine (int):   for krum / multi_krum
                  - num_select (int):      for multi_krum
                  - losses (list[float]):  for resil_agg
                  - data_sizes (list[int]):for resil_agg
                  - mad_threshold (float): for resil_agg

    Returns:
        Aggregated model parameters
    """
    strategy = strategy.lower()

    if strategy == "fedavg":
        return fedavg_aggregate(model_params, weights)

    elif strategy == "median":
        return median_aggregate(model_params, weights)

    elif strategy == "trimmed_mean":
        return trimmed_mean_aggregate(
            model_params, weights,
            trim_ratio=kwargs.get("trim_ratio", 0.1)
        )

    elif strategy == "krum":
        return krum_aggregate(
            model_params, weights,
            num_byzantine=kwargs.get("num_byzantine", 1)
        )

    elif strategy == "multi_krum":
        return multi_krum_aggregate(
            model_params, weights,
            num_byzantine=kwargs.get("num_byzantine", 1),
            num_select=kwargs.get("num_select", None)
        )

    elif strategy == "resil_agg":
        n = len(model_params)
        losses     = kwargs.get("losses",     [1.0] * n)
        data_sizes = kwargs.get("data_sizes", [1]   * n)
        return resil_agg_aggregate(
            model_params,
            losses=losses,
            data_sizes=data_sizes,
            mad_threshold=kwargs.get("mad_threshold", 3.0),
        )

    elif strategy == "h_fl":
        # Hierarchical FL: lazy import to avoid circular dependency
        try:
            from federated_learning.hierarchical import hierarchical_aggregate
        except ImportError:
            from hierarchical import hierarchical_aggregate
        n = len(model_params)
        return hierarchical_aggregate(
            model_params=model_params,
            losses=kwargs.get("losses",     [1.0] * n),
            data_sizes=kwargs.get("data_sizes", [1] * n),
            num_intersections=n,
            fog_strategy=kwargs.get("fog_strategy", "resil_agg"),
            cloud_strategy=kwargs.get("cloud_strategy", "multi_krum"),
            num_clusters=kwargs.get("num_clusters", 3),
        )

    else:
        raise ValueError(
            f"Unknown aggregation strategy: '{strategy}'. "
            "Choose from: fedavg, median, trimmed_mean, krum, multi_krum, "
            "resil_agg, h_fl"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Aggregation Strategies (incl. ResilAgg)")
    print("=" * 60)

    np.random.seed(42)
    n_clients = 7  # 5 honest + 2 Byzantine

    honest_params = [
        [
            np.random.randn(10, 6).astype(np.float32) + i * 0.05,
            np.random.randn(10).astype(np.float32),
            np.random.randn(1, 10).astype(np.float32),
        ]
        for i in range(n_clients - 2)
    ]

    byzantine_params = [
        [
            np.random.randn(10, 6).astype(np.float32) * 100,
            np.random.randn(10).astype(np.float32) * 100,
            np.random.randn(1, 10).astype(np.float32) * 100,
        ]
        for _ in range(2)
    ]

    all_params  = honest_params + byzantine_params
    losses      = [0.05 + i * 0.01 for i in range(n_clients - 2)] + [999.0, 999.0]
    data_sizes  = [500] * (n_clients - 2) + [1, 1]

    print(f"\nClients: {n_clients - 2} honest + 2 Byzantine (100x noise)")

    strategies = [
        ("fedavg",       {}),
        ("median",       {}),
        ("trimmed_mean", {"trim_ratio": 0.15}),
        ("krum",         {"num_byzantine": 2}),
        ("multi_krum",   {"num_byzantine": 2}),
        ("resil_agg",    {"losses": losses, "data_sizes": data_sizes}),
    ]

    for name, kwargs in strategies:
        result  = robust_aggregate(all_params, strategy=name, **kwargs)
        max_val = max(np.max(np.abs(p)) for p in result)
        tag = "(Corrupted!)" if max_val > 10 else "(Robust    )"
        print(f"  {name:<14} max |w| = {max_val:10.4f}  {tag}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
