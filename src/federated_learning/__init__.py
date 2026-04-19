"""Federated Learning Module for Traffic Signal Optimization"""
from .server import FederatedServer, start_server
from .client import TrafficClient, start_client
from .aggregation import (
    robust_aggregate,
    resil_agg_aggregate,
    fedavg_aggregate,
    AggregationStrategy,
)
from .hierarchical import (
    hierarchical_aggregate,
    HierarchicalFLController,
    FogNode,
    assign_clusters_balanced,
    CLUSTER_LABELS,
)
from .cuda_krum import pairwise_l2_gpu, compute_model_distances

__all__ = [
    # Flower server/client
    "FederatedServer", "start_server", "TrafficClient", "start_client",
    # Aggregation
    "robust_aggregate", "resil_agg_aggregate", "fedavg_aggregate",
    "AggregationStrategy",
    # Hierarchical FL
    "hierarchical_aggregate", "HierarchicalFLController", "FogNode",
    "assign_clusters_balanced", "CLUSTER_LABELS",
    # GPU distance backend
    "pairwise_l2_gpu", "compute_model_distances",
]
