"""Federated Learning Module for Traffic Signal Optimization"""
from .server import FederatedServer, start_server
from .client import TrafficClient, start_client

__all__ = ["FederatedServer", "start_server", "TrafficClient", "start_client"]
