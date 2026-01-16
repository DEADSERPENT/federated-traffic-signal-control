"""
Comprehensive Evaluation Metrics for Traffic Signal Control.
Includes traffic-level and ML-level metrics.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TrafficMetrics:
    """Container for traffic performance metrics."""
    avg_waiting_time: float
    max_waiting_time: float
    avg_queue_length: float
    max_queue_length: float
    total_throughput: int
    throughput_per_hour: float
    vehicles_served: int
    fairness_index: float  # Jain's fairness index


@dataclass
class MLMetrics:
    """Container for ML performance metrics."""
    mse: float
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    r2_score: float


def calculate_traffic_metrics(
    step_metrics: List[Dict],
    intersections: List,
    duration: int
) -> TrafficMetrics:
    """
    Calculate comprehensive traffic metrics.

    Args:
        step_metrics: List of per-step metrics
        intersections: List of intersection objects
        duration: Simulation duration in seconds

    Returns:
        TrafficMetrics object
    """
    waiting_times = [m["avg_waiting_time"] for m in step_metrics]
    queue_lengths = [m["avg_queue_length"] for m in step_metrics]

    # Per-intersection throughput for fairness
    throughputs = [i.total_throughput for i in intersections]

    # Jain's Fairness Index: (sum(x))^2 / (n * sum(x^2))
    if sum(throughputs) > 0:
        fairness = (sum(throughputs) ** 2) / (len(throughputs) * sum(x**2 for x in throughputs))
    else:
        fairness = 1.0

    return TrafficMetrics(
        avg_waiting_time=np.mean(waiting_times),
        max_waiting_time=np.max(waiting_times),
        avg_queue_length=np.mean(queue_lengths),
        max_queue_length=np.max(queue_lengths),
        total_throughput=sum(throughputs),
        throughput_per_hour=sum(throughputs) * (3600 / duration),
        vehicles_served=sum(throughputs),
        fairness_index=fairness
    )


def calculate_ml_metrics(
    predictions: np.ndarray,
    actual: np.ndarray
) -> MLMetrics:
    """
    Calculate comprehensive ML metrics.

    Args:
        predictions: Predicted values
        actual: Actual values

    Returns:
        MLMetrics object
    """
    mse = np.mean((predictions - actual) ** 2)
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(mse)

    # MAPE (avoid division by zero)
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
    else:
        mape = 0.0

    # R² score
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return MLMetrics(
        mse=mse,
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2_score=r2
    )


def calculate_improvement(baseline: float, improved: float, higher_is_better: bool = False) -> float:
    """
    Calculate percentage improvement.

    Args:
        baseline: Baseline value
        improved: Improved value
        higher_is_better: Whether higher values are better

    Returns:
        Improvement percentage
    """
    if baseline == 0:
        return 0.0

    if higher_is_better:
        return ((improved - baseline) / baseline) * 100
    else:
        return ((baseline - improved) / baseline) * 100


def compare_methods(results: List[Dict]) -> Dict[str, Any]:
    """
    Compare multiple methods and generate comparison table.

    Args:
        results: List of result dictionaries from different methods

    Returns:
        Comparison dictionary
    """
    comparison = {
        "methods": [],
        "metrics": {
            "avg_waiting_time": [],
            "avg_queue_length": [],
            "throughput_per_hour": [],
            "mse": [],
            "mae": []
        },
        "improvements": {}
    }

    # Extract metrics
    for result in results:
        method = result.get("method", "Unknown")
        comparison["methods"].append(method)
        comparison["metrics"]["avg_waiting_time"].append(result.get("avg_waiting_time", 0))
        comparison["metrics"]["avg_queue_length"].append(result.get("avg_queue_length", 0))
        comparison["metrics"]["throughput_per_hour"].append(result.get("throughput_per_hour", 0))
        comparison["metrics"]["mse"].append(result.get("mse", 0))
        comparison["metrics"]["mae"].append(result.get("mae", 0))

    # Calculate improvements (assuming first is baseline)
    if len(results) >= 2:
        baseline = results[0]
        for i, result in enumerate(results[1:], 1):
            method = result.get("method", f"Method_{i}")
            comparison["improvements"][method] = {
                "waiting_time_reduction": calculate_improvement(
                    baseline.get("avg_waiting_time", 1),
                    result.get("avg_waiting_time", 1)
                ),
                "queue_reduction": calculate_improvement(
                    baseline.get("avg_queue_length", 1),
                    result.get("avg_queue_length", 1)
                ),
                "throughput_increase": calculate_improvement(
                    baseline.get("throughput_per_hour", 1),
                    result.get("throughput_per_hour", 1),
                    higher_is_better=True
                ),
                "mae_reduction": calculate_improvement(
                    baseline.get("mae", 1),
                    result.get("mae", 1)
                )
            }

    return comparison


def generate_comparison_table(comparison: Dict) -> str:
    """
    Generate formatted comparison table.

    Args:
        comparison: Comparison dictionary

    Returns:
        Formatted table string
    """
    methods = comparison["methods"]
    metrics = comparison["metrics"]

    # Header
    table = "\n" + "=" * 80 + "\n"
    table += "PERFORMANCE COMPARISON TABLE\n"
    table += "=" * 80 + "\n\n"

    # Column widths
    col_width = 18

    # Header row
    header = f"{'Metric':<{col_width}}"
    for method in methods:
        header += f"{method:<{col_width}}"
    table += header + "\n"
    table += "-" * (col_width * (len(methods) + 1)) + "\n"

    # Data rows
    metric_labels = {
        "avg_waiting_time": "Avg Wait Time (s)",
        "avg_queue_length": "Avg Queue Length",
        "throughput_per_hour": "Throughput/hr",
        "mse": "MSE",
        "mae": "MAE"
    }

    for metric_key, label in metric_labels.items():
        row = f"{label:<{col_width}}"
        values = metrics.get(metric_key, [])
        for val in values:
            if isinstance(val, float):
                row += f"{val:<{col_width}.4f}"
            else:
                row += f"{val:<{col_width}}"
        table += row + "\n"

    # Improvements section
    if comparison.get("improvements"):
        table += "\n" + "-" * 80 + "\n"
        table += "IMPROVEMENTS OVER BASELINE (Fixed-Time)\n"
        table += "-" * 80 + "\n"

        for method, imps in comparison["improvements"].items():
            table += f"\n{method}:\n"
            for metric, value in imps.items():
                direction = "(+)" if value > 0 else "(-)"
                table += f"  {metric}: {abs(value):.2f}% {direction}\n"

    table += "=" * 80 + "\n"
    return table


def calculate_convergence_metrics(round_metrics: List[Dict]) -> Dict:
    """
    Calculate convergence-related metrics for FL.

    Args:
        round_metrics: List of per-round metrics

    Returns:
        Convergence metrics
    """
    maes = [m.get("global_mae", m.get("mae", 0)) for m in round_metrics]
    mses = [m.get("global_mse", m.get("mse", 0)) for m in round_metrics]

    # Convergence rate (how fast it improves)
    if len(maes) >= 2:
        early_mae = np.mean(maes[:5])
        late_mae = np.mean(maes[-5:])
        convergence_rate = (early_mae - late_mae) / early_mae * 100 if early_mae > 0 else 0
    else:
        convergence_rate = 0

    # Stability (variance in last 10 rounds)
    if len(maes) >= 10:
        stability = np.std(maes[-10:])
    else:
        stability = np.std(maes)

    # Rounds to 90% improvement
    if len(maes) >= 2:
        improvement_target = maes[0] - 0.9 * (maes[0] - maes[-1])
        rounds_to_90 = next(
            (i for i, m in enumerate(maes) if m <= improvement_target),
            len(maes)
        )
    else:
        rounds_to_90 = 0

    return {
        "convergence_rate": convergence_rate,
        "final_stability": stability,
        "rounds_to_90_percent": rounds_to_90 + 1,
        "initial_mae": maes[0] if maes else 0,
        "final_mae": maes[-1] if maes else 0,
        "total_improvement": ((maes[0] - maes[-1]) / maes[0] * 100) if maes and maes[0] > 0 else 0
    }
