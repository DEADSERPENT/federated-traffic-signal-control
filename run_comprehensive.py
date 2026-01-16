#!/usr/bin/env python
"""
ONE-COMMAND COMPREHENSIVE EXPERIMENT RUNNER
============================================

This script runs ALL experiments and generates:
- Baseline comparisons (Fixed-Time, Local-ML vs Federated Learning)
- Network stress tests (varying latency and packet loss)
- Scalability tests (2, 4, 6, 8 clients)
- Professional visualizations
- Complete experiment report

Usage:
    python run_comprehensive.py          # Full experiment
    python run_comprehensive.py --quick  # Skip scalability (faster)

Output:
    results/comprehensive/
    ├── fl_convergence.png
    ├── method_comparison.png
    ├── network_stress.png
    ├── scalability.png
    ├── summary_dashboard.png
    ├── experiment_report.txt
    └── results.json
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.comprehensive_runner import run_comprehensive_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Federated Learning Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_comprehensive.py              # Full experiment (~10-15 mins)
    python run_comprehensive.py --quick      # Quick mode (~5 mins)

This will generate:
    - Performance comparison: Fixed-Time vs Local-ML vs Federated Learning
    - Network stress analysis: How FL performs under degraded networks
    - Scalability analysis: How FL scales with more clients
    - Publication-quality figures and complete report
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip scalability tests for faster execution"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("""
    ======================================================================
    |                                                                    |
    |   FEDERATED LEARNING-BASED ADAPTIVE TRAFFIC SIGNAL CONTROL         |
    |                                                                    |
    |                 COMPREHENSIVE EXPERIMENT SUITE                     |
    |                                                                    |
    ======================================================================
    """)

    if args.quick:
        print("    Mode: QUICK (scalability tests skipped)")
    else:
        print("    Mode: FULL (all experiments)")

    print(f"    Seed: {args.seed}")
    print()

    # Run experiment
    results = run_comprehensive_experiment(skip_scalability=args.quick)

    print("""
    ======================================================================
    |                     EXPERIMENT COMPLETE!                           |
    ======================================================================

    Results saved to: results/comprehensive/

    Generated files:
    +-- fl_convergence.png       - FL training convergence plot
    +-- method_comparison.png    - Baseline vs FL comparison
    +-- network_stress.png       - Network resilience analysis
    +-- scalability.png          - Scalability analysis
    +-- summary_dashboard.png    - Complete summary dashboard
    +-- experiment_report.txt    - Full text report
    +-- results.json             - Raw results data

    For your thesis/report, use:
    - summary_dashboard.png for overview
    - method_comparison.png for main results
    - fl_convergence.png for FL performance
    """)

    return results


if __name__ == "__main__":
    main()
