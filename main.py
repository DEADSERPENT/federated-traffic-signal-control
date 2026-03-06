#!/usr/bin/env python3
"""
ResilNet-FL — Main Entry Point
================================
Central launcher for all experiment modes.

Usage
-----
    python main.py --mode sumo-gui          # SUMO-GUI + FL  (recommended)
    python main.py --mode ieee              # Full IEEE experiments
    python main.py --mode demo              # Quick demo
    python main.py --mode byzantine         # Byzantine robustness test
    python main.py --mode simulation        # Traffic simulation only

Run  python main.py --help  for all options.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"


MODES = {
    "sumo-gui":   ("run_sumo_gui.py",    "SUMO-GUI + Federated Learning (visual)"),
    "ieee":       ("run_ieee.py",        "Full IEEE publication experiments"),
    "demo":       ("run_demo.py",        "Quick end-to-end demo"),
    "byzantine":  ("run_byzantine.py",   "Byzantine robustness evaluation"),
    "simulation": ("run_simulation.py",  "Traffic simulation only"),
    "cloudsim":   ("run_cloudsim.py",    "CloudSim edge-computing integration"),
    "ns3":        ("run_with_ns3.py",    "NS-3 network simulation"),
    "comprehensive": ("run_comprehensive.py", "Comprehensive experiments"),
    "publication":   ("run_publication.py",   "Publication-quality experiments"),
}


def list_modes():
    print("\nAvailable modes:")
    for key, (_, desc) in MODES.items():
        print(f"  {key:<16} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ResilNet-FL — Traffic Signal Control with Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<16} {v[1]}" for k, v in MODES.items()),
    )
    parser.add_argument(
        "--mode", choices=list(MODES.keys()), default="sumo-gui",
        help="Experiment mode to run (default: sumo-gui)",
    )
    parser.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the chosen script",
    )

    args, unknown = parser.parse_known_args()
    extra = args.extra + unknown

    script_file, description = MODES[args.mode]
    script_path = SCRIPTS / script_file

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    print(f"ResilNet-FL  |  mode: {args.mode}  |  {description}")
    print("-" * 60)

    cmd = [sys.executable, str(script_path)] + extra
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
