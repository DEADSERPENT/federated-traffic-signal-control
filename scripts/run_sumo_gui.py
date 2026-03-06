#!/usr/bin/env python3
"""
SUMO-GUI + Federated Learning Integrated Runner
================================================

Two-phase workflow
------------------
Phase 1 (offline training)
    Train the FL model using the Poisson fallback environment.
    Fast: ~30-60 s, no GUI needed.

Phase 2 (visual deployment)
    Open SUMO-GUI and deploy the trained FL model via TraCI.
    Watch 9 intersections adapt their green phases in real time.
    Results (wait times, queue lengths) are recorded at every step.

Combined results are saved to  results/sumo/sumo_results.json
and a summary plot to          results/sumo/sumo_summary.png

Usage
-----
    # from project root (recommended)
    python scripts/run_sumo_gui.py

    # with options
    python scripts/run_sumo_gui.py --train-rounds 30 --sim-steps 500 --seed 42

    # skip GUI (headless, for CI / servers without display)
    python scripts/run_sumo_gui.py --no-gui

Requirements
------------
    SUMO installed (already detected at SUMO_HOME)
    pip install traci sumolib  (already in venv)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# ── SUMO paths ────────────────────────────────────────────────────────────────
SUMO_HOME = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
SUMO_BIN  = SUMO_HOME / "bin"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"

NETWORK_DIR = PROJECT_ROOT / "sumo" / "networks" / "grid3x3"
RESULTS_DIR = PROJECT_ROOT / "results" / "sumo"

# ── TraCI import (already verified available) ──────────────────────────────────
try:
    sys.path.append(str(SUMO_HOME / "tools"))
    import traci
    TRACI_OK = True
except ImportError:
    TRACI_OK = False


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — build SUMO network with netgenerate
# ─────────────────────────────────────────────────────────────────────────────

def build_network(force: bool = False) -> Path:
    """
    Generate a 3x3 SUMO grid network using netgenerate.
    Returns path to the .net.xml file.
    """
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    net_xml = NETWORK_DIR / "grid3x3.net.xml"

    if net_xml.exists() and not force:
        print(f"[Network] Using existing network: {net_xml.name}")
        return net_xml

    netgenerate = str(SUMO_BIN / ("netgenerate.exe" if sys.platform == "win32" else "netgenerate"))
    print("[Network] Building 3x3 grid with netgenerate ...")
    subprocess.run([
        netgenerate,
        "--grid",
        "--grid.number=3",
        "--grid.length=200",
        "--default.lanenumber=2",
        "--default.speed=13.89",
        "--tls.guess=true",
        "--output-file", str(net_xml),
        "--no-warnings",
    ], check=True)
    print(f"[Network] Written: {net_xml}")
    return net_xml


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — generate vehicle demand with randomTrips.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_demand(net_xml: Path, seed: int = 42, force: bool = False) -> Path:
    """
    Generate random vehicle trips for the 3x3 grid.
    Returns path to the .rou.xml routes file.
    """
    rou_xml = NETWORK_DIR / "grid3x3.rou.xml"

    if rou_xml.exists() and not force:
        print(f"[Demand]  Using existing routes: {rou_xml.name}")
        return rou_xml

    if RANDOM_TRIPS.exists():
        print("[Demand]  Generating routes with randomTrips.py ...")
        trips_xml = NETWORK_DIR / "trips.xml"
        subprocess.run([
            sys.executable, str(RANDOM_TRIPS),
            "-n", str(net_xml),
            "-o", str(trips_xml),
            "-r", str(rou_xml),
            "--end", "3600",
            "--period", "3",        # ~1 200 vehicles / hour
            "--seed", str(seed),
            "--validate",
        ], check=True)
    else:
        # Fallback: hand-craft a simple route file (no randomTrips available)
        _write_simple_routes(net_xml, rou_xml, seed)

    print(f"[Demand]  Written: {rou_xml}")
    return rou_xml


def _write_simple_routes(net_xml: Path, rou_xml: Path, seed: int):
    """Minimal fallback when randomTrips.py is absent."""
    rng = np.random.default_rng(seed)
    lines = [
        '<routes>',
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>',
    ]
    # netgenerate edge ids for a 3x3 grid follow the pattern A0B0, B0C0, etc.
    # Use a simple route that traverses one row to keep it valid.
    sample_routes = [
        "A0B0 B0C0",
        "A1B1 B1C1",
        "A2B2 B2C2",
    ]
    veh_id = 0
    for t in np.sort(rng.uniform(0, 3600, 800)):
        route = sample_routes[veh_id % len(sample_routes)]
        lines.append(
            f'  <vehicle id="v{veh_id}" type="car" depart="{t:.1f}">'
            f'<route edges="{route}"/></vehicle>'
        )
        veh_id += 1
    lines.append("</routes>")
    rou_xml.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — write sumocfg
# ─────────────────────────────────────────────────────────────────────────────

def write_sumocfg(net_xml: Path, rou_xml: Path) -> Path:
    cfg = NETWORK_DIR / "grid3x3.sumocfg"
    cfg.write_text(f"""<configuration>
  <input>
    <net-file value="{net_xml.name}"/>
    <route-files value="{rou_xml.name}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
    <step-length value="1"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>""", encoding="utf-8")
    print(f"[Config]  Written: {cfg.name}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — offline FL training (Poisson fallback, no GUI needed)
# ─────────────────────────────────────────────────────────────────────────────

def train_fl_offline(num_rounds: int, seed: int):
    """
    Train the AdaptiveFLController on the 9-intersection fallback environment.
    Returns (controller, training_metrics).
    """
    from utils.reproducibility import set_global_seed
    from traffic_generator.sumo_integration import SUMOFallbackEnvironment
    from baselines.adaptive_fl import AdaptiveFLController

    set_global_seed(seed)

    print("\n" + "=" * 60)
    print("Phase 1 — Offline FL Training (Poisson fallback)")
    print("=" * 60)

    env = SUMOFallbackEnvironment(max_steps=720, step_size=5.0)
    training_data = env.generate_training_data(num_samples=500)

    controller = AdaptiveFLController(
        num_intersections=9,
        num_rounds=num_rounds,
        local_epochs=10,
    )
    metrics = controller.train_federated(training_data)

    print(f"\n[FL]  Training complete. Final MAE: "
          f"{metrics[-1].get('global_mae', metrics[-1].get('avg_mae', metrics[-1].get('mae', float('nan')))):.4f}")
    return controller, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — deploy trained model in SUMO-GUI via TraCI
# ─────────────────────────────────────────────────────────────────────────────

def run_sumo_gui(controller, cfg_path: Path, sim_steps: int, use_gui: bool):
    """
    Launch SUMO (with or without GUI) and control TLS with the trained FL model.

    Every 10 simulation seconds the model reads the queue state from TraCI
    and sets optimal green-phase durations for each of the 9 TLS controllers.

    Returns list of per-step metrics dicts.
    """
    if not TRACI_OK:
        raise RuntimeError("traci is not importable. Check SUMO installation.")

    binary_name = "sumo-gui.exe" if use_gui else "sumo.exe"
    if sys.platform != "win32":
        binary_name = binary_name.replace(".exe", "")
    sumo_binary = str(SUMO_BIN / binary_name)

    print("\n" + "=" * 60)
    mode = "SUMO-GUI (visual)" if use_gui else "SUMO (headless)"
    print(f"Phase 2 — {mode} Deployment via TraCI")
    print("=" * 60)
    print(f"[TraCI]  Launching {binary_name} ...")

    traci.start([sumo_binary, "-c", str(cfg_path),
                 "--start",           # begin simulation immediately (GUI only)
                 "--quit-on-end"])

    # Discover TLS IDs from the network
    tls_ids = traci.trafficlight.getIDList()
    print(f"[TraCI]  Connected — {len(tls_ids)} traffic lights: {list(tls_ids)}")

    step_metrics = []
    control_interval = 10   # update TLS every 10 sim-seconds

    try:
        for step in range(sim_steps):
            traci.simulationStep()

            # Collect per-intersection queue/wait data
            step_data = {"sim_time": step, "intersections": {}}
            for tls_id in tls_ids:
                lanes   = traci.trafficlight.getControlledLanes(tls_id)
                total_q = sum(traci.lane.getLastStepHaltingNumber(ln) for ln in lanes)
                total_w = sum(traci.lane.getWaitingTime(ln) for ln in lanes)
                step_data["intersections"][tls_id] = {
                    "queue": total_q,
                    "wait":  round(total_w / max(len(lanes), 1), 3),
                }

            step_metrics.append(step_data)

            # Apply FL model every control_interval steps
            if step % control_interval == 0:
                for idx, tls_id in enumerate(tls_ids):
                    lanes   = traci.trafficlight.getControlledLanes(tls_id)
                    queues  = [traci.lane.getLastStepHaltingNumber(ln) for ln in lanes[:4]]
                    # Pad to 4 directions
                    while len(queues) < 4:
                        queues.append(0)
                    phase_idx  = traci.trafficlight.getPhase(tls_id)
                    phase_norm = 1.0 if phase_idx == 0 else 0.0
                    features   = np.array(queues[:4] + [phase_norm, 30.0 / 90.0],
                                          dtype=np.float32)

                    green_dur = float(
                        np.clip(
                            controller.global_model.predict(features.reshape(1, -1))[0],
                            10, 60
                        )
                    )
                    traci.trafficlight.setPhaseDuration(tls_id, green_dur)

            # Progress print every 100 steps
            if step > 0 and step % 100 == 0:
                all_waits = [
                    v["wait"]
                    for d in step_data["intersections"].values()
                    for v in [d]
                ]
                avg_w = np.mean(all_waits) if all_waits else 0.0
                print(f"[TraCI]  Step {step:4d}/{sim_steps}  "
                      f"avg wait = {avg_w:.2f} s")

            if traci.simulation.getMinExpectedNumber() == 0:
                print("[TraCI]  All vehicles finished — ending early.")
                break

    finally:
        traci.close()
        print("[TraCI]  Connection closed.")

    return step_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — save results and plot
# ─────────────────────────────────────────────────────────────────────────────

def save_results(fl_metrics, sumo_metrics, args):
    """Save JSON results and a summary PNG to results/sumo/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── JSON ──
    out = {
        "timestamp": ts,
        "config": vars(args),
        "fl_training": fl_metrics,
        "sumo_simulation": {
            "total_steps": len(sumo_metrics),
            "per_step_sample": sumo_metrics[::50],   # every 50th step to keep file small
        },
    }
    if sumo_metrics:
        all_waits = [
            v["wait"]
            for step in sumo_metrics
            for v in step["intersections"].values()
        ]
        out["sumo_simulation"]["mean_wait_s"]   = round(float(np.mean(all_waits)), 4)
        out["sumo_simulation"]["median_wait_s"] = round(float(np.median(all_waits)), 4)
        out["sumo_simulation"]["max_wait_s"]    = round(float(np.max(all_waits)), 4)

    json_path = RESULTS_DIR / f"sumo_results_{ts}.json"
    # Also overwrite the canonical latest file
    latest_path = RESULTS_DIR / "sumo_results.json"
    for p in (json_path, latest_path):
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[Results] JSON  -> {json_path}")

    # ── Plot ──
    _plot_results(fl_metrics, sumo_metrics, ts)


def _plot_results(fl_metrics, sumo_metrics, ts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResilNet-FL × SUMO-GUI — Combined Results", fontsize=13, fontweight="bold")

    # --- Left: FL training convergence ---
    ax = axes[0]
    if fl_metrics:
        rounds = [m.get("round", i + 1) for i, m in enumerate(fl_metrics)]
        maes   = [m.get("global_mae", m.get("avg_mae", m.get("mae", float("nan")))) for m in fl_metrics]
        maes   = [v for v in maes if not np.isnan(v)]
        if maes:
            ax.plot(rounds[:len(maes)], maes, "b-o", markersize=3, linewidth=1.5)
            ax.set_xlabel("FL Round")
            ax.set_ylabel("MAE (seconds)")
            ax.set_title("FL Training Convergence")
            ax.grid(True, alpha=0.3)

    # --- Right: SUMO per-intersection mean wait ---
    ax = axes[1]
    if sumo_metrics:
        tls_ids = list(sumo_metrics[0]["intersections"].keys())
        for tls_id in tls_ids:
            waits = [s["intersections"][tls_id]["wait"] for s in sumo_metrics]
            ax.plot(waits, linewidth=0.8, alpha=0.75, label=tls_id)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Avg Wait Time (s)")
        ax.set_title("SUMO: Per-Intersection Wait Time")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = RESULTS_DIR / f"sumo_summary_{ts}.png"
    latest_png = RESULTS_DIR / "sumo_summary.png"
    for p in (png_path, latest_png):
        plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[Results] Plot  -> {png_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SUMO-GUI + FL integrated runner for ResilNet-FL"
    )
    parser.add_argument("--train-rounds", type=int, default=30,
                        help="FL training rounds in Phase 1 (default 30)")
    parser.add_argument("--sim-steps", type=int, default=600,
                        help="SUMO simulation steps in Phase 2 (default 600 = 10 min)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default 42)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run SUMO headless (no GUI window)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of SUMO network and routes")
    args = parser.parse_args()

    use_gui = not args.no_gui

    print("=" * 60)
    print("ResilNet-FL — SUMO-GUI Integration Runner")
    print(f"  SUMO_HOME : {SUMO_HOME}")
    print(f"  TraCI     : {'available' if TRACI_OK else 'NOT available'}")
    print(f"  Mode      : {'GUI' if use_gui else 'Headless'}")
    print(f"  FL rounds : {args.train_rounds}")
    print(f"  Sim steps : {args.sim_steps}")
    print("=" * 60)

    if not TRACI_OK:
        print("\nERROR: traci is not importable.")
        print("Run:  pip install traci  or check SUMO_HOME.")
        sys.exit(1)

    # -- Phase 0: build SUMO assets -------------------------------------------
    net_xml = build_network(force=args.rebuild)
    rou_xml = generate_demand(net_xml, seed=args.seed, force=args.rebuild)
    cfg_path = write_sumocfg(net_xml, rou_xml)

    # -- Phase 1: offline FL training -----------------------------------------
    controller, fl_metrics = train_fl_offline(args.train_rounds, args.seed)

    # -- Phase 2: SUMO-GUI deployment -----------------------------------------
    print(f"\n{'='*60}")
    if use_gui:
        print("SUMO-GUI will open now.")
        print("  - Watch the 9-intersection 3x3 grid animate in real time.")
        print("  - The trained FL model controls every traffic light.")
        print("  - Close the SUMO-GUI window or wait for all vehicles to finish.")
        print(f"{'='*60}")
    sumo_metrics = run_sumo_gui(controller, cfg_path, args.sim_steps, use_gui)

    # -- Save combined results ------------------------------------------------
    save_results(fl_metrics, sumo_metrics, args)

    # -- Summary print --------------------------------------------------------
    if sumo_metrics:
        all_waits = [
            v["wait"]
            for step in sumo_metrics
            for v in step["intersections"].values()
        ]
        print(f"\n{'='*60}")
        print("COMBINED RESULTS SUMMARY")
        final_mae = fl_metrics[-1].get('global_mae', fl_metrics[-1].get('avg_mae', fl_metrics[-1].get('mae', float('nan'))))
        print(f"  FL final MAE          : {final_mae:.4f}")
        print(f"  SUMO mean wait        : {np.mean(all_waits):.2f} s")
        print(f"  SUMO median wait      : {np.median(all_waits):.2f} s")
        print(f"  SUMO max wait         : {np.max(all_waits):.2f} s")
        print(f"  Simulation steps      : {len(sumo_metrics)}")
        print(f"  Results saved to      : {RESULTS_DIR}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
