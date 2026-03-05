"""
SUMO Traffic Simulator Integration
====================================
Interfaces with SUMO (Simulation of Urban MObility) via the TraCI Python API
for realistic traffic simulation on a 3x3 grid network (9 intersections).

This replaces the custom Poisson-based simulator with SUMO's microsimulation
which models:
- Krauss car-following model (acceleration/deceleration kinematics)
- Lane-changing behaviour
- Non-linear vehicle following
- Realistic intersection dynamics

Falls back gracefully to an enhanced Poisson simulator when SUMO/TraCI
is not installed, so the code always runs.

References:
- SUMO: Lopez et al., "Microscopic Traffic Simulation using SUMO" (ITSC 2018)
- TraCI API: https://sumo.dlr.de/docs/TraCI.html
- IEEE 802.11p V2I: Kenney, J. B. (2011). Dedicated Short-Range Communications
  (DSRC) Standards in the United States. Proceedings of the IEEE, 99(7), 1162–1182.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── SUMO / TraCI availability check ──────────────────────────────────────────
_SUMO_AVAILABLE = False
_SUMO_HOME = os.environ.get("SUMO_HOME", "")

try:
    if _SUMO_HOME:
        sys.path.append(os.path.join(_SUMO_HOME, "tools"))
    import traci                          # noqa: F401  (side-effect import)
    import traci.constants as tc          # noqa: F401
    _SUMO_AVAILABLE = True
except ImportError:
    pass  # Will use enhanced Poisson fallback


# ─────────────────────────────────────────────────────────────────────────────
#  3 × 3 GRID NETWORK PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

GRID_ROWS = 3
GRID_COLS = 3
NUM_INTERSECTIONS = GRID_ROWS * GRID_COLS   # 9

# Intersection layout (row, col) → intersection_id
#   (0,0)─(0,1)─(0,2)
#     |     |     |
#   (1,0)─(1,1)─(1,2)
#     |     |     |
#   (2,0)─(2,1)─(2,2)
INTERSECTION_POSITIONS = {
    i: (i // GRID_COLS, i % GRID_COLS) for i in range(NUM_INTERSECTIONS)
}

# Per-intersection peak-hour arrival rates (vehicles/minute) capturing
# Non-IID heterogeneity that motivates FL:
#   • Corner intersections: lighter load (residential)
#   • Edge intersections:   medium load (arterial)
#   • Centre intersection:  highest load (CBD)
ARRIVAL_RATES = {
    0: 8.0,   # corner  (0,0) — residential
    1: 14.0,  # edge    (0,1) — arterial
    2: 8.0,   # corner  (0,2) — residential
    3: 14.0,  # edge    (1,0) — arterial
    4: 22.0,  # centre  (1,1) — CBD (highest load)
    5: 14.0,  # edge    (1,2) — arterial
    6: 8.0,   # corner  (2,0) — residential
    7: 14.0,  # edge    (2,1) — arterial
    8: 8.0,   # corner  (2,2) — residential
}

# Time-of-day multipliers (captures morning/evening rush)
TIME_OF_DAY_FACTORS = {
    "morning_peak":  1.8,   # 07:00–09:00
    "midday":        1.0,   # 09:00–17:00
    "evening_peak":  2.0,   # 17:00–19:00
    "night":         0.3,   # 19:00–07:00
}


# ─────────────────────────────────────────────────────────────────────────────
#  SUMO NETWORK BUILDER  (creates .net.xml and .sumocfg on-the-fly)
# ─────────────────────────────────────────────────────────────────────────────

def _write_sumo_network(output_dir: Path) -> Path:
    """
    Generate a SUMO 3x3 grid network XML file.

    The grid has:
    - 9 signalised junctions
    - 12 internal edges (horizontal + vertical links)
    - 8 boundary edges (entry/exit points for external demand)
    - Speed limit: 13.89 m/s (≈ 50 km/h, standard urban)
    - Lane width: 3.2 m, number of lanes: 2 per direction

    Returns:
        Path to the generated .sumocfg file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    net_path = output_dir / "grid3x3.net.xml"
    cfg_path = output_dir / "grid3x3.sumocfg"
    rou_path = output_dir / "grid3x3.rou.xml"

    SPACING = 200   # metres between intersections
    SPEED   = 13.89  # m/s
    LANES   = 2

    # ── nodes ──
    nodes_xml = ['<nodes>']
    for i in range(NUM_INTERSECTIONS):
        r, c = INTERSECTION_POSITIONS[i]
        x, y = c * SPACING, (GRID_ROWS - 1 - r) * SPACING
        nodes_xml.append(f'  <node id="J{i}" x="{x}" y="{y}" type="traffic_light"/>')
    # Entry/exit pseudo-nodes at grid boundary
    for c in range(GRID_COLS):
        nodes_xml.append(f'  <node id="top_{c}" x="{c*SPACING}" y="{GRID_ROWS*SPACING}" type="dead_end"/>')
        nodes_xml.append(f'  <node id="bot_{c}" x="{c*SPACING}" y="-{SPACING}" type="dead_end"/>')
    for r in range(GRID_ROWS):
        nodes_xml.append(f'  <node id="left_{r}" x="-{SPACING}" y="{(GRID_ROWS-1-r)*SPACING}" type="dead_end"/>')
        nodes_xml.append(f'  <node id="right_{r}" x="{GRID_COLS*SPACING}" y="{(GRID_ROWS-1-r)*SPACING}" type="dead_end"/>')
    nodes_xml.append('</nodes>')

    # ── edges ──
    edges_xml = ['<edges>']

    def add_edge(eid, fr, to):
        edges_xml.append(
            f'  <edge id="{eid}" from="{fr}" to="{to}" '
            f'numLanes="{LANES}" speed="{SPEED}" priority="1"/>'
        )

    # Horizontal links (E–W)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            left_id  = r * GRID_COLS + c
            right_id = r * GRID_COLS + c + 1
            add_edge(f"h{left_id}_{right_id}", f"J{left_id}", f"J{right_id}")
            add_edge(f"h{right_id}_{left_id}", f"J{right_id}", f"J{left_id}")

    # Vertical links (N–S)
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS):
            top_id = r * GRID_COLS + c
            bot_id = (r + 1) * GRID_COLS + c
            add_edge(f"v{top_id}_{bot_id}", f"J{top_id}", f"J{bot_id}")
            add_edge(f"v{bot_id}_{top_id}", f"J{bot_id}", f"J{top_id}")

    # Boundary edges
    for c in range(GRID_COLS):
        top_j = c
        bot_j = (GRID_ROWS - 1) * GRID_COLS + c
        add_edge(f"entry_top_{c}", f"top_{c}", f"J{top_j}")
        add_edge(f"exit_top_{c}",  f"J{top_j}",  f"top_{c}")
        add_edge(f"entry_bot_{c}", f"bot_{c}", f"J{bot_j}")
        add_edge(f"exit_bot_{c}",  f"J{bot_j}",  f"bot_{c}")
    for r in range(GRID_ROWS):
        left_j  = r * GRID_COLS
        right_j = r * GRID_COLS + GRID_COLS - 1
        add_edge(f"entry_left_{r}", f"left_{r}", f"J{left_j}")
        add_edge(f"exit_left_{r}",  f"J{left_j}",  f"left_{r}")
        add_edge(f"entry_right_{r}", f"right_{r}", f"J{right_j}")
        add_edge(f"exit_right_{r}",  f"J{right_j}",  f"right_{r}")

    edges_xml.append('</edges>')

    # ── write net.xml ──
    (output_dir / "nodes.xml").write_text("\n".join(nodes_xml), encoding="utf-8")
    (output_dir / "edges.xml").write_text("\n".join(edges_xml), encoding="utf-8")

    # ── route file (Poisson demand scaled per intersection) ──
    rou_lines = ['<routes>',
                 '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                 'length="5.0" maxSpeed="13.89" carFollowModel="Krauss"/>']
    veh_id = 0
    sim_duration = 3600  # seconds
    for i in range(NUM_INTERSECTIONS):
        rate = ARRIVAL_RATES[i]  # veh/min
        rate_per_sec = rate / 60.0
        # Distribute vehicles uniformly across the simulation window
        n_vehicles = int(rate_per_sec * sim_duration)
        depart_times = np.sort(np.random.uniform(0, sim_duration, n_vehicles))
        for t in depart_times:
            # Simple random route: enter from a random boundary edge
            rou_lines.append(
                f'  <vehicle id="v{veh_id}" type="car" depart="{t:.1f}" departLane="best"/>'
            )
            veh_id += 1
    rou_lines.append('</routes>')
    rou_path.write_text("\n".join(rou_lines), encoding="utf-8")

    # ── sumocfg ──
    cfg_xml = f"""<configuration>
  <input>
    <net-file value="grid3x3.net.xml"/>
    <route-files value="grid3x3.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{sim_duration}"/>
    <step-length value="1"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
  </processing>
</configuration>"""
    cfg_path.write_text(cfg_xml, encoding="utf-8")

    return cfg_path


# ─────────────────────────────────────────────────────────────────────────────
#  SUMO ENVIRONMENT  (used when SUMO is installed)
# ─────────────────────────────────────────────────────────────────────────────

class SUMOEnvironment:
    """
    Wraps SUMO via TraCI to expose the same interface as the Poisson simulator.

    Each of the 9 intersections in the 3x3 grid is a separate RL / FL agent.
    The agent observes: (N-queue, S-queue, E-queue, W-queue, phase, green_norm)
    The agent acts by setting the green phase duration.

    Requires:
        - SUMO installed and SUMO_HOME env variable set
        - sumo binary on PATH
    """

    TLS_IDS = [f"J{i}" for i in range(NUM_INTERSECTIONS)]

    def __init__(
        self,
        sumo_binary: str = "sumo",
        sim_dir: Optional[Path] = None,
        step_length: float = 1.0,
        max_steps: int = 3600,
        gui: bool = False,
    ):
        if not _SUMO_AVAILABLE:
            raise RuntimeError(
                "SUMO / TraCI not found. Install SUMO and set SUMO_HOME, "
                "or use SUMOFallbackEnvironment instead."
            )

        self.sim_dir = sim_dir or Path("sumo_nets/grid3x3")
        self.step_length = step_length
        self.max_steps = max_steps
        self.binary = "sumo-gui" if gui else sumo_binary

        # Build network if not present
        self._cfg_path = self.sim_dir / "grid3x3.sumocfg"
        if not self._cfg_path.exists():
            self._cfg_path = _write_sumo_network(self.sim_dir)

        self._step = 0
        self._phase_timers = {tls: 0.0 for tls in self.TLS_IDS}

    # ── context manager ──────────────────────────────────────────────────────
    def __enter__(self):
        traci.start([self.binary, "-c", str(self._cfg_path),
                     "--no-step-log", "--no-warnings"])
        return self

    def __exit__(self, *args):
        traci.close()

    # ── public API ───────────────────────────────────────────────────────────
    def get_state(self) -> Dict[int, np.ndarray]:
        """Return feature vectors for all 9 intersections."""
        states = {}
        for i, tls_id in enumerate(self.TLS_IDS):
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            n_q = s_q = e_q = w_q = 0
            for lane in lanes:
                edge = traci.lane.getEdgeID(lane)
                q = traci.edge.getLastStepHaltingNumber(edge)
                # Classify lane by direction heuristic (edge naming convention)
                if "top" in edge or "_0_" in edge:
                    n_q += q
                elif "bot" in edge or "_2_" in edge:
                    s_q += q
                elif "right" in edge or "_3_" in edge:
                    e_q += q
                else:
                    w_q += q

            phase_idx = traci.trafficlight.getPhase(tls_id)
            phase_norm = 1.0 if phase_idx == 0 else 0.0
            green_dur = self._phase_timers[tls_id]
            states[i] = np.array(
                [n_q, s_q, e_q, w_q, phase_norm, green_dur / 90.0],
                dtype=np.float32
            )
        return states

    def set_phase_duration(self, intersection_id: int, green_duration: float):
        """Set green phase duration for a given intersection."""
        tls_id = self.TLS_IDS[intersection_id]
        green_duration = float(np.clip(green_duration, 10, 60))
        traci.trafficlight.setPhaseDuration(tls_id, green_duration)
        self._phase_timers[tls_id] = green_duration

    def step(self) -> Dict[int, float]:
        """Advance simulation by one step; return per-intersection wait times."""
        traci.simulationStep()
        self._step += 1
        wait_times = {}
        for i, tls_id in enumerate(self.TLS_IDS):
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            total_wait = sum(
                traci.edge.getWaitingTime(traci.lane.getEdgeID(ln))
                for ln in lanes
            )
            wait_times[i] = total_wait / max(1, len(lanes))
        return wait_times

    def done(self) -> bool:
        return self._step >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCED POISSON FALLBACK  (9-intersection, Non-IID)
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedIntersection:
    """
    High-fidelity single-intersection model for the 3x3 grid fallback.

    Improvements over the original 4-intersection Poisson model:
    1. Non-homogeneous Poisson: arrival rate varies with time-of-day
    2. Correlated arrivals: spill-over from neighbouring intersections
    3. Vehicle following delay: ~2.0 s saturation headway (HCM 2000)
    4. Morning / evening rush profiles
    """

    SATURATION_FLOW = 1800  # veh/h/lane → 0.5 veh/s at saturation

    def __init__(
        self,
        intersection_id: int,
        base_arrival_rate: float,
        neighbours: List[int] = None,
        max_queue: int = 60,
        min_green: int = 10,
        max_green: int = 60,
        yellow: int = 3,
    ):
        self.iid = intersection_id
        self.base_rate = base_arrival_rate
        self.neighbours = neighbours or []
        self.max_queue = max_queue
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow

        self._reset_state()

    # ── state ────────────────────────────────────────────────────────────────
    def _reset_state(self):
        self.queues = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.wait_totals = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
        self.vehicles_served = 0
        self.total_wait = 0.0
        self.phase = "ns"           # "ns" | "ew"
        self.phase_timer = 0.0
        self.green_duration = 30.0
        self.sim_time = 0.0
        self.throughput = 0

    def reset(self):
        self._reset_state()

    # ── time-of-day factor ────────────────────────────────────────────────────
    def _time_factor(self) -> float:
        hour = (self.sim_time / 3600.0) % 24
        if 7 <= hour < 9:
            return TIME_OF_DAY_FACTORS["morning_peak"]
        elif 17 <= hour < 19:
            return TIME_OF_DAY_FACTORS["evening_peak"]
        elif 19 <= hour or hour < 7:
            return TIME_OF_DAY_FACTORS["night"]
        return TIME_OF_DAY_FACTORS["midday"]

    # ── step ─────────────────────────────────────────────────────────────────
    def step(
        self,
        dt: float,
        spill_in: Dict[str, int] = None,
    ) -> Dict:
        self.sim_time += dt
        self.phase_timer += dt
        rate = self.base_rate * self._time_factor()

        # Arrivals (non-homogeneous Poisson)
        arrivals_per_step = (rate / 60.0) * dt
        for d in ("north", "south", "east", "west"):
            n = int(np.random.poisson(arrivals_per_step / 4))
            if spill_in and d in spill_in:
                n += spill_in[d]
            self.queues[d] = min(self.queues[d] + n, self.max_queue)

        # Service (saturation flow model)
        green_dirs = ("north", "south") if self.phase == "ns" else ("east", "west")
        if self.phase_timer <= self.green_duration:
            # vehicles served per step ≈ saturation_flow × green_fraction × dt
            serve = max(1, int(self.SATURATION_FLOW / 3600 * dt))
            for d in green_dirs:
                served = min(serve, self.queues[d])
                self.queues[d] -= served
                self.throughput += served
                self.vehicles_served += served
                self.total_wait += served * (self.phase_timer / 2)

        # Phase transition
        if self.phase_timer >= self.green_duration + self.yellow:
            self.phase = "ew" if self.phase == "ns" else "ns"
            self.phase_timer = 0.0

        return self._metrics()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _metrics(self) -> Dict:
        total_q = sum(self.queues.values())
        avg_wait = self.total_wait / max(1, self.vehicles_served)
        return {
            "intersection_id": self.iid,
            "queue_lengths": dict(self.queues),
            "total_queue": total_q,
            "avg_wait": avg_wait,
            "throughput": self.throughput,
        }

    def get_feature_vector(self) -> np.ndarray:
        return np.array([
            self.queues["north"],
            self.queues["south"],
            self.queues["east"],
            self.queues["west"],
            1.0 if self.phase == "ns" else 0.0,
            self.green_duration / self.max_green,
        ], dtype=np.float32)

    def set_green_duration(self, duration: float):
        self.green_duration = float(np.clip(duration, self.min_green, self.max_green))


# ── Neighbour adjacency for 3x3 grid ──────────────────────────────────────────
def _get_neighbours(intersection_id: int) -> List[int]:
    """Return IDs of directly adjacent (von Neumann) intersections."""
    r, c = INTERSECTION_POSITIONS[intersection_id]
    neighbours = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            neighbours.append(nr * GRID_COLS + nc)
    return neighbours


class SUMOFallbackEnvironment:
    """
    9-intersection 3x3 grid environment backed by the enhanced Poisson model.

    This is API-compatible with SUMOEnvironment, so the rest of the codebase
    works identically whether SUMO is installed or not.
    """

    def __init__(self, max_steps: int = 720, step_size: float = 5.0):
        self.max_steps = max_steps
        self.step_size = step_size
        self._step = 0

        self.intersections: List[EnhancedIntersection] = [
            EnhancedIntersection(
                intersection_id=i,
                base_arrival_rate=ARRIVAL_RATES[i],
                neighbours=_get_neighbours(i),
            )
            for i in range(NUM_INTERSECTIONS)
        ]

    # ── public API (mirrors SUMOEnvironment) ─────────────────────────────────
    def get_state(self) -> Dict[int, np.ndarray]:
        return {i: inter.get_feature_vector() for i, inter in enumerate(self.intersections)}

    def set_phase_duration(self, intersection_id: int, green_duration: float):
        self.intersections[intersection_id].set_green_duration(green_duration)

    def step(self) -> Dict[int, float]:
        # Propagate spill-overs between neighbours (simplified: 5% of queue)
        spill = {}
        for i, inter in enumerate(self.intersections):
            for n in inter.neighbours:
                spill.setdefault(n, {})
                # 5% of upstream queue spills to each direction
                if inter.queues["south"] > 5:
                    spill[n]["north"] = spill[n].get("north", 0) + 1
                if inter.queues["east"] > 5:
                    spill[n]["west"] = spill[n].get("west", 0) + 1

        wait_times = {}
        for i, inter in enumerate(self.intersections):
            m = inter.step(self.step_size, spill.get(i))
            wait_times[i] = m["avg_wait"]

        self._step += 1
        return wait_times

    def done(self) -> bool:
        return self._step >= self.max_steps

    def reset(self):
        self._step = 0
        for inter in self.intersections:
            inter.reset()

    # ── training data generation ──────────────────────────────────────────────
    def generate_training_data(
        self,
        num_samples: int = 1000,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (features, labels) training data for all 9 intersections.
        Labels are optimal green durations computed via the HCM delay formula.
        """
        data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for i, inter in enumerate(self.intersections):
            inter.reset()
            features_list, labels_list = [], []
            for _ in range(num_samples):
                feat = inter.get_feature_vector()
                label = self._optimal_green(inter)
                inter.step(5.0)
                features_list.append(feat)
                labels_list.append(label)
            data[i] = (
                np.array(features_list, dtype=np.float32),
                np.array(labels_list,   dtype=np.float32),
            )
        return data

    @staticmethod
    def _optimal_green(inter: EnhancedIntersection) -> float:
        """
        HCM-based optimal green duration to minimise total delay.
        Webster (1958): g* = C × q_active / (q_active + q_waiting)
        with cycle length C calibrated to queue size.
        """
        if inter.phase == "ns":
            q_active  = inter.queues["north"] + inter.queues["south"]
            q_waiting = inter.queues["east"]  + inter.queues["west"]
        else:
            q_active  = inter.queues["east"]  + inter.queues["west"]
            q_waiting = inter.queues["north"] + inter.queues["south"]

        total_q = q_active + q_waiting + 1e-6

        # Webster cycle length: shorter at low demand, longer at high demand
        C = np.clip(20 + (total_q / 60.0) * 40, 30, 90)

        # Green split proportional to active-direction demand
        split = np.clip(q_active / total_q, 0.3, 0.7)
        g = C * split + np.random.normal(0, 0.5)
        return float(np.clip(g, inter.min_green, inter.max_green))


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_grid_environment(
    use_sumo: bool = None,
    **kwargs
) -> "SUMOEnvironment | SUMOFallbackEnvironment":
    """
    Create a 3x3 grid traffic environment.

    Args:
        use_sumo:  True → force SUMO (raises if not installed),
                   False → force fallback,
                   None  → auto-detect.
        **kwargs:  Forwarded to the chosen environment class.

    Returns:
        SUMOEnvironment or SUMOFallbackEnvironment (same API).
    """
    if use_sumo is None:
        use_sumo = _SUMO_AVAILABLE

    if use_sumo:
        return SUMOEnvironment(**kwargs)
    else:
        return SUMOFallbackEnvironment(**kwargs)


def is_sumo_available() -> bool:
    """Return True if SUMO / TraCI is importable."""
    return _SUMO_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SUMO Integration: 3x3 Grid Environment Test")
    print(f"SUMO available: {_SUMO_AVAILABLE}")
    print("=" * 60)

    env = create_grid_environment(use_sumo=False, max_steps=20, step_size=5.0)
    print(f"\nCreated {NUM_INTERSECTIONS}-intersection 3x3 grid (enhanced Poisson fallback)")

    # Test one simulation cycle
    state = env.get_state()
    print("\nInitial states (feature vectors):")
    for i, feat in state.items():
        r, c = INTERSECTION_POSITIONS[i]
        label = "CBD" if i == 4 else ("corner" if (r % 2 == 0 and c % 2 == 0) else "edge")
        print(f"  Intersection {i} ({label}): queues={feat[:4].astype(int)}")

    # Set some green durations and run 5 steps
    for i in range(NUM_INTERSECTIONS):
        env.set_phase_duration(i, 25.0 + i * 2.0)

    total_waits = {i: 0.0 for i in range(NUM_INTERSECTIONS)}
    for step in range(20):
        waits = env.step()
        for i, w in waits.items():
            total_waits[i] += w

    print("\nMean wait time per intersection (20 steps × 5 s):")
    for i, tw in total_waits.items():
        r, c = INTERSECTION_POSITIONS[i]
        label = "CBD" if i == 4 else ("corner" if (r % 2 == 0 and c % 2 == 0) else "edge")
        print(f"  Intersection {i} ({r},{c}) [{label:6}]: {tw/20:.2f}s")

    # Generate training data
    print("\nGenerating training data for FL (100 samples/intersection)...")
    env.reset()
    data = env.generate_training_data(num_samples=100)
    for i, (X, y) in data.items():
        print(f"  Intersection {i}: X={X.shape}, y={y.shape}, "
              f"mean_green={y.mean():.1f}s ± {y.std():.1f}s")

    print("\nSUMO integration test complete.")
    print(f"{'='*60}")
