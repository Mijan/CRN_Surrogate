"""Benchmark standard vs Numba-accelerated SSA.

Usage:
    python experiments/scripts/benchmark_ssa.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.generation.mass_action_generator import (
    MassActionCRNGenerator,
    MassActionGeneratorConfig,
)
from crn_surrogate.data.generation.mass_action_topology import birth_death_topology
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory

try:
    from crn_surrogate.simulation.fast_ssa import (
        NUMBA_AVAILABLE,
        FastMassActionSSA,
    )
except ImportError:
    NUMBA_AVAILABLE = False


def _benchmark_standard(crn, init, t_max, time_grid, n_traj, n_repeats=5):
    ssa = GillespieSSA()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        trajs = ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init,
            t_max=t_max,
            n_trajectories=n_traj,
        )
        Trajectory.stack_on_grid(trajs, time_grid)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _benchmark_fast(stoich, reactant, rates, init, t_max, time_grid, n_traj, n_repeats=5):
    fast = FastMassActionSSA()
    # Warm up (excludes JIT compilation from timing)
    fast.simulate_batch(stoich, reactant, rates, init, t_max, time_grid, 2)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fast.simulate_batch(stoich, reactant, rates, init, t_max, time_grid, n_traj)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    """Compare standard and fast SSA across several CRN topologies."""
    torch.manual_seed(42)
    t_max = 20.0
    n_traj = 32
    time_grid = torch.linspace(0.0, t_max, 50)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())

    print(f"{'CRN':>20} {'n_sp':>5} {'n_rx':>5} {'Standard':>10} {'Fast':>10} {'Speedup':>8}")
    print("-" * 65)

    # Named topology: birth-death
    topo = birth_death_topology()
    crn = topo.to_crn([2.0, 0.5])
    init = torch.tensor([0.0])
    t_std = _benchmark_standard(crn, init, t_max, time_grid, n_traj)

    if NUMBA_AVAILABLE:
        stoich = topo.net_stoichiometry.numpy().astype(np.float64)
        reactant = topo.reactant_matrix.numpy().astype(np.float64)
        rates = np.array([2.0, 0.5])
        t_fast = _benchmark_fast(
            stoich, reactant, rates,
            init.numpy().astype(np.float64),
            t_max,
            time_grid.numpy().astype(np.float64),
            n_traj,
        )
        speedup = f"{t_std / t_fast:.1f}x"
        print(f"{'birth-death':>20} {1:>5} {2:>5} {t_std:>10.4f}s {t_fast:>10.4f}s {speedup:>8}")
    else:
        print(f"{'birth-death':>20} {1:>5} {2:>5} {t_std:>10.4f}s {'N/A':>10} {'N/A':>8}")

    # Random topologies
    for i in range(5):
        crn = gen.sample()
        init = gen.sample_initial_state(crn)
        t_std = _benchmark_standard(crn, init, t_max, time_grid, n_traj)

        if NUMBA_AVAILABLE:
            try:
                arrays = FastMassActionSSA.extract_topology_arrays(crn)
                t_fast = _benchmark_fast(
                    arrays["stoichiometry"],
                    arrays["reactant_matrix"],
                    arrays["rate_constants"],
                    init.numpy().astype(np.float64),
                    t_max,
                    time_grid.numpy().astype(np.float64),
                    n_traj,
                )
                speedup = f"{t_std / t_fast:.1f}x"
                t_fast_str = f"{t_fast:.4f}s"
            except ValueError:
                t_fast_str = "N/A"
                speedup = "N/A"
            print(
                f"{'random_' + str(i):>20} {crn.n_species:>5} {crn.n_reactions:>5} "
                f"{t_std:>10.4f}s {t_fast_str:>10} {speedup:>8}"
            )
        else:
            print(
                f"{'random_' + str(i):>20} {crn.n_species:>5} {crn.n_reactions:>5} "
                f"{t_std:>10.4f}s {'N/A':>10} {'N/A':>8}"
            )


if __name__ == "__main__":
    main()
