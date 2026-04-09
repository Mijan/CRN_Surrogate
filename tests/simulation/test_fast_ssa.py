"""Tests for FastMassActionSSA.

All tests require numba and are skipped when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crn_surrogate.simulation.fast_ssa import NUMBA_AVAILABLE, FastMassActionSSA
from crn_surrogate.simulation.gillespie import GillespieSSA

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")

# Birth-death arrays: ∅ -> X (rate 2.0), X -> ∅ (rate 0.5*X)
_STOICH = np.array([[1.0], [-1.0]], dtype=np.float64)
_REACTANT = np.array([[0.0], [1.0]], dtype=np.float64)
_RATES = np.array([2.0, 0.5], dtype=np.float64)
_INIT = np.array([0.0], dtype=np.float64)
_GRID = np.linspace(0.0, 50.0, 100)
_T_MAX = 50.0


def _fast_ssa() -> FastMassActionSSA:
    return FastMassActionSSA()


def test_output_shape():
    ssa = _fast_ssa()
    result = ssa.simulate_batch(
        stoichiometry=_STOICH,
        reactant_matrix=_REACTANT,
        rate_constants=_RATES,
        initial_state=_INIT,
        t_max=_T_MAX,
        time_grid=_GRID,
        n_trajectories=10,
    )
    assert result.shape == (10, 100, 1)


def test_output_dtype():
    ssa = _fast_ssa()
    result = ssa.simulate_batch(
        stoichiometry=_STOICH,
        reactant_matrix=_REACTANT,
        rate_constants=_RATES,
        initial_state=_INIT,
        t_max=_T_MAX,
        time_grid=_GRID,
        n_trajectories=5,
    )
    assert result.dtype == torch.float32


def test_birth_death_mean():
    k_birth, k_death = 2.0, 0.5
    rates = np.array([k_birth, k_death], dtype=np.float64)
    ssa = _fast_ssa()
    torch.manual_seed(0)
    result = ssa.simulate_batch(
        stoichiometry=_STOICH,
        reactant_matrix=_REACTANT,
        rate_constants=rates,
        initial_state=_INIT,
        t_max=_T_MAX,
        time_grid=_GRID,
        n_trajectories=500,
    )
    # result: (500, 100, 1) — take final time step
    final_mean = result[:, -1, 0].mean().item()
    expected = k_birth / k_death  # 4.0
    assert abs(final_mean - expected) < 1.5


def test_standard_and_fast_agree():
    """Distributional means should match between GillespieSSA and FastMassActionSSA."""
    k_birth, k_death = 2.0, 0.5
    x0 = torch.tensor([0.0])
    t_max = 30.0
    n_trajs = 300
    time_grid_np = np.linspace(0.0, t_max, 50)
    time_grid = torch.tensor(time_grid_np, dtype=torch.float32)

    from tests.simulation.conftest import _birth_death_crn

    crn = _birth_death_crn(k_birth=k_birth, k_death=k_death)

    # GillespieSSA ensemble mean at final time
    ssa = GillespieSSA()
    torch.manual_seed(1)
    trajs = ssa.simulate_batch(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=x0.clone(),
        t_max=t_max,
        n_trajectories=n_trajs,
    )
    from crn_surrogate.simulation.trajectory import Trajectory

    stacked = Trajectory.stack_on_grid(trajs, time_grid)
    ssa_mean = stacked[:, -1, 0].mean().item()

    # FastMassActionSSA ensemble mean at final time
    fast = _fast_ssa()
    torch.manual_seed(1)
    result = fast.simulate_batch(
        stoichiometry=np.array([[1.0], [-1.0]]),
        reactant_matrix=np.array([[0.0], [1.0]]),
        rate_constants=np.array([k_birth, k_death]),
        initial_state=np.array([0.0]),
        t_max=t_max,
        time_grid=time_grid_np,
        n_trajectories=n_trajs,
    )
    fast_mean = result[:, -1, 0].mean().item()

    expected = k_birth / k_death  # 4.0
    assert abs(ssa_mean - expected) < 2.0
    assert abs(fast_mean - expected) < 2.0


def test_extract_topology_arrays_shape():
    from crn_surrogate.data.generation.reference_crns import birth_death

    crn = birth_death(k_birth=2.0, k_death=0.5)
    arrays = FastMassActionSSA.extract_topology_arrays(crn)

    assert arrays["stoichiometry"].shape == (2, 1)
    assert arrays["reactant_matrix"].shape == (2, 1)
    assert arrays["rate_constants"].shape == (2,)
