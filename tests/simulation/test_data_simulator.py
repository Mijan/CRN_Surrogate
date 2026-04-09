"""Tests for SSASimulator, ODESimulator, and FastSSASimulator."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.simulation.data_simulator import (
    DataSimulator,
    FastSSASimulator,
    ODESimulator,
    SSASimulator,
)
from crn_surrogate.simulation.fast_ssa import NUMBA_AVAILABLE
from crn_surrogate.simulation.mass_action_ode import MassActionODE
from tests.simulation.conftest import (
    _birth_death_crn,
    _decay_crn,
    _two_species_crn,
)

_T_MAX = 10.0
_N_POINTS = 30
_TIME_GRID = torch.linspace(0.0, _T_MAX, _N_POINTS)


def test_ssa_simulator_output_shape():
    crn = _birth_death_crn()
    sim = SSASimulator(timeout=0)
    result = sim.simulate(crn, torch.tensor([4.0]), _T_MAX, 5, _TIME_GRID)
    assert result is not None
    assert result.shape == (5, _N_POINTS, 1)


def test_ssa_simulator_no_timeout_on_fast_crn():
    # Pure decay from x=0: no reactions fire, completes instantly
    crn = _decay_crn(k=1.0)
    sim = SSASimulator(timeout=30)
    result = sim.simulate(crn, torch.tensor([0.0]), _T_MAX, 3, _TIME_GRID)
    assert result is not None
    assert isinstance(result, torch.Tensor)


def test_ode_simulator_output_shape():
    crn = _two_species_crn()
    sim = ODESimulator()
    result = sim.simulate(crn, torch.tensor([10.0, 5.0]), _T_MAX, 1, _TIME_GRID)
    assert result is not None
    assert result.shape == (1, _N_POINTS, 2)


def test_ode_simulator_ignores_n_trajectories():
    crn = _decay_crn()
    sim = ODESimulator()
    result = sim.simulate(crn, torch.tensor([5.0]), _T_MAX, 10, _TIME_GRID)
    assert result is not None
    # ODE always produces exactly 1 trajectory
    assert result.shape[0] == 1


def test_ode_simulator_returns_none_on_blowup():
    crn = _birth_death_crn(k_birth=1e6, k_death=0.0)
    sim = ODESimulator(n_substeps=5, blowup_threshold=100.0)
    result = sim.simulate(crn, torch.tensor([1.0]), _T_MAX, 1, _TIME_GRID)
    assert result is None


def test_ode_simulator_matches_mass_action_ode():
    crn = _decay_crn(k=0.2)
    x0 = torch.tensor([10.0])
    time_grid = torch.linspace(0.0, 5.0, 20)

    sim = ODESimulator(n_substeps=10)
    ode = MassActionODE(n_substeps=10)

    sim_result = sim.simulate(crn, x0, 5.0, 1, time_grid)
    ode_result = ode.integrate(crn, x0, time_grid)

    assert sim_result is not None and ode_result is not None
    assert torch.allclose(sim_result[0], ode_result)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_fast_ssa_simulator_output_shape():
    crn = _birth_death_crn()
    sim = FastSSASimulator(timeout=0)
    result = sim.simulate(crn, torch.tensor([4.0]), _T_MAX, 5, _TIME_GRID)
    assert result is not None
    assert result.shape == (5, _N_POINTS, 1)


def test_all_simulators_are_data_simulator():
    assert isinstance(SSASimulator(), DataSimulator)
    assert isinstance(ODESimulator(), DataSimulator)
    if NUMBA_AVAILABLE:
        assert isinstance(FastSSASimulator(), DataSimulator)
