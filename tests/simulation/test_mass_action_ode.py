"""Tests for MassActionODE."""

from __future__ import annotations

import math

import pytest
import torch

from crn_surrogate.simulation.mass_action_ode import NUMBA_AVAILABLE, MassActionODE
from tests.simulation.conftest import _birth_death_crn, _decay_crn, _two_species_crn


def test_exponential_decay():
    k = 0.1
    x0 = 100.0
    crn = _decay_crn(k=k)
    ode = MassActionODE(n_substeps=50)
    time_grid = torch.linspace(0.0, 10.0, 50)
    result = ode.integrate(crn, torch.tensor([x0]), time_grid)
    assert result is not None
    analytical = torch.tensor([x0 * math.exp(-k * t.item()) for t in time_grid])
    assert torch.allclose(result[:, 0], analytical, atol=1.0)


def test_birth_death_steady_state():
    k_birth, k_death = 2.0, 0.5
    crn = _birth_death_crn(k_birth=k_birth, k_death=k_death)
    ode = MassActionODE(n_substeps=20)
    time_grid = torch.linspace(0.0, 50.0, 100)
    result = ode.integrate(crn, torch.tensor([0.0]), time_grid)
    assert result is not None
    steady_state = k_birth / k_death  # 4.0
    assert abs(result[-1, 0].item() - steady_state) < 0.5


def test_output_shape():
    crn = _two_species_crn()
    ode = MassActionODE()
    time_grid = torch.linspace(0.0, 5.0, 20)
    result = ode.integrate(crn, torch.tensor([10.0, 5.0]), time_grid)
    assert result is not None
    assert result.shape == (20, 2)


def test_output_dtype():
    crn = _decay_crn()
    ode = MassActionODE()
    time_grid = torch.linspace(0.0, 5.0, 10)
    result = ode.integrate(crn, torch.tensor([10.0]), time_grid)
    assert result is not None
    assert result.dtype == torch.float32


def test_blowup_returns_none():
    # Very fast birth with essentially no death blows up
    crn = _birth_death_crn(k_birth=1e6, k_death=0.0)
    ode = MassActionODE(n_substeps=5, blowup_threshold=100.0)
    time_grid = torch.linspace(0.0, 10.0, 20)
    result = ode.integrate(crn, torch.tensor([1.0]), time_grid)
    assert result is None


def test_zero_initial_state():
    crn = _decay_crn(k=1.0)
    ode = MassActionODE()
    time_grid = torch.linspace(0.0, 5.0, 20)
    result = ode.integrate(crn, torch.tensor([0.0]), time_grid)
    assert result is not None
    assert torch.allclose(result, torch.zeros(20, 1))


def test_deterministic_reproducibility():
    crn = _decay_crn()
    ode = MassActionODE()
    time_grid = torch.linspace(0.0, 5.0, 20)
    x0 = torch.tensor([10.0])
    result1 = ode.integrate(crn, x0, time_grid)
    result2 = ode.integrate(crn, x0, time_grid)
    assert result1 is not None and result2 is not None
    assert torch.allclose(result1, result2)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_and_python_agree():
    # _integrate_fast requires a real CRN with mass-action reactions
    from crn_surrogate.data.generation.reference_crns import birth_death

    crn = birth_death(k_birth=2.0, k_death=0.5)
    x0 = torch.tensor([8.0])
    time_grid = torch.linspace(0.0, 10.0, 30)

    ode = MassActionODE(n_substeps=20)
    result_fast = ode._integrate_fast(crn, x0, time_grid)
    result_python = ode._integrate_python(crn, x0, time_grid)

    from crn_surrogate.simulation.mass_action_ode import _NOT_MASS_ACTION

    assert result_fast is not _NOT_MASS_ACTION
    assert result_fast is not None and result_python is not None
    assert torch.allclose(result_fast, result_python, atol=0.01)
