"""Tests for EulerMaruyamaSolver."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.simulator.state_transform import Log1pTransform

from .conftest import make_fake_context

N_SPECIES = 2
T_SPAN = torch.tensor([0.0, 0.5, 1.0])
DT = 0.5


def _make_model(config: SDEConfig) -> NeuralSDE:
    return NeuralSDE(config, N_SPECIES).eval()


def _make_solver(clip: bool = True, transform=None) -> EulerMaruyamaSolver:
    return EulerMaruyamaSolver(SolverConfig(clip_state=clip), state_transform=transform)


# ── Basic output contracts ────────────────────────────────────────────────────


def test_output_type(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.zeros(N_SPECIES)
    result = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert isinstance(result, Trajectory)


def test_output_trajectory_length(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.zeros(N_SPECIES)
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert len(traj.states) == len(T_SPAN)


def test_output_shape(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.zeros(N_SPECIES)
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert traj.states.shape == (len(T_SPAN), N_SPECIES)


def test_stochastic_different_runs(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.tensor([5.0, 5.0])
    traj1 = solver.solve(model, initial, ctx, T_SPAN, DT)
    traj2 = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert not torch.allclose(traj1.states, traj2.states)


# ── Clip state ────────────────────────────────────────────────────────────────


class _NegativeDriftSDE(NeuralSDE):
    """NeuralSDE subclass whose drift always returns -100."""

    def drift(self, t, state, crn_context, protocol_embedding=None):
        return torch.full_like(state, -100.0)


def test_clip_state(small_sde_config: SDEConfig) -> None:
    solver = _make_solver(clip=True)
    ctx = make_fake_context(small_sde_config.d_model)
    model = _NegativeDriftSDE(small_sde_config, N_SPECIES).eval()
    initial = torch.tensor([1.0, 1.0])
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert (traj.states >= 0).all()


# ── State transform ───────────────────────────────────────────────────────────


def test_state_transform_applied(small_sde_config: SDEConfig) -> None:
    transform = Log1pTransform()
    solver = _make_solver(clip=False, transform=transform)
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.tensor([5.0, 10.0])
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert traj.states.shape == (len(T_SPAN), N_SPECIES)
    assert traj.states.isfinite().all()


# ── Error on non-stochastic model ─────────────────────────────────────────────


def test_requires_stochastic_model(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    drift_only = NeuralDrift(small_sde_config, N_SPECIES).eval()
    initial = torch.tensor([1.0, 1.0])
    with pytest.raises(AttributeError):
        solver.solve(drift_only, initial, ctx, T_SPAN, DT)


# ── clip_state property ───────────────────────────────────────────────────────


def test_clip_state_property_true() -> None:
    assert _make_solver(clip=True).clip_state is True


def test_clip_state_property_false() -> None:
    assert _make_solver(clip=False).clip_state is False
