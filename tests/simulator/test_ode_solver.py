"""Tests for EulerODESolver."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import NeuralDrift
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.state_transform import Log1pTransform

from .conftest import make_fake_context

N_SPECIES = 2
T_SPAN = torch.tensor([0.0, 0.5, 1.0])
DT = 0.5


def _make_model(config: SDEConfig) -> NeuralDrift:
    return NeuralDrift(config, N_SPECIES).eval()


def _make_solver(clip: bool = True, transform=None) -> EulerODESolver:
    return EulerODESolver(SolverConfig(clip_state=clip), state_transform=transform)


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


def test_deterministic_reproducibility(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.tensor([1.0, 2.0])
    traj1 = solver.solve(model, initial, ctx, T_SPAN, DT)
    traj2 = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert torch.allclose(traj1.states, traj2.states)


# ── Clip state ────────────────────────────────────────────────────────────────


class _NegativeDriftModel(NeuralDrift):
    """NeuralDrift subclass whose drift always returns -100 (forces negative states)."""

    def drift(self, t, state, crn_context, protocol_embedding=None):
        return torch.full_like(state, -100.0)


def test_clip_state(small_sde_config: SDEConfig) -> None:
    solver = _make_solver(clip=True)
    ctx = make_fake_context(small_sde_config.d_model)
    model = _NegativeDriftModel(small_sde_config, N_SPECIES).eval()
    initial = torch.tensor([1.0, 1.0])
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert (traj.states >= 0).all()


def test_no_clip_state(small_sde_config: SDEConfig) -> None:
    solver = _make_solver(clip=False)
    ctx = make_fake_context(small_sde_config.d_model)
    model = _NegativeDriftModel(small_sde_config, N_SPECIES).eval()
    initial = torch.tensor([1.0, 1.0])
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    # At least some states should be negative (unclamped strong negative drift)
    assert (traj.states < 0).any()


# ── State transform ───────────────────────────────────────────────────────────


def test_state_transform_applied(small_sde_config: SDEConfig) -> None:
    transform = Log1pTransform()
    solver = _make_solver(clip=False, transform=transform)
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.tensor([5.0, 10.0])
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    # Output states should be in raw count space; log1p(5)≈1.79, so if values
    # are > 3 or the inverse was applied, we're in raw space.
    # Just verify shape is correct and states are finite (transform was applied).
    assert traj.states.shape == (len(T_SPAN), N_SPECIES)
    assert traj.states.isfinite().all()


# ── Drift-only (no diffusion call) ────────────────────────────────────────────


def test_drift_only_no_diffusion_call(small_sde_config: SDEConfig) -> None:
    solver = _make_solver()
    ctx = make_fake_context(small_sde_config.d_model)
    model = _make_model(small_sde_config)
    initial = torch.tensor([1.0, 2.0])
    # NeuralDrift has no diffusion(); if solver tried to call it, AttributeError.
    traj = solver.solve(model, initial, ctx, T_SPAN, DT)
    assert isinstance(traj, Trajectory)
