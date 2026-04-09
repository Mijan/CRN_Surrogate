"""Tests for MeanMatchingLoss, VarianceMatchingLoss, TransitionNLL, CombinedTrajectoryLoss."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.measurement.config import MeasurementConfig
from crn_surrogate.measurement.direct import DirectObservation
from crn_surrogate.simulator.neural_sde import NeuralSDE
from crn_surrogate.training.losses import (
    CombinedTrajectoryLoss,
    MeanMatchingLoss,
    TransitionNLL,
    VarianceMatchingLoss,
)


@pytest.fixture
def nll_setup():
    """Build a small NeuralSDE, CRNContext, and fake trajectory for NLL tests."""
    config = SDEConfig(d_model=16, d_hidden=32, n_noise_channels=2, n_hidden_layers=1)
    sde = NeuralSDE(config, n_species=2)
    context = CRNContext(
        species_embeddings=torch.randn(2, 16),
        reaction_embeddings=torch.randn(2, 16),
        context_vector=torch.randn(32),
    )
    # (M=2, T=5, n_species=2) fake trajectory
    trajectory = torch.rand(2, 5, 2) * 10 + 1  # positive values
    times = torch.linspace(0, 1, 5)
    return sde, context, trajectory, times


# ── MeanMatchingLoss ──────────────────────────────────────────────────────────


def test_perfect_prediction_zero_loss() -> None:
    states = torch.rand(3, 20, 2)
    loss = MeanMatchingLoss().compute(states, states)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_nonzero_loss_on_mismatch() -> None:
    pred = torch.ones(3, 20, 2) * 5.0
    true = torch.zeros(3, 20, 2)
    loss = MeanMatchingLoss().compute(pred, true)
    assert loss.item() > 0.0


def test_rejects_2d_input() -> None:
    with pytest.raises(ValueError):
        MeanMatchingLoss().compute(torch.rand(20, 2), torch.rand(20, 2))


def test_mask_excludes_species() -> None:
    # pred and true differ only on species 0; species 1 is identical
    pred = torch.zeros(3, 20, 2)
    true = torch.zeros(3, 20, 2)
    pred[:, :, 0] = 5.0  # species 0 differs
    mask = torch.tensor([False, True])
    loss = MeanMatchingLoss().compute(pred, true, mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── VarianceMatchingLoss ──────────────────────────────────────────────────────


def test_perfect_variance_zero_loss() -> None:
    torch.manual_seed(0)
    states = torch.rand(5, 20, 2) * 10
    loss = VarianceMatchingLoss().compute(states, states)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_rejects_single_sample() -> None:
    with pytest.raises(ValueError):
        VarianceMatchingLoss().compute(torch.rand(1, 20, 2), torch.rand(3, 20, 2))
    with pytest.raises(ValueError):
        VarianceMatchingLoss().compute(torch.rand(3, 20, 2), torch.rand(1, 20, 2))


def test_variance_mask_excludes_species() -> None:
    # pred and true agree on species 1 variance but differ on species 0
    torch.manual_seed(1)
    pred = torch.rand(4, 20, 2) * 10
    true = torch.rand(4, 20, 2) * 10
    # Make species 1 identical
    pred[:, :, 1] = true[:, :, 1]
    mask = torch.tensor([False, True])
    loss = VarianceMatchingLoss().compute(pred, true, mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── TransitionNLL ─────────────────────────────────────────────────────────────


def test_nll_returns_scalar(nll_setup) -> None:
    sde, ctx, traj, times = nll_setup
    loss_fn = TransitionNLL()
    result = loss_fn.compute(sde, ctx, traj, times, dt=0.25)
    assert result.dim() == 0


def test_nll_finite(nll_setup) -> None:
    sde, ctx, traj, times = nll_setup
    loss_fn = TransitionNLL()
    result = loss_fn.compute(sde, ctx, traj, times, dt=0.25)
    assert torch.isfinite(result)


def test_nll_gradients_flow(nll_setup) -> None:
    sde, ctx, traj, times = nll_setup
    loss_fn = TransitionNLL()
    result = loss_fn.compute(sde, ctx, traj, times, dt=0.25)
    result.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in sde.parameters()
    )
    assert has_grad


def test_nll_rejects_single_timestep(nll_setup) -> None:
    sde, ctx, _, _ = nll_setup
    single_t = torch.rand(2, 1, 2)
    times = torch.tensor([0.0])
    with pytest.raises(ValueError):
        TransitionNLL().compute(sde, ctx, single_t, times, dt=0.1)


def test_nll_with_measurement_model(nll_setup) -> None:
    sde, ctx, traj, times = nll_setup
    meas = DirectObservation.from_config(MeasurementConfig(), n_species=2)

    loss_no_meas = TransitionNLL().compute(sde, ctx, traj, times, dt=0.25).item()
    loss_with_meas = (
        TransitionNLL(measurement_model=meas)
        .compute(sde, ctx, traj, times, dt=0.25)
        .item()
    )
    assert torch.isfinite(torch.tensor(loss_with_meas))
    assert torch.isfinite(torch.tensor(loss_no_meas))
    assert loss_no_meas != pytest.approx(loss_with_meas, rel=1e-3)


def test_nll_with_mask(nll_setup) -> None:
    sde, ctx, traj, times = nll_setup
    mask = torch.tensor([True, False])
    result = TransitionNLL().compute(sde, ctx, traj, times, dt=0.25, mask=mask)
    assert result.dim() == 0
    assert torch.isfinite(result)


def test_nll_2d_trajectory_auto_unsqueeze(nll_setup) -> None:
    sde, ctx, _, times = nll_setup
    traj_2d = torch.rand(5, 2) * 10 + 1  # (T, n_species) — 2D
    result = TransitionNLL().compute(sde, ctx, traj_2d, times, dt=0.25)
    assert result.dim() == 0
    assert torch.isfinite(result)


# ── CombinedTrajectoryLoss ────────────────────────────────────────────────────


def test_combined_loss_returns_scalar() -> None:
    pred = torch.rand(3, 20, 2) * 10
    true = torch.rand(3, 20, 2) * 10
    result = CombinedTrajectoryLoss().compute(pred, true)
    assert result.dim() == 0


def test_combined_loss_default_components() -> None:
    # Default = MeanMatchingLoss + VarianceMatchingLoss; requires K >= 2, M >= 2
    pred = torch.rand(3, 20, 2) * 10
    true = torch.rand(3, 20, 2) * 10
    result = CombinedTrajectoryLoss().compute(pred, true)
    assert torch.isfinite(result)
    assert result.item() >= 0.0
