"""Tests for MeanMatchingLoss, VarianceMatchingLoss, StepLoss impls, CombinedRolloutLoss."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.measurement.config import MeasurementConfig
from crn_surrogate.measurement.direct import DirectObservation
from crn_surrogate.training.losses import (
    CombinedRolloutLoss,
    CombinedTrajectoryLoss,
    MeanMatchingLoss,
    MSEStepLoss,
    NLLStepLoss,
    RolloutLoss,
    VarianceMatchingLoss,
)


def _make_meas(n_species: int = 2) -> DirectObservation:
    return DirectObservation.from_config(MeasurementConfig(), n_species=n_species)


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
    torch.manual_seed(1)
    pred = torch.rand(4, 20, 2) * 10
    true = torch.rand(4, 20, 2) * 10
    pred[:, :, 1] = true[:, :, 1]
    mask = torch.tensor([False, True])
    loss = VarianceMatchingLoss().compute(pred, true, mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── MSEStepLoss ───────────────────────────────────────────────────────────────


def test_mse_output_shape() -> None:
    N, S = 100, 3
    out = MSEStepLoss().compute(torch.rand(N, S), torch.rand(N, S), torch.rand(N, S))
    assert out.shape == (N, S)


def test_mse_zero_on_perfect_prediction() -> None:
    N, S = 50, 4
    y = torch.rand(N, S)
    out = MSEStepLoss().compute(y, y, torch.rand(N, S))
    assert out.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_mse_positive_on_mismatch() -> None:
    y_next = torch.rand(20, 3) + 1.0
    mu = torch.zeros(20, 3)
    out = MSEStepLoss().compute(y_next, mu, torch.zeros(20, 3))
    assert (out > 0).all()


def test_mse_ignores_variance() -> None:
    y_next = torch.rand(20, 3)
    mu = torch.rand(20, 3)
    out1 = MSEStepLoss().compute(y_next, mu, torch.ones(20, 3))
    out2 = MSEStepLoss().compute(y_next, mu, torch.ones(20, 3) * 100.0)
    torch.testing.assert_close(out1, out2)


def test_mse_parameters_empty() -> None:
    assert MSEStepLoss().parameters() == []


def test_mse_state_dict_empty() -> None:
    assert MSEStepLoss().state_dict() == {}


def test_mse_extra_metrics_empty() -> None:
    assert MSEStepLoss().extra_metrics() == {}


# ── NLLStepLoss ───────────────────────────────────────────────────────────────


def test_nll_output_shape() -> None:
    N, S = 100, 3
    y_next = torch.rand(N, S) * 10 + 1
    mu = torch.rand(N, S) * 10 + 1
    variance = torch.rand(N, S) * 0.1 + 0.01
    out = NLLStepLoss().compute(y_next, mu, variance)
    assert out.shape == (N, S)


def test_nll_finite() -> None:
    N, S = 50, 2
    y_next = torch.rand(N, S) * 10 + 1
    mu = torch.rand(N, S) * 10 + 1
    variance = torch.rand(N, S) * 0.5 + 0.1
    out = NLLStepLoss(min_variance=1e-2).compute(y_next, mu, variance)
    assert torch.isfinite(out).all()


def test_nll_with_measurement_model() -> None:
    N, S = 30, 2
    y_next = torch.rand(N, S) * 10 + 1
    mu = torch.rand(N, S) * 10 + 1
    variance = torch.rand(N, S) * 0.1 + 0.01

    out_no_meas = NLLStepLoss().compute(y_next, mu, variance)
    out_with_meas = NLLStepLoss(measurement_model=_make_meas(S)).compute(
        y_next, mu, variance
    )

    assert out_no_meas.shape == (N, S)
    assert out_with_meas.shape == (N, S)
    assert not torch.allclose(out_no_meas, out_with_meas)


def test_nll_parameters_with_measurement() -> None:
    meas = _make_meas(n_species=2)
    loss = NLLStepLoss(measurement_model=meas)
    params = loss.parameters()
    assert len(params) > 0


def test_nll_parameters_without_measurement() -> None:
    assert NLLStepLoss(None).parameters() == []


def test_nll_state_dict_round_trip() -> None:
    meas = _make_meas(n_species=2)
    loss = NLLStepLoss(measurement_model=meas)

    state = loss.state_dict()

    # Corrupt eps in-place so we can verify restoration
    with torch.no_grad():
        meas.eps.fill_(999.0)

    loss.load_state_dict(state)
    assert not torch.allclose(meas.eps, torch.tensor(999.0))


def test_nll_extra_metrics_has_obs_eps() -> None:
    meas = _make_meas(n_species=2)
    loss = NLLStepLoss(measurement_model=meas)
    metrics = loss.extra_metrics()
    assert "obs_eps" in metrics


def test_nll_extra_metrics_empty_without_measurement() -> None:
    assert NLLStepLoss(None).extra_metrics() == {}


def test_nll_gradient_flows_through_measurement() -> None:
    N, S = 20, 2
    meas = _make_meas(n_species=S)
    y_next = torch.rand(N, S) * 10 + 1
    mu = torch.rand(N, S) * 10 + 1
    variance = torch.rand(N, S) * 0.1 + 0.01

    loss = NLLStepLoss(measurement_model=meas).compute(y_next, mu, variance).sum()
    loss.backward()

    meas_param = next(meas.parameters())
    assert meas_param.grad is not None
    assert meas_param.grad.abs().sum().item() > 0


def test_mse_vs_nll_differ() -> None:
    N, S = 40, 3
    y_next = torch.rand(N, S) * 5 + 1
    mu = torch.rand(N, S) * 5 + 1
    variance = torch.rand(N, S) * 0.5 + 0.1

    mse_out = MSEStepLoss().compute(y_next, mu, variance)
    nll_out = NLLStepLoss().compute(y_next, mu, variance)

    assert not torch.allclose(mse_out, nll_out)


# ── CombinedRolloutLoss (alias: CombinedTrajectoryLoss) ───────────────────────


def test_combined_loss_returns_scalar() -> None:
    pred = torch.rand(3, 20, 2) * 10
    true = torch.rand(3, 20, 2) * 10
    result = CombinedRolloutLoss().compute(pred, true)
    assert result.dim() == 0


def test_combined_loss_default_components() -> None:
    pred = torch.rand(3, 20, 2) * 10
    true = torch.rand(3, 20, 2) * 10
    result = CombinedRolloutLoss().compute(pred, true)
    assert torch.isfinite(result)
    assert result.item() >= 0.0


def test_combined_trajectory_loss_alias() -> None:
    """CombinedTrajectoryLoss is the backward-compat alias for CombinedRolloutLoss."""
    assert CombinedTrajectoryLoss is CombinedRolloutLoss


def test_rollout_loss_is_abstract() -> None:
    """RolloutLoss cannot be instantiated directly."""
    import inspect

    assert inspect.isabstract(RolloutLoss)
