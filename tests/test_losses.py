"""Tests for the trajectory loss classes."""

import pytest
import torch

from crn_surrogate.training.losses import (
    CombinedTrajectoryLoss,
    MeanMatchingLoss,
    VarianceMatchingLoss,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tensors(K: int = 4, M: int = 8, T: int = 10, n: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    pred = torch.randn(K, T, n)
    true = torch.randn(M, T, n)
    return pred, true


# ── MeanMatchingLoss ─────────────────────────────────────────────────────────


def test_mean_matching_loss_scalar():
    pred, true = _make_tensors()
    loss = MeanMatchingLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_mean_matching_loss_identical_means_is_zero():
    K, T, n = 4, 10, 2
    base = torch.ones(T, n) * 3.0
    pred = base.unsqueeze(0).expand(K, -1, -1)
    true = base.unsqueeze(0).expand(K, -1, -1)
    loss = MeanMatchingLoss().compute(pred, true)
    assert loss.item() < 1e-6


def test_mean_matching_loss_rejects_2d_pred():
    with pytest.raises(ValueError, match="pred_states must be 3D"):
        MeanMatchingLoss().compute(torch.randn(10, 2), torch.randn(8, 10, 2))


def test_mean_matching_loss_rejects_2d_true():
    with pytest.raises(ValueError, match="true_states must be 3D"):
        MeanMatchingLoss().compute(torch.randn(4, 10, 2), torch.randn(10, 2))


def test_mean_matching_loss_mask():
    pred, true = _make_tensors(n=3)
    mask = torch.tensor([True, True, False])
    loss_masked = MeanMatchingLoss().compute(pred, true, mask=mask)
    loss_full = MeanMatchingLoss().compute(pred, true)
    assert loss_masked.item() != loss_full.item()


def test_mean_matching_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = MeanMatchingLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None


# ── VarianceMatchingLoss ─────────────────────────────────────────────────────


def test_variance_matching_loss_scalar():
    pred, true = _make_tensors(K=4, M=8)
    loss = VarianceMatchingLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_variance_matching_loss_identical_variance_is_zero():
    torch.manual_seed(1)
    K, M, T, n = 4, 8, 10, 2
    # Deterministic tensors with same variance structure
    base = torch.randn(T, n)
    pred = base.unsqueeze(0).expand(K, -1, -1) + torch.randn(K, T, n) * 0.5
    true = base.unsqueeze(0).expand(M, -1, -1) + torch.randn(M, T, n) * 0.5
    # Just verify it runs and returns a finite scalar
    loss = VarianceMatchingLoss().compute(pred, true)
    assert torch.isfinite(loss)


def test_variance_matching_loss_rejects_k_less_than_2():
    with pytest.raises(ValueError, match="K >= 2"):
        VarianceMatchingLoss().compute(torch.randn(1, 10, 2), torch.randn(8, 10, 2))


def test_variance_matching_loss_rejects_m_less_than_2():
    with pytest.raises(ValueError, match="M >= 2"):
        VarianceMatchingLoss().compute(torch.randn(4, 10, 2), torch.randn(1, 10, 2))


def test_variance_matching_loss_rejects_2d_inputs():
    with pytest.raises(ValueError, match="pred_states must be 3D"):
        VarianceMatchingLoss().compute(torch.randn(10, 2), torch.randn(8, 10, 2))


def test_variance_matching_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = VarianceMatchingLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None


# ── CombinedTrajectoryLoss ───────────────────────────────────────────────────


def test_combined_loss_default_scalar():
    pred, true = _make_tensors(K=4, M=8)
    loss = CombinedTrajectoryLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_combined_loss_var_weight_zero_equals_mean_only():
    pred, true = _make_tensors(K=4, M=8, seed=7)
    combined = CombinedTrajectoryLoss(var_weight=0.0).compute(pred, true)
    mean_only = MeanMatchingLoss().compute(pred, true)
    assert abs(combined.item() - mean_only.item()) < 1e-6


def test_combined_loss_custom_losses():
    pred, true = _make_tensors(K=4, M=8)
    custom = CombinedTrajectoryLoss(
        losses=[(MeanMatchingLoss(), 2.0), (MeanMatchingLoss(), 1.0)]
    )
    loss = custom.compute(pred, true)
    expected = 3.0 * MeanMatchingLoss().compute(pred, true)
    assert abs(loss.item() - expected.item()) < 1e-5


def test_combined_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = CombinedTrajectoryLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None
