"""Tests for TrajectoryNormalizer."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.training.normalization import TrajectoryNormalizer


def test_compute_scale_values():
    """Scale is mean absolute value per species, clamped at min_scale=1.0."""
    trajs = torch.tensor(
        [
            [[10.0, 0.1], [20.0, 0.2]],
            [[30.0, 0.3], [40.0, 0.4]],
        ]
    )  # (M=2, T=2, n_species=2)
    normalizer = TrajectoryNormalizer()
    scale = normalizer.compute_scale(trajs)
    assert scale.shape == (2,)
    assert scale[0].item() == pytest.approx(25.0)  # mean of [10,20,30,40]
    assert scale[1].item() == pytest.approx(1.0)  # mean=0.25 → clamped to 1.0


def test_compute_scale_shape():
    """Output shape matches n_species."""
    normalizer = TrajectoryNormalizer()
    trajs = torch.randn(4, 10, 5).abs() * 50
    scale = normalizer.compute_scale(trajs)
    assert scale.shape == (5,)


def test_compute_scale_min_clamp():
    """Near-zero species are not amplified below min_scale."""
    normalizer = TrajectoryNormalizer(min_scale=1.0)
    trajs = torch.zeros(2, 5, 3)
    trajs[:, :, 0] = 100.0  # species 0: large count
    trajs[:, :, 1] = 0.01  # species 1: tiny count
    trajs[:, :, 2] = 0.0  # species 2: zero
    scale = normalizer.compute_scale(trajs)
    assert scale[0].item() == pytest.approx(100.0)
    assert scale[1].item() == pytest.approx(1.0)  # clamped
    assert scale[2].item() == pytest.approx(1.0)  # clamped (0.0 < 1.0)


def test_compute_scale_custom_min():
    """Custom min_scale is respected."""
    normalizer = TrajectoryNormalizer(min_scale=2.0)
    trajs = torch.ones(2, 5, 2) * 0.5
    scale = normalizer.compute_scale(trajs)
    assert (scale >= 2.0).all()


def test_min_scale_property():
    """min_scale property returns the value passed to the constructor."""
    assert TrajectoryNormalizer().min_scale == 1.0
    assert TrajectoryNormalizer(min_scale=5.0).min_scale == 5.0


def test_normalize_denormalize_roundtrip():
    """normalize then denormalize recovers the original tensor."""
    normalizer = TrajectoryNormalizer()
    trajs = torch.randn(5, 10, 3).abs() * 100
    scale = normalizer.compute_scale(trajs)
    normed = normalizer.normalize(trajs, scale)
    recovered = normalizer.denormalize(normed, scale)
    assert torch.allclose(trajs, recovered, atol=1e-5)


def test_normalized_trajectories_are_order_one():
    """After normalization every species has mean absolute value ≈ 1."""
    torch.manual_seed(0)
    normalizer = TrajectoryNormalizer()
    trajs = torch.randn(10, 20, 3).abs() * torch.tensor([1.0, 100.0, 10000.0])
    scale = normalizer.compute_scale(trajs)
    normed = normalizer.normalize(trajs, scale)
    for s in range(3):
        mean_abs = normed[:, :, s].abs().mean().item()
        assert 0.1 < mean_abs < 10.0, f"Species {s}: mean_abs={mean_abs:.4f}"


def test_normalize_preserves_shape():
    """normalize and denormalize preserve tensor shape."""
    normalizer = TrajectoryNormalizer()
    trajs = torch.rand(3, 7, 4) * 50
    scale = normalizer.compute_scale(trajs)
    assert normalizer.normalize(trajs, scale).shape == trajs.shape
    assert normalizer.denormalize(trajs, scale).shape == trajs.shape


def test_normalize_1d_input():
    """normalize broadcasts correctly for (n_species,) input."""
    normalizer = TrajectoryNormalizer()
    scale = torch.tensor([10.0, 100.0])
    state = torch.tensor([5.0, 200.0])
    normed = normalizer.normalize(state, scale)
    assert normed.shape == (2,)
    assert normed[0].item() == pytest.approx(0.5)
    assert normed[1].item() == pytest.approx(2.0)
