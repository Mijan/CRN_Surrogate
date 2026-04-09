"""Tests for StateTransform, Log1pTransform, and get_state_transform."""

from __future__ import annotations

import math

import pytest
import torch

from crn_surrogate.simulator.state_transform import (
    Log1pTransform,
    StateTransform,
    get_state_transform,
)

# ── StateTransform (identity) ─────────────────────────────────────────────────


def test_identity_forward() -> None:
    t = StateTransform()
    x = torch.randn(5)
    assert torch.allclose(t.forward(x), x)


def test_identity_inverse() -> None:
    t = StateTransform()
    x = torch.randn(5)
    assert torch.allclose(t.inverse(x), x)


def test_identity_round_trip() -> None:
    t = StateTransform()
    x = torch.randn(5)
    assert torch.allclose(t.inverse(t.forward(x)), x)


# ── Log1pTransform ────────────────────────────────────────────────────────────


def test_log1p_forward_values() -> None:
    t = Log1pTransform()
    x = torch.tensor([0.0, 1.0, 9.0])
    out = t.forward(x)
    expected = torch.tensor([0.0, math.log(2), math.log(10)])
    assert torch.allclose(out, expected, atol=1e-5)


def test_log1p_inverse_values() -> None:
    t = Log1pTransform()
    z = torch.tensor([0.0, math.log(2), math.log(10)])
    out = t.inverse(z)
    expected = torch.tensor([0.0, 1.0, 9.0])
    assert torch.allclose(out, expected, atol=1e-5)


def test_log1p_round_trip() -> None:
    t = Log1pTransform()
    x = torch.tensor([0.0, 5.0, 100.0, 1000.0])
    assert torch.allclose(t.inverse(t.forward(x)), x, atol=1e-5)


def test_log1p_clamps_negative() -> None:
    t = Log1pTransform()
    out = t.forward(torch.tensor([-5.0]))
    assert out.item() == pytest.approx(0.0)


def test_log1p_inverse_clamps_negative() -> None:
    t = Log1pTransform()
    out = t.inverse(torch.tensor([-100.0]))
    assert out.item() == pytest.approx(0.0, abs=1e-5)


# ── get_state_transform factory ───────────────────────────────────────────────


def test_factory_identity() -> None:
    t = get_state_transform(False)
    assert type(t) is StateTransform


def test_factory_log1p() -> None:
    t = get_state_transform(True)
    assert isinstance(t, Log1pTransform)


# ── Trajectory transforms ─────────────────────────────────────────────────────


def test_transform_trajectory_shape() -> None:
    t = Log1pTransform()
    traj = torch.rand(5, 10, 3)
    out = t.transform_trajectory(traj)
    assert out.shape == (5, 10, 3)


def test_inverse_trajectory_round_trip() -> None:
    t = Log1pTransform()
    traj = torch.rand(5, 10, 3) * 10  # raw counts
    recovered = t.inverse_trajectory(t.transform_trajectory(traj))
    assert torch.allclose(recovered, traj, atol=1e-5)
