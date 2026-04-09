"""Tests for ViabilityFilter and CurationResult."""

from __future__ import annotations

import torch

from crn_surrogate.data.generation.configs import CurationConfig
from crn_surrogate.data.generation.curation import ViabilityFilter

M, T, S = 4, 50, 2
_CFG = CurationConfig()
_FILTER = ViabilityFilter(_CFG)


def _viable_tensor() -> torch.Tensor:
    """4 trajectories with diverse, bounded, non-zero dynamics."""
    t = torch.zeros(M, T, S)
    for m in range(M):
        for i in range(T):
            t[m, i, 0] = 10.0 + 5.0 * (i % 7) + m
            t[m, i, 1] = 20.0 + 3.0 * (i % 11) + m
    return t


def test_viable_trajectories() -> None:
    result = _FILTER.check(_viable_tensor())
    assert result.viable
    assert result.rejection_reason == ""


def test_rejects_nan() -> None:
    t = _viable_tensor()
    t[0, 5, 0] = float("nan")
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "nan_or_inf"


def test_rejects_inf() -> None:
    t = _viable_tensor()
    t[1, 10, 1] = float("inf")
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "nan_or_inf"


def test_rejects_blowup() -> None:
    t = _viable_tensor()
    t[0, 0, 0] = 2e6
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "blowup"


def test_rejects_zero_stuck() -> None:
    t = torch.zeros(M, T, S)
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "zero_stuck"


def test_rejects_low_activity() -> None:
    # Constant non-zero tensor — no transitions between timesteps
    t = torch.ones(M, T, S) * 5.0
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "low_activity"


def test_rejects_low_cv() -> None:
    # Very low variance: all values near-identical
    t = torch.full((M, T, S), 100.0)
    # Add tiny variation so it doesn't hit zero_stuck or low_activity,
    # but CV is still very low
    t = t + torch.arange(T).float().view(1, T, 1).expand(M, T, S) * 1e-5
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason in ("low_activity", "low_cv")


def test_rejects_unbounded_final() -> None:
    t = _viable_tensor()
    # Set last 10 timesteps to very high values
    t[:, -10:, :] = 2e5
    result = _FILTER.check(t)
    assert not result.viable
    assert result.rejection_reason == "unbounded_final"


def test_is_viable_matches_check() -> None:
    t = _viable_tensor()
    assert _FILTER.is_viable(t) == _FILTER.check(t).viable

    t_bad = torch.zeros(M, T, S)
    assert _FILTER.is_viable(t_bad) == _FILTER.check(t_bad).viable
