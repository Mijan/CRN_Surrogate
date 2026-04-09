"""Tests for TimegridUtils.interpolate_to_grid."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.simulation.timegrid_utils import TimegridUtils


def test_basic_interpolation():
    event_times = torch.tensor([0.0, 1.0, 3.0])
    event_states = torch.tensor([[0.0], [10.0], [30.0]])
    grid = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    expected = torch.tensor([[0.0], [0.0], [10.0], [10.0], [30.0]])
    assert torch.allclose(result, expected)


def test_single_event():
    event_times = torch.tensor([0.0])
    event_states = torch.tensor([[5.0, 5.0]])
    grid = torch.linspace(0.0, 1.0, 10)
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    assert result.shape == (10, 2)
    assert torch.allclose(result, torch.tensor([[5.0, 5.0]]).expand(10, 2))


def test_grid_beyond_last_event():
    event_times = torch.tensor([0.0, 1.0, 2.0])
    event_states = torch.tensor([[1.0], [2.0], [3.0]])
    grid = torch.tensor([0.5, 1.5, 2.5, 3.5, 5.0])
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    # Points after t=2 should get the last state (3.0)
    assert result[-1, 0].item() == pytest.approx(3.0)
    assert result[-2, 0].item() == pytest.approx(3.0)


def test_grid_at_exact_event_times():
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    grid = event_times.clone()
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    assert torch.allclose(result, event_states)


def test_empty_grid():
    event_times = torch.tensor([0.0, 1.0])
    event_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    grid = torch.tensor([], dtype=torch.float32)
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    assert result.shape == (0, 2)
