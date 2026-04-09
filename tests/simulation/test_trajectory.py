"""Tests for Trajectory and Trajectory.stack_on_grid."""

from __future__ import annotations

import torch

from crn_surrogate.simulation.trajectory import Trajectory


def test_properties():
    times = torch.tensor([0.0, 1.0, 2.0])
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    traj = Trajectory(times=times, states=states)
    assert traj.n_steps == 3
    assert traj.n_species == 2


def test_mean():
    times = torch.tensor([0.0, 1.0, 2.0])
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    traj = Trajectory(times=times, states=states)
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(traj.mean(), expected)


def test_to_grid_exact_times():
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    states = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    traj = Trajectory(times=times, states=states)
    grid = torch.tensor([0.0, 1.0, 2.0, 3.0])
    result = traj.to_grid(grid)
    assert torch.allclose(result, states)


def test_to_grid_intermediate_times():
    times = torch.tensor([0.0, 1.0, 3.0])
    states = torch.tensor([[10.0], [20.0], [30.0]])
    traj = Trajectory(times=times, states=states)
    grid = torch.tensor([0.5, 1.5, 2.5])
    result = traj.to_grid(grid)
    expected = torch.tensor([[10.0], [20.0], [20.0]])
    assert torch.allclose(result, expected)


def test_to_grid_before_first_event():
    times = torch.tensor([0.0, 1.0, 2.0])
    states = torch.tensor([[5.0], [10.0], [15.0]])
    traj = Trajectory(times=times, states=states)
    # Query before first event — should clamp to first state
    grid = torch.tensor([-1.0])
    result = traj.to_grid(grid)
    assert torch.allclose(result, torch.tensor([[5.0]]))


def test_stack_on_grid_shape():
    grid = torch.linspace(0.0, 1.0, 20)
    trajectories = []
    for i in range(5):
        n_events = 3 + i
        t = torch.linspace(0.0, 1.0, n_events)
        s = torch.rand(n_events, 2)
        trajectories.append(Trajectory(times=t, states=s))
    result = Trajectory.stack_on_grid(trajectories, grid)
    assert result.shape == (5, 20, 2)


def test_stack_on_grid_consistency():
    times = torch.tensor([0.0, 0.5, 1.0])
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    traj = Trajectory(times=times, states=states)
    grid = torch.linspace(0.0, 1.0, 10)
    stacked = Trajectory.stack_on_grid([traj], grid)
    expected = traj.to_grid(grid).unsqueeze(0)
    assert torch.allclose(stacked, expected)
