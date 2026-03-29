"""Tests for the Gillespie SSA simulator."""

import torch

from crn_surrogate.data.gillespie import (
    GillespieSSA,
    birth_death_crn,
    interpolate_to_grid,
)


def test_gillespie_birth_death_runs():
    crn = birth_death_crn(k1=2.0, k2=1.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(crn, initial_state=torch.tensor([0.0]), t_max=10.0)
    assert traj.states.shape[1] == 1
    assert traj.times[-1] <= 10.0


def test_gillespie_state_nonnegative():
    crn = birth_death_crn(k1=2.0, k2=1.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(crn, initial_state=torch.tensor([5.0]), t_max=20.0)
    assert (traj.states >= 0).all()


def test_gillespie_birth_death_stationary_mean():
    """For birth-death, E[X] = k1/k2. Test with many samples."""
    k1, k2 = 10.0, 1.0
    expected_mean = k1 / k2
    crn = birth_death_crn(k1=k1, k2=k2)
    ssa = GillespieSSA()

    final_counts = []
    for _ in range(50):
        traj = ssa.simulate(crn, initial_state=torch.tensor([10.0]), t_max=50.0)
        final_counts.append(traj.states[-1, 0].item())

    sample_mean = sum(final_counts) / len(final_counts)
    # Allow generous tolerance since it's stochastic
    assert abs(sample_mean - expected_mean) < 5.0, (
        f"Mean {sample_mean:.2f} != {expected_mean}"
    )


def test_interpolate_to_grid():
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    grid = torch.tensor([0.5, 1.5, 2.5])
    result = interpolate_to_grid(event_times, event_states, grid)
    assert result.shape == (3, 1)
    # zero-order hold: 0.5 → state at time 0, 1.5 → state at time 1, etc.
    assert result[0, 0].item() == 0.0
    assert result[1, 0].item() == 1.0
    assert result[2, 0].item() == 2.0
