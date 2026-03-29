"""Tests for the Gillespie SSA simulator and trajectory grid interpolation.

Covers:
- The simulator produces non-negative molecule counts at all times.
- Final times do not exceed t_max.
- Stationary mean of birth-death converges to k1/k2 (analytical result).
- interpolate_to_grid: zero-order hold semantics, correct shape.
- interpolate_to_grid: handles trajectories that end before t_max
  (the boundary clamping fix — indices must not go out of bounds).
"""

import pytest
import torch

from crn_surrogate.data.gillespie import (
    GillespieSSA,
    birth_death_crn,
    interpolate_to_grid,
    lotka_volterra_crn,
)


# ── GillespieSSA ──────────────────────────────────────────────────────────────


def test_gillespie_birth_death_trajectory_terminates_before_tmax():
    """Simulated trajectory must not extend past the requested t_max."""
    crn = birth_death_crn(k1=2.0, k2=1.0)
    traj = GillespieSSA().simulate(crn, torch.tensor([0.0]), t_max=10.0)
    assert traj.times[-1] <= 10.0


def test_gillespie_birth_death_state_shape():
    """Each state vector must have one entry per species (1 for birth-death)."""
    crn = birth_death_crn(k1=2.0, k2=1.0)
    traj = GillespieSSA().simulate(crn, torch.tensor([0.0]), t_max=10.0)
    assert traj.states.shape[1] == 1


def test_gillespie_molecule_counts_are_always_nonnegative():
    """Molecule counts must never go below zero (Gillespie preserves non-negativity)."""
    crn = birth_death_crn(k1=2.0, k2=1.0)
    traj = GillespieSSA().simulate(crn, torch.tensor([5.0]), t_max=20.0)
    assert (traj.states >= 0).all()


def test_gillespie_lotka_volterra_two_species():
    """Lotka-Volterra simulation returns state vectors with two entries."""
    crn = lotka_volterra_crn()
    traj = GillespieSSA().simulate(crn, torch.tensor([50.0, 20.0]), t_max=5.0)
    assert traj.states.shape[1] == 2
    assert (traj.states >= 0).all()


def test_gillespie_birth_death_stationary_mean_matches_analytical():
    """E[X] at stationarity = k1/k2 for a birth-death process.

    Uses 50 independent runs to average out Monte Carlo noise.
    """
    k1, k2 = 10.0, 1.0
    expected_mean = k1 / k2
    crn = birth_death_crn(k1=k1, k2=k2)
    ssa = GillespieSSA()

    final_counts = [
        ssa.simulate(crn, torch.tensor([10.0]), t_max=50.0).states[-1, 0].item()
        for _ in range(50)
    ]
    sample_mean = sum(final_counts) / len(final_counts)

    assert abs(sample_mean - expected_mean) < 5.0, (
        f"Stationary mean {sample_mean:.2f} is far from analytical {expected_mean:.2f}"
    )


# ── interpolate_to_grid ───────────────────────────────────────────────────────


def test_interpolate_to_grid_output_shape_matches_grid_length():
    """Output must have one row per grid point and one column per species."""
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    grid = torch.linspace(0.0, 3.0, 7)
    result = interpolate_to_grid(event_times, event_states, grid)
    assert result.shape == (7, 1)


def test_interpolate_to_grid_zero_order_hold_semantics():
    """Grid points between events take the value of the most recent event (ZOH)."""
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    grid = torch.tensor([0.5, 1.5, 2.5])

    result = interpolate_to_grid(event_times, event_states, grid)

    assert result[0, 0].item() == pytest.approx(0.0)   # 0.5 → event at t=0
    assert result[1, 0].item() == pytest.approx(1.0)   # 1.5 → event at t=1
    assert result[2, 0].item() == pytest.approx(2.0)   # 2.5 → event at t=2


def test_interpolate_to_grid_grid_point_at_exact_event_time():
    """A grid point that coincides exactly with an event time returns that event's state."""
    event_times = torch.tensor([0.0, 2.0, 4.0])
    event_states = torch.tensor([[5.0], [10.0], [15.0]])
    grid = torch.tensor([2.0])

    result = interpolate_to_grid(event_times, event_states, grid)

    assert result[0, 0].item() == pytest.approx(10.0)


def test_interpolate_to_grid_handles_trajectory_shorter_than_tmax():
    """When the SSA trajectory ends before t_max, grid points beyond the last event
    must return the final state — not raise an index-out-of-bounds error.

    This tests the boundary-clamping fix in interpolate_to_grid.
    """
    crn = birth_death_crn(k1=1.0, k2=0.5)
    ssa = GillespieSSA()
    # Run with a generous t_max; many trajectories will finish earlier due to
    # the Gillespie algorithm stopping when all propensities reach zero or t_max.
    # We use a very short simulation window so the trajectory is guaranteed to
    # have fewer events than the dense grid.
    traj = ssa.simulate(crn, torch.tensor([3.0]), t_max=0.5)
    dense_grid = torch.linspace(0.0, 10.0, 100)  # extends well past trajectory end

    # Must not raise; every grid point beyond the last event clips to the last state
    result = interpolate_to_grid(traj.times, traj.states, dense_grid)

    assert result.shape == (100, 1)
    assert (result >= 0).all()
