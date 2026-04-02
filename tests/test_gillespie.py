"""Tests for the CRN-agnostic GillespieSSA simulator and interpolation utilities.

Covers:
- The simulator accepts (stoichiometry, propensity_fn, initial_state, t_max).
- Works identically with CRN.evaluate_propensities and a raw lambda.
- Non-negative molecule counts at all times.
- Final times do not exceed t_max.
- Stationary mean of birth-death converges to k_birth/k_death (analytical result).
- TimegridUtils.interpolate_to_grid: zero-order hold semantics, correct shape.
- TimegridUtils.interpolate_to_grid: handles trajectories that end before t_max.
"""

import pytest
import torch

from crn_surrogate.crn.examples import birth_death, lotka_volterra
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.interpolation import TimegridUtils

# ── GillespieSSA with CRN.evaluate_propensities ────────────────────────────────


def test_gillespie_birth_death_trajectory_terminates_before_tmax():
    """Simulated trajectory must not extend past the requested t_max."""
    crn = birth_death(k_birth=2.0, k_death=1.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=10.0,
    )
    assert traj.times[-1] <= 10.0


def test_gillespie_birth_death_state_shape():
    """Each state vector must have one entry per species."""
    crn = birth_death(k_birth=2.0, k_death=1.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=10.0,
    )
    assert traj.states.shape[1] == 1


def test_gillespie_molecule_counts_are_nonnegative():
    """Molecule counts must never go below zero."""
    crn = birth_death(k_birth=2.0, k_death=1.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([5.0]),
        t_max=20.0,
    )
    assert (traj.states >= 0).all()


def test_gillespie_lotka_volterra_two_species():
    """Lotka-Volterra simulation returns state vectors with two entries."""
    crn = lotka_volterra()
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([50.0, 20.0]),
        t_max=5.0,
    )
    assert traj.states.shape[1] == 2
    assert (traj.states >= 0).all()


def test_gillespie_birth_death_stationary_mean_matches_analytical():
    """E[X] at stationarity = k_birth/k_death for birth-death.

    Uses 50 independent runs to average out Monte Carlo noise.
    """
    k_birth, k_death = 10.0, 1.0
    expected_mean = k_birth / k_death
    crn = birth_death(k_birth=k_birth, k_death=k_death)
    ssa = GillespieSSA()

    final_counts = [
        ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=torch.tensor([10.0]),
            t_max=50.0,
        )
        .states[-1, 0]
        .item()
        for _ in range(50)
    ]
    sample_mean = sum(final_counts) / len(final_counts)
    assert abs(sample_mean - expected_mean) < 5.0, (
        f"Stationary mean {sample_mean:.2f} is far from analytical {expected_mean:.2f}"
    )


# ── GillespieSSA with a raw lambda (no CRN object) ────────────────────────────


def test_gillespie_accepts_raw_lambda_propensity():
    """Simulator works with a raw lambda as the propensity function."""
    stoich = torch.tensor([[1], [-1]])
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=stoich,
        propensity_fn=lambda state, t: torch.tensor([1.0, 0.1 * state[0]]),
        initial_state=torch.tensor([10.0]),
        t_max=5.0,
    )
    assert traj.states.shape[1] == 1
    assert (traj.states >= 0).all()


def test_gillespie_warns_for_time_varying_propensity():
    """Simulator issues a UserWarning when propensity changes between t=0 and t=1."""
    stoich = torch.tensor([[1]])
    with pytest.warns(UserWarning, match="time-varying"):
        GillespieSSA().simulate(
            stoichiometry=stoich,
            propensity_fn=lambda state, t: torch.tensor([1.0 + t]),
            initial_state=torch.tensor([0.0]),
            t_max=1.0,
        )


# ── interpolate_to_grid ───────────────────────────────────────────────────────


def test_interpolate_to_grid_output_shape_matches_grid_length():
    """Output must have one row per grid point and one column per species."""
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    grid = torch.linspace(0.0, 3.0, 7)
    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)
    assert result.shape == (7, 1)


def test_interpolate_to_grid_zero_order_hold_semantics():
    """Grid points between events take the value of the most recent event (ZOH)."""
    event_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    event_states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    grid = torch.tensor([0.5, 1.5, 2.5])

    result = TimegridUtils.interpolate_to_grid(event_times, event_states, grid)

    assert result[0, 0].item() == pytest.approx(0.0)  # 0.5 → event at t=0
    assert result[1, 0].item() == pytest.approx(1.0)  # 1.5 → event at t=1
    assert result[2, 0].item() == pytest.approx(2.0)  # 2.5 → event at t=2


def test_interpolate_to_grid_grid_point_at_exact_event_time():
    """A grid point coinciding with an event returns that event's state."""
    event_times = torch.tensor([0.0, 2.0, 4.0])
    event_states = torch.tensor([[5.0], [10.0], [15.0]])
    result = TimegridUtils.interpolate_to_grid(
        event_times, event_states, torch.tensor([2.0])
    )
    assert result[0, 0].item() == pytest.approx(10.0)


def test_interpolate_to_grid_handles_trajectory_shorter_than_tmax():
    """Grid points beyond the last event clip to the final state (no index error)."""
    crn = birth_death(k_birth=1.0, k_death=0.5)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([3.0]),
        t_max=0.5,
    )
    dense_grid = torch.linspace(0.0, 10.0, 100)  # extends well past trajectory end

    result = TimegridUtils.interpolate_to_grid(traj.times, traj.states, dense_grid)

    assert result.shape == (100, 1)
    assert (result >= 0).all()


# ── max_reactions cap ─────────────────────────────────────────────────────────


def test_gillespie_max_reactions_cap_stops_simulation_early():
    """When max_reactions is set, the trajectory has at most max_reactions + 1 events.

    Uses a high birth rate to ensure many reactions would fire before t_max.
    """
    crn = birth_death(k_birth=1000.0, k_death=0.0)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=1000.0,
        max_reactions=10,
    )
    # +1 for the initial state, +1 for the final t_max entry appended by Fix 1
    assert traj.times.shape[0] <= 12
