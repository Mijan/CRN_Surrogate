"""Tests for GillespieSSA."""

from __future__ import annotations

import torch

from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from tests.simulation.conftest import _birth_death_crn


def _run(crn, initial_state: torch.Tensor, t_max: float = 10.0, **kwargs) -> Trajectory:
    ssa = GillespieSSA()
    return ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=initial_state,
        t_max=t_max,
        **kwargs,
    )


def test_single_trajectory_returns_trajectory(birth_death_crn):
    traj = _run(birth_death_crn, torch.tensor([4.0]))
    assert isinstance(traj, Trajectory)
    assert torch.all(traj.times[1:] >= traj.times[:-1])  # sorted
    assert torch.all(traj.states >= 0.0)


def test_trajectory_starts_at_initial_state(birth_death_crn):
    x0 = torch.tensor([7.0])
    traj = _run(birth_death_crn, x0)
    assert torch.allclose(traj.states[0], x0)


def test_trajectory_ends_at_or_before_t_max(birth_death_crn):
    t_max = 5.0
    traj = _run(birth_death_crn, torch.tensor([4.0]), t_max=t_max)
    assert traj.times[-1].item() <= t_max + 1e-6


def test_states_non_negative(birth_death_crn):
    traj = _run(birth_death_crn, torch.tensor([4.0]))
    assert torch.all(traj.states >= 0.0)


def test_birth_death_stationary_mean():
    k_birth, k_death = 2.0, 0.5
    crn = _birth_death_crn(k_birth=k_birth, k_death=k_death)
    ssa = GillespieSSA()
    x0 = torch.tensor([4.0])
    n_trajs = 500
    final_states = []
    for _ in range(n_trajs):
        traj = ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=x0.clone(),
            t_max=50.0,
        )
        final_states.append(traj.states[-1, 0].item())
    mean = sum(final_states) / n_trajs
    expected = k_birth / k_death  # 4.0
    assert abs(mean - expected) < 2.0


def test_batch_returns_correct_count(birth_death_crn):
    ssa = GillespieSSA()
    trajs = ssa.simulate_batch(
        stoichiometry=birth_death_crn.stoichiometry_matrix,
        propensity_fn=birth_death_crn.evaluate_propensities,
        initial_state=torch.tensor([4.0]),
        t_max=5.0,
        n_trajectories=10,
    )
    assert len(trajs) == 10


def test_batch_trajectories_are_independent(birth_death_crn):
    ssa = GillespieSSA()
    trajs = ssa.simulate_batch(
        stoichiometry=birth_death_crn.stoichiometry_matrix,
        propensity_fn=birth_death_crn.evaluate_propensities,
        initial_state=torch.tensor([4.0]),
        t_max=10.0,
        n_trajectories=2,
    )
    # Two independent trajectories should not be identical
    # (equal number of steps AND identical states would be astronomically unlikely)
    if trajs[0].n_steps == trajs[1].n_steps:
        assert not torch.allclose(trajs[0].states, trajs[1].states)


def test_max_reactions_cap(birth_death_crn):
    traj = _run(birth_death_crn, torch.tensor([10.0]), t_max=1000.0, max_reactions=5)
    # At most 5 reaction events + initial state + t_max endpoint
    assert traj.n_steps <= 5 + 2


def test_absorbing_state(decay_crn):
    # Start from X=0: no reactions should fire
    x0 = torch.tensor([0.0])
    traj = _run(decay_crn, x0, t_max=10.0)
    # Should have just the initial state and t_max endpoint
    assert traj.n_steps == 2
    assert torch.all(traj.states == 0.0)


def test_reproducibility_with_seed(birth_death_crn):
    ssa = GillespieSSA()
    kwargs = dict(
        stoichiometry=birth_death_crn.stoichiometry_matrix,
        propensity_fn=birth_death_crn.evaluate_propensities,
        initial_state=torch.tensor([4.0]),
        t_max=5.0,
        n_trajectories=1,
    )
    torch.manual_seed(42)
    trajs1 = ssa.simulate_batch(**kwargs)

    torch.manual_seed(42)
    trajs2 = ssa.simulate_batch(**kwargs)

    assert trajs1[0].n_steps == trajs2[0].n_steps
    assert torch.allclose(trajs1[0].states, trajs2[0].states)
    assert torch.allclose(trajs1[0].times, trajs2[0].times)
