"""Tests for GillespieSSA.simulate_batch and Trajectory.stack_on_grid."""

import torch

from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory


def test_simulate_batch_shape() -> None:
    """stack_on_grid output shape is (n_trajectories, len(time_grid), n_species)."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 10.0, 20)
    trajs = ssa.simulate_batch(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=10.0,
        n_trajectories=5,
        n_workers=1,
    )
    result = Trajectory.stack_on_grid(trajs, time_grid)
    assert result.shape == (5, 20, 1)


def test_simulate_batch_parallel_shape() -> None:
    """Parallel execution produces the same shape as sequential."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 10.0, 20)
    trajs = ssa.simulate_batch(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=10.0,
        n_trajectories=10,
        n_workers=2,
    )
    result = Trajectory.stack_on_grid(trajs, time_grid)
    assert result.shape == (10, 20, 1)


def test_simulate_batch_parallel_statistics() -> None:
    """Parallel and sequential produce similar distributional statistics."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 15.0, 30)
    init = torch.tensor([0.0])

    torch.manual_seed(0)
    seq = Trajectory.stack_on_grid(
        ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init,
            t_max=15.0,
            n_trajectories=50,
            n_workers=1,
        ),
        time_grid,
    )

    torch.manual_seed(1)
    par = Trajectory.stack_on_grid(
        ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init,
            t_max=15.0,
            n_trajectories=50,
            n_workers=2,
        ),
        time_grid,
    )

    # Both should have final mean near k_birth/k_death = 4.0
    seq_final_mean = seq[:, -1, 0].mean().item()
    par_final_mean = par[:, -1, 0].mean().item()
    assert abs(seq_final_mean - 4.0) < 2.0, (
        f"Sequential mean {seq_final_mean} too far from 4.0"
    )
    assert abs(par_final_mean - 4.0) < 2.0, (
        f"Parallel mean {par_final_mean} too far from 4.0"
    )


def test_simulate_batch_no_nan() -> None:
    """No NaN or Inf in output trajectories."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 10.0, 20)
    result = Trajectory.stack_on_grid(
        ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=torch.tensor([0.0]),
            t_max=10.0,
            n_trajectories=10,
            n_workers=2,
        ),
        time_grid,
    )
    assert torch.isfinite(result).all()
