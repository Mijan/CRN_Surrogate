"""Tests for the Numba-accelerated SSA kernel.

All tests are skipped automatically when numba is not installed.

Covers:
- _gillespie_mass_action_inner: basic run, final time, seed diversity, stationarity, multi-species.
- _gillespie_batch_inner: output shape, no NaN.
- FastMassActionSSA: returns torch.Tensor with correct shape, statistical agreement with standard SSA.
- FastMassActionSSA.extract_topology_arrays: correct extraction from a CRN.
"""

import numpy as np
import pytest
import torch

# Skip all tests if numba is not installed
pytest.importorskip("numba")

from crn_surrogate.simulation.fast_ssa import (
    NUMBA_AVAILABLE,
    FastMassActionSSA,
    _gillespie_batch_inner,
    _gillespie_mass_action_inner,
)

# ── _gillespie_mass_action_inner ──────────────────────────────────────────────


class TestGillespieMassActionInner:
    """Tests for the JIT-compiled single-trajectory SSA kernel."""

    def test_birth_death_runs(self):
        """Birth-death CRN produces non-empty output with correct shapes."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        times, states = _gillespie_mass_action_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            10000,
            42,
        )
        assert times.shape[0] == states.shape[0]
        assert states.shape[1] == 1
        assert times[0] == 0.0
        assert times[-1] <= 10.0
        assert (states >= 0).all()

    def test_final_time_is_t_max(self):
        """The last recorded time equals t_max."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        times, _ = _gillespie_mass_action_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            100_000,
            42,
        )
        assert times[-1] == pytest.approx(10.0)

    def test_different_seeds_different_trajectories(self):
        """Two different seeds produce different trajectory realisations."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        _, states1 = _gillespie_mass_action_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            10000,
            42,
        )
        _, states2 = _gillespie_mass_action_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            10000,
            99,
        )
        assert not np.array_equal(states1, states2)

    def test_stationary_mean_birth_death(self):
        """Birth-death with k_b=2, k_d=0.5 has analytical stationary mean 4."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        final_counts = [
            _gillespie_mass_action_inner(
                stoich, reactant, rates, init, 50.0, 100_000, seed
            )[1][-1, 0]
            for seed in range(200)
        ]
        mean = float(np.mean(final_counts))
        assert abs(mean - 4.0) < 1.5, f"Stationary mean {mean:.2f} too far from 4.0"

    def test_two_species_lotka_volterra(self):
        """Two-species topology runs without error and keeps all states non-negative."""
        stoich = np.array(
            [
                [1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
            ]
        )
        reactant = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        rates = np.array([1.0, 0.01, 0.5])
        init = np.array([50.0, 50.0])
        times, states = _gillespie_mass_action_inner(
            stoich,
            reactant,
            rates,
            init,
            20.0,
            100_000,
            42,
        )
        assert states.shape[1] == 2
        assert (states >= 0).all()


# ── _gillespie_batch_inner ────────────────────────────────────────────────────


class TestGillespieBatchInner:
    """Tests for the batch + interpolation kernel."""

    def test_output_shape(self):
        """Result has shape (n_trajectories, T, n_species)."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        grid = np.linspace(0.0, 10.0, 20)
        seeds = np.arange(5, dtype=np.int64)
        result = _gillespie_batch_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            10_000,
            seeds,
            5,
            grid,
        )
        assert result.shape == (5, 20, 1)

    def test_no_nan_or_inf(self):
        """All interpolated values are finite."""
        stoich = np.array([[1.0], [-1.0]])
        reactant = np.array([[0.0], [1.0]])
        rates = np.array([2.0, 0.5])
        init = np.array([0.0])
        grid = np.linspace(0.0, 10.0, 20)
        seeds = np.arange(10, dtype=np.int64)
        result = _gillespie_batch_inner(
            stoich,
            reactant,
            rates,
            init,
            10.0,
            10_000,
            seeds,
            10,
            grid,
        )
        assert np.isfinite(result).all()


# ── FastMassActionSSA ─────────────────────────────────────────────────────────


class TestFastMassActionSSA:
    """Tests for the Python wrapper class."""

    def test_returns_torch_tensor(self):
        """simulate_batch returns a torch.Tensor with the expected shape."""
        fast = FastMassActionSSA()
        stoich = torch.tensor([[1.0], [-1.0]])
        reactant = torch.tensor([[0.0], [1.0]])
        rates = torch.tensor([2.0, 0.5])
        init = torch.tensor([0.0])
        grid = torch.linspace(0.0, 10.0, 20)
        result = fast.simulate_batch(stoich, reactant, rates, init, 10.0, grid, 5)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 20, 1)

    def test_matches_standard_ssa_statistics(self):
        """Fast SSA final-time mean agrees with the standard SSA (both near 4.0)."""
        from crn_surrogate.data.generation.mass_action_topology import (
            birth_death_topology,
        )
        from crn_surrogate.simulation.gillespie import GillespieSSA
        from crn_surrogate.simulation.trajectory import Trajectory

        topo = birth_death_topology()
        crn = topo.to_crn([2.0, 0.5])
        init = torch.tensor([0.0])
        grid = torch.linspace(0.0, 15.0, 30)

        torch.manual_seed(0)
        ssa = GillespieSSA()
        std_trajs = ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init,
            t_max=15.0,
            n_trajectories=100,
        )
        std_tensor = Trajectory.stack_on_grid(std_trajs, grid)

        torch.manual_seed(1)
        fast = FastMassActionSSA()
        fast_tensor = fast.simulate_batch(
            topo.net_stoichiometry,
            topo.reactant_matrix,
            torch.tensor([2.0, 0.5]),
            init,
            15.0,
            grid,
            100,
        )

        std_mean = std_tensor[:, -1, 0].mean().item()
        fast_mean = fast_tensor[:, -1, 0].mean().item()
        assert abs(std_mean - 4.0) < 2.0, (
            f"Standard SSA mean {std_mean:.2f} far from 4.0"
        )
        assert abs(fast_mean - 4.0) < 2.0, f"Fast SSA mean {fast_mean:.2f} far from 4.0"

    def test_extract_topology_arrays_birth_death(self):
        """extract_topology_arrays correctly extracts stoichiometry, reactant, rates."""
        from crn_surrogate.data.generation.mass_action_topology import (
            birth_death_topology,
        )

        topo = birth_death_topology()
        crn = topo.to_crn([2.0, 0.5])
        arrays = FastMassActionSSA.extract_topology_arrays(crn)

        assert "stoichiometry" in arrays
        assert "reactant_matrix" in arrays
        assert "rate_constants" in arrays

        np.testing.assert_allclose(
            arrays["stoichiometry"], topo.net_stoichiometry.numpy()
        )
        np.testing.assert_allclose(
            arrays["reactant_matrix"], topo.reactant_matrix.numpy()
        )
        np.testing.assert_allclose(arrays["rate_constants"], [2.0, 0.5])

    def test_extract_topology_arrays_raises_for_non_mass_action(self):
        """extract_topology_arrays raises ValueError for Hill-kinetics propensities."""
        from crn_surrogate.crn.crn import CRN
        from crn_surrogate.crn.propensities import hill
        from crn_surrogate.crn.reaction import Reaction

        crn = CRN(
            reactions=[
                Reaction(
                    stoichiometry=torch.tensor([1.0]),
                    propensity=hill(
                        v_max=1.0, k_m=5.0, hill_coefficient=2.0, species_index=0
                    ),
                )
            ]
        )
        with pytest.raises(ValueError, match="not supported"):
            FastMassActionSSA.extract_topology_arrays(crn)

    def test_numba_available_flag(self):
        """NUMBA_AVAILABLE is True when tests are running (importorskip ensures numba is present)."""
        assert NUMBA_AVAILABLE is True
