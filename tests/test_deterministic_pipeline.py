"""Tests for the deterministic ODE simulation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crn_surrogate.data.generation.reference_crns import birth_death, lotka_volterra
from experiments.scripts.generate_dataset import _simulate_ode


class TestODESimulation:
    """Test the forward Euler ODE simulation."""

    def test_birth_death_reaches_steady_state(self):
        """Birth-death ODE should reach steady state k_birth/k_death."""
        crn = birth_death(k_birth=5.0, k_death=0.5)
        initial_state = torch.tensor([0.0])
        time_grid = torch.linspace(0.0, 50.0, 200)
        result = _simulate_ode(crn, initial_state, 50.0, time_grid)
        assert result is not None
        final_state = result[0, -1, 0].item()
        expected_steady_state = 5.0 / 0.5  # = 10.0
        assert abs(final_state - expected_steady_state) < 1.0

    def test_single_trajectory_format(self):
        """ODE output should be (1, T, n_species) tensor."""
        crn = birth_death(k_birth=1.0, k_death=0.1)
        initial_state = torch.tensor([5.0])
        T = 50
        time_grid = torch.linspace(0.0, 20.0, T)
        result = _simulate_ode(crn, initial_state, 20.0, time_grid)
        assert result is not None
        assert result.shape == (1, T, 1)

    def test_blowup_returns_none(self):
        """ODE with extreme production should return None when threshold is exceeded."""
        crn = birth_death(k_birth=1000.0, k_death=0.001)
        initial_state = torch.tensor([10.0])
        # Use a very low threshold so that linear growth from ~10 to >1000 triggers it.
        time_grid = torch.linspace(0.0, 100.0, 100)
        result = _simulate_ode(
            crn, initial_state, 100.0, time_grid, blowup_threshold=1e3
        )
        assert result is None

    def test_lotka_volterra_oscillates(self):
        """Lotka-Volterra ODE should show oscillatory dynamics (nonzero std)."""
        crn = lotka_volterra(k_prey_birth=1.0, k_predation=0.01, k_predator_death=0.5)
        initial_state = torch.tensor([40.0, 9.0])
        time_grid = torch.linspace(0.0, 30.0, 300)
        result = _simulate_ode(crn, initial_state, 30.0, time_grid)
        assert result is not None
        # Both species should have nonzero temporal variation
        temporal_std = result[0, :, :].std(dim=0)
        assert temporal_std[0].item() > 0.5
        assert temporal_std[1].item() > 0.5


class TestDeterministicSolver:
    """Test that EulerODESolver produces noise-free trajectories."""

    @pytest.fixture
    def solver_setup(self):
        """Create a minimal SDE + context for solver tests."""
        from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
        from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
        from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
        from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
        from crn_surrogate.simulator.ode_solver import EulerODESolver
        from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver

        crn = birth_death(k_birth=1.0, k_death=0.1)
        crn_repr = crn_to_tensor_repr(crn)

        encoder_config = EncoderConfig(d_model=64)
        encoder = BipartiteGNNEncoder(encoder_config)
        with torch.no_grad():
            crn_context = encoder(crn_repr)

        sde_config = SDEConfig(
            d_model=64, d_hidden=128, n_noise_channels=crn.n_reactions
        )
        sde = CRNNeuralSDE(sde_config, n_species=crn.n_species)

        ode_solver = EulerODESolver(sde_config)
        stoch_solver = EulerMaruyamaSolver(sde_config)
        t_span = torch.linspace(0.0, 5.0, 20)
        initial_state = torch.tensor([5.0])

        return ode_solver, stoch_solver, sde, initial_state, crn_context, t_span

    def test_ode_solver_is_reproducible(self, solver_setup):
        """Two EulerODESolver rollouts from the same state should be identical."""
        ode_solver, _, sde, initial_state, crn_context, t_span = solver_setup
        with torch.no_grad():
            traj1 = ode_solver.solve(
                sde, initial_state.clone(), crn_context, t_span, dt=0.1
            )
            traj2 = ode_solver.solve(
                sde, initial_state.clone(), crn_context, t_span, dt=0.1
            )
        torch.testing.assert_close(traj1.states, traj2.states)

    def test_ode_solver_differs_from_stochastic(self, solver_setup):
        """EulerODESolver should differ from EulerMaruyamaSolver (with overwhelming probability)."""
        ode_solver, stoch_solver, sde, initial_state, crn_context, t_span = solver_setup
        with torch.no_grad():
            det_traj = ode_solver.solve(
                sde, initial_state.clone(), crn_context, t_span, dt=0.1
            )
            stoch_traj = stoch_solver.solve(
                sde, initial_state.clone(), crn_context, t_span, dt=0.1
            )
        assert not torch.allclose(det_traj.states, stoch_traj.states)
