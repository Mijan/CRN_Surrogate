"""Tests verifying type-system contracts between models and simulators.

Covers:
- EulerODESolver works with NeuralDrift (no diffusion needed).
- EulerMaruyamaSolver requires a StochasticSurrogate; NeuralDrift causes AttributeError.
- NeuralDrift allocates fewer parameters than NeuralSDE.
- NeuralDrift isinstance checks are correct.
"""

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulator.base import StochasticSurrogate, SurrogateModel
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver


def _make_context(d_model: int = 16):
    crn = birth_death()
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
    return encoder(crn_to_tensor_repr(crn))


def _minimal_config(n_noise_channels: int = 4) -> SDEConfig:
    return SDEConfig(d_model=16, d_hidden=32, n_noise_channels=n_noise_channels)


def test_euler_ode_accepts_drift_only_model():
    """EulerODESolver works with NeuralDrift (no diffusion network needed)."""
    config = _minimal_config()
    model = NeuralDrift(config, n_species=1)
    solver = EulerODESolver(SolverConfig())
    ctx = _make_context()
    t_span = torch.linspace(0.0, 2.0, 5)
    traj = solver.solve(model, torch.tensor([3.0]), ctx, t_span, dt=0.1)
    assert traj.states.shape == (5, 1)


def test_euler_maruyama_requires_stochastic_model():
    """EulerMaruyamaSolver.solve() raises AttributeError when given NeuralDrift."""
    config = _minimal_config()
    drift_model = NeuralDrift(config, n_species=1)
    solver = EulerMaruyamaSolver(SolverConfig())
    ctx = _make_context()
    t_span = torch.linspace(0.0, 2.0, 5)
    with pytest.raises(AttributeError):
        solver.solve(drift_model, torch.tensor([3.0]), ctx, t_span, dt=0.1)


def test_neural_drift_has_no_diffusion_params():
    """NeuralDrift allocates no diffusion parameters; its count is strictly less than NeuralSDE."""
    config = _minimal_config(n_noise_channels=4)
    drift_model = NeuralDrift(config, n_species=3)
    sde_model = NeuralSDE(config, n_species=3)
    assert sum(p.numel() for p in drift_model.parameters()) < sum(
        p.numel() for p in sde_model.parameters()
    )


def test_neural_drift_isinstance():
    """NeuralDrift is a SurrogateModel but not a StochasticSurrogate."""
    model = NeuralDrift(_minimal_config(), n_species=3)
    assert isinstance(model, SurrogateModel)
    assert not isinstance(model, StochasticSurrogate)
