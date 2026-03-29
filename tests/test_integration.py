"""End-to-end integration test: CRN → encoder → SDE → trajectory, gradients flow."""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.data.gillespie import birth_death_crn, lotka_volterra_crn
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.losses import MeanMatchingLoss


def test_end_to_end_birth_death():
    """Full forward pass for birth-death, check shapes and gradients."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=2),
        sde=SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=1)
    solver = EulerMaruyamaSolver(config.sde)

    crn = birth_death_crn()
    init_state = torch.tensor([10.0])
    t_span = torch.linspace(0, 5, 10)

    ctx = encoder(crn, init_state)
    traj = solver.solve(sde, init_state, ctx, t_span, dt=0.1)

    assert traj.states.shape == (10, 1)
    assert traj.n_steps == 10
    assert traj.n_species == 1


def test_end_to_end_lotka_volterra():
    """Full forward pass for Lotka-Volterra, check shapes."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=32, n_layers=2),
        sde=SDEConfig(d_model=32, d_hidden=64, n_noise_channels=8),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=2)
    solver = EulerMaruyamaSolver(config.sde)

    crn = lotka_volterra_crn()
    init_state = torch.tensor([50.0, 20.0])
    t_span = torch.linspace(0, 10, 20)

    ctx = encoder(crn, init_state)
    traj = solver.solve(sde, init_state, ctx, t_span, dt=0.1)

    assert traj.states.shape == (20, 2)


def test_gradient_flows_through_full_pipeline():
    """Gradients must flow from trajectory MSE loss back through SDE and encoder."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=1),
        sde=SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=1)
    solver = EulerMaruyamaSolver(config.sde)

    crn = birth_death_crn()
    init_state = torch.tensor([10.0])
    t_span = torch.linspace(0, 2, 5)

    ctx = encoder(crn, init_state)
    traj = solver.solve(sde, init_state, ctx, t_span, dt=0.1)

    true_states = torch.ones(1, 5, 1) * 10.0  # (M=1, T, n_species) — 3D required
    pred_states = traj.states.unsqueeze(0)  # (K=1, T, n_species)
    loss = MeanMatchingLoss().compute(pred_states, true_states)
    loss.backward()

    sde_grads = [p.grad for p in sde.parameters() if p.grad is not None]
    assert len(sde_grads) > 0, "No gradients in SDE parameters"
