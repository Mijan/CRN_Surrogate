"""Tests for the neural SDE and Euler-Maruyama solver."""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.data.gillespie import birth_death_crn
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver


def _make_context(n_species: int = 1, d_model: int = 16):
    config = EncoderConfig(d_model=d_model, n_layers=1)
    encoder = BipartiteGNNEncoder(config)
    crn = birth_death_crn()
    init = torch.tensor([5.0] * n_species)
    return encoder(crn, init), config


def test_sde_drift_shape():
    ctx, config = _make_context(n_species=1, d_model=16)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    state = torch.tensor([5.0])
    drift = sde.drift(torch.tensor(0.0), state, ctx)
    assert drift.shape == (1,)


def test_sde_diffusion_shape():
    ctx, config = _make_context(n_species=1, d_model=16)
    n_noise = 4
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=n_noise), n_species=1
    )
    state = torch.tensor([5.0])
    diff = sde.diffusion(torch.tensor(0.0), state, ctx)
    assert diff.shape == (1, n_noise)
    assert (diff >= 0).all()


def test_euler_maruyama_trajectory_shape():
    d_model = 16
    ctx, config = _make_context(n_species=1, d_model=d_model)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=d_model, d_hidden=32, n_noise_channels=4, clip_state=True),
        n_species=1,
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=d_model, clip_state=True))

    t_span = torch.linspace(0, 5, 10)
    traj = solver.solve(sde, torch.tensor([5.0]), ctx, t_span, dt=0.1)
    assert traj.states.shape == (10, 1)


def test_euler_maruyama_nonnegative_with_clip():
    d_model = 16
    ctx, _ = _make_context(n_species=1, d_model=d_model)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=d_model, d_hidden=32, n_noise_channels=4, clip_state=True),
        n_species=1,
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=d_model, clip_state=True))

    t_span = torch.linspace(0, 10, 20)
    traj = solver.solve(sde, torch.tensor([0.0]), ctx, t_span, dt=0.1)
    assert (traj.states >= 0).all()
