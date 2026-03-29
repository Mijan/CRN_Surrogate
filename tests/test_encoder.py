"""Tests for the bipartite GNN encoder."""

import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.data.gillespie import birth_death_crn, lotka_volterra_crn
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder


def test_encoder_birth_death_shapes():
    config = EncoderConfig(d_model=16, n_layers=2)
    encoder = BipartiteGNNEncoder(config)
    crn = birth_death_crn()
    initial_state = torch.tensor([5.0])

    ctx = encoder(crn, initial_state)
    assert ctx.species_embeddings.shape == (1, 16)
    assert ctx.reaction_embeddings.shape == (2, 16)
    assert ctx.context_vector.shape == (32,)


def test_encoder_lotka_volterra_shapes():
    config = EncoderConfig(d_model=32, n_layers=2)
    encoder = BipartiteGNNEncoder(config)
    crn = lotka_volterra_crn()
    initial_state = torch.tensor([50.0, 20.0])

    ctx = encoder(crn, initial_state)
    assert ctx.species_embeddings.shape == (2, 32)
    assert ctx.reaction_embeddings.shape == (3, 32)
    assert ctx.context_vector.shape == (64,)


def test_encoder_gradients_flow():
    config = EncoderConfig(d_model=16, n_layers=2)
    encoder = BipartiteGNNEncoder(config)
    crn = birth_death_crn()
    initial_state = torch.tensor([5.0])

    ctx = encoder(crn, initial_state)
    loss = ctx.context_vector.sum()
    loss.backward()

    for p in encoder.parameters():
        assert p.grad is not None
