"""Tests verifying the v4 state-independent encoder invariants."""

from __future__ import annotations

import inspect

import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.embeddings import SpeciesEmbedding
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr


def test_encoder_output_independent_of_initial_state():
    """Same CRN with repeated calls produces identical context in eval mode."""
    config = EncoderConfig(d_model=32, n_layers=2)
    encoder = BipartiteGNNEncoder(config)
    encoder.eval()

    crn_repr = crn_to_tensor_repr(birth_death(k_birth=2.0, k_death=0.5))

    with torch.no_grad():
        ctx = encoder(crn_repr)
        ctx2 = encoder(crn_repr)

    assert torch.equal(ctx.context_vector, ctx2.context_vector)


def test_species_embedding_no_concentration_input():
    """SpeciesEmbedding forward takes n_species: int, not a concentration tensor."""
    config = EncoderConfig(d_model=32)
    embed = SpeciesEmbedding(config)
    h = embed(n_species=3)
    assert h.shape == (3, 32)


def test_species_embedding_no_conc_proj():
    """SpeciesEmbedding should not have a concentration projection layer."""
    config = EncoderConfig(d_model=32)
    embed = SpeciesEmbedding(config)
    assert not hasattr(embed, "_conc_proj")


def test_encoder_forward_signature_no_initial_state():
    """Encoder forward takes only crn_repr; initial_state must not be a parameter."""
    config = EncoderConfig(d_model=32, n_layers=2)
    encoder = BipartiteGNNEncoder(config)

    crn_repr = crn_to_tensor_repr(birth_death(k_birth=2.0, k_death=0.5))
    ctx = encoder(crn_repr)
    assert ctx.context_vector.shape == (2 * 32,)

    sig = inspect.signature(encoder.forward)
    params = list(sig.parameters.keys())
    assert "initial_state" not in params


def test_encoder_different_crns_produce_different_contexts():
    """Two structurally distinct CRNs must produce different context vectors."""
    from crn_surrogate.data.generation.reference_crns import lotka_volterra

    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=32, n_layers=2))
    encoder.eval()

    repr_bd = crn_to_tensor_repr(birth_death(k_birth=2.0, k_death=0.5))
    repr_lv = crn_to_tensor_repr(lotka_volterra())

    with torch.no_grad():
        ctx_bd = encoder(repr_bd)
        ctx_lv = encoder(repr_lv)

    assert ctx_bd.context_vector.shape == (2 * 32,)
    assert ctx_lv.context_vector.shape == (2 * 32,)
    assert not torch.allclose(ctx_bd.context_vector, ctx_lv.context_vector), (
        "Encoder produced identical context for birth-death and Lotka-Volterra"
    )
