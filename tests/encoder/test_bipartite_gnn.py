"""Tests for BipartiteGNNEncoder end-to-end."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr


def test_forward_output_type(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config)
    result = encoder(birth_death_repr)
    assert isinstance(result, CRNContext)


def test_context_vector_shape(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config)
    ctx = encoder(birth_death_repr)
    assert ctx.context_vector.shape == (2 * small_encoder_config.d_model,)


def test_species_embeddings_shape(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config)
    ctx = encoder(birth_death_repr)
    assert ctx.species_embeddings.shape == (
        birth_death_repr.n_species,
        small_encoder_config.d_model,
    )


def test_reaction_embeddings_shape(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config)
    ctx = encoder(birth_death_repr)
    assert ctx.reaction_embeddings.shape == (
        birth_death_repr.n_reactions,
        small_encoder_config.d_model,
    )


def test_same_crn_same_context(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config).eval()
    with torch.no_grad():
        ctx1 = encoder(birth_death_repr)
        ctx2 = encoder(birth_death_repr)
    assert torch.allclose(ctx1.context_vector, ctx2.context_vector)


def test_different_crns_different_contexts(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
    two_species_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config).eval()
    with torch.no_grad():
        ctx_bd = encoder(birth_death_repr)
        ctx_ts = encoder(two_species_repr)
    # Context vectors have different dimensions (different n_species, n_reactions)
    # but we can check they differ in pooled representation; just confirm different shapes or values
    # Both have d_model=32 → context_vector is 64-d for both
    assert not torch.allclose(ctx_bd.context_vector, ctx_ts.context_vector)


def test_forward_batch_output_count(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
    two_species_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config).eval()
    with torch.no_grad():
        results = encoder.forward_batch(
            [birth_death_repr, birth_death_repr, two_species_repr]
        )
    assert len(results) == 3


def test_forward_batch_matches_sequential(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config).eval()
    reprs = [birth_death_repr, birth_death_repr, birth_death_repr]

    with torch.no_grad():
        batch_results = encoder.forward_batch(reprs)
        sequential_results = [encoder(r) for r in reprs]

    for batch_ctx, seq_ctx in zip(batch_results, sequential_results):
        assert torch.allclose(
            batch_ctx.context_vector, seq_ctx.context_vector, atol=1e-5
        )


def test_gradients_flow(
    small_encoder_config: EncoderConfig,
    birth_death_repr: CRNTensorRepr,
) -> None:
    encoder = BipartiteGNNEncoder(small_encoder_config)
    ctx = encoder(birth_death_repr)
    ctx.context_vector.sum().backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in encoder.parameters()
    )
    assert has_grad
