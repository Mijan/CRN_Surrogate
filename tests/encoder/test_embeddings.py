"""Tests for SpeciesEmbedding and ReactionEmbedding."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.encoder.embeddings import ReactionEmbedding, SpeciesEmbedding

# ── SpeciesEmbedding ──────────────────────────────────────────────────────────


def test_species_embedding_output_shape(small_encoder_config: EncoderConfig) -> None:
    emb = SpeciesEmbedding(small_encoder_config)
    out = emb(n_species=3)
    assert out.shape == (3, small_encoder_config.d_model)


def test_different_species_different_embeddings(
    small_encoder_config: EncoderConfig,
) -> None:
    emb = SpeciesEmbedding(small_encoder_config)
    out = emb(n_species=3)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[0], out[2])


def test_external_flag_changes_embedding(small_encoder_config: EncoderConfig) -> None:
    emb = SpeciesEmbedding(small_encoder_config)
    is_external_false = torch.tensor([False])
    is_external_true = torch.tensor([True])
    out_internal = emb(n_species=1, is_external=is_external_false)
    out_external = emb(n_species=1, is_external=is_external_true)
    assert not torch.allclose(out_internal, out_external)


# ── ReactionEmbedding ─────────────────────────────────────────────────────────


def test_reaction_embedding_output_shape(small_encoder_config: EncoderConfig) -> None:
    emb = ReactionEmbedding(small_encoder_config)
    type_ids = torch.zeros(4, dtype=torch.long)
    params = torch.zeros(4, small_encoder_config.max_propensity_params)
    out = emb(type_ids, params)
    assert out.shape == (4, small_encoder_config.d_model)


def test_different_types_different_embeddings(
    small_encoder_config: EncoderConfig,
) -> None:
    emb = ReactionEmbedding(small_encoder_config)
    params = torch.zeros(2, small_encoder_config.max_propensity_params)
    type_ids = torch.tensor([0, 2], dtype=torch.long)  # MASS_ACTION vs CONSTANT_RATE
    out = emb(type_ids, params)
    assert not torch.allclose(out[0], out[1])
