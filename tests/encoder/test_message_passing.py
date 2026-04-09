"""Tests for SumMessagePassingLayer and AttentiveMessagePassingLayer."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.encoder.message_passing import (
    AttentiveMessagePassingLayer,
    SumMessagePassingLayer,
)
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr

D_MODEL = 32

_LAYER_CLASSES = [SumMessagePassingLayer, AttentiveMessagePassingLayer]


@pytest.fixture(params=_LAYER_CLASSES, ids=["sum", "attentive"])
def layer(request):
    return request.param(D_MODEL)


def test_output_shapes(layer, birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    n_species = birth_death_repr.n_species
    n_reactions = birth_death_repr.n_reactions

    h_species = torch.randn(n_species, D_MODEL)
    h_reactions = torch.randn(n_reactions, D_MODEL)

    out_species, out_reactions = layer(h_species, h_reactions, edges)
    assert out_species.shape == (n_species, D_MODEL)
    assert out_reactions.shape == (n_reactions, D_MODEL)


def test_residual_connection(layer, birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    h_species = torch.randn(birth_death_repr.n_species, D_MODEL)
    h_reactions = torch.randn(birth_death_repr.n_reactions, D_MODEL)

    out_species, out_reactions = layer(h_species, h_reactions, edges)

    # Output should differ from input (embeddings are updated)
    assert not torch.allclose(out_species, h_species)
    assert not torch.allclose(out_reactions, h_reactions)


def test_message_passing_changes_embeddings(
    layer, birth_death_repr: CRNTensorRepr
) -> None:
    edges = birth_death_repr.bipartite_edges
    h_species = torch.randn(birth_death_repr.n_species, D_MODEL)
    h_reactions = torch.randn(birth_death_repr.n_reactions, D_MODEL)

    out_species, _ = layer(h_species, h_reactions, edges)
    assert not torch.allclose(out_species, h_species)


# ── Attention-specific ────────────────────────────────────────────────────────


def test_attention_produces_different_result_from_sum(
    birth_death_repr: CRNTensorRepr,
) -> None:
    edges = birth_death_repr.bipartite_edges
    h_species = torch.randn(birth_death_repr.n_species, D_MODEL)
    h_reactions = torch.randn(birth_death_repr.n_reactions, D_MODEL)

    sum_layer = SumMessagePassingLayer(D_MODEL)
    att_layer = AttentiveMessagePassingLayer(D_MODEL)

    out_sum, _ = sum_layer(h_species, h_reactions, edges)
    out_att, _ = att_layer(h_species, h_reactions, edges)

    # Different architectures with different random init should produce different outputs
    assert not torch.allclose(out_sum, out_att)
