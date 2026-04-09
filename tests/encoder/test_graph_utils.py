"""Tests for BipartiteGraphBuilder, BipartiteEdges, EdgeFeature, merge_bipartite_edges."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.encoder.graph_utils import (
    BipartiteGraphBuilder,
    EdgeFeature,
    merge_bipartite_edges,
)
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr

# ── BipartiteGraphBuilder ─────────────────────────────────────────────────────


def test_birth_death_edge_count(birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    # Birth: stoich +1 for species 0 → 1 edge
    # Death: stoich -1 and dependency on species 0 → 1 edge
    # Both reactions connect to species 0 → 2 edges total in rxn_to_species
    assert edges.rxn_to_species_index.shape[1] == 2
    assert edges.species_to_rxn_index.shape[1] == 2


def test_edge_feature_channels(birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    # rxn_to_species has shape (2, E); features have shape (E, 3)
    assert edges.rxn_to_species_feat.shape[1] == 3

    # Find the birth reaction edge (reaction 0, species 0): stoich=+1, no dependency
    rxn_indices = edges.rxn_to_species_index[0]
    birth_edge_mask = rxn_indices == 0
    birth_feats = edges.rxn_to_species_feat[birth_edge_mask]
    assert birth_feats.shape[0] == 1
    feat = birth_feats[0]
    assert feat[EdgeFeature.NET_CHANGE].item() == pytest.approx(1.0)
    assert feat[EdgeFeature.IS_STOICHIOMETRIC].item() == pytest.approx(1.0)
    assert feat[EdgeFeature.IS_DEPENDENCY].item() == pytest.approx(0.0)


def test_external_species_no_incoming_edges() -> None:
    """External species excluded from rxn_to_species but included in species_to_rxn."""
    # 1 reaction, 2 species; species 0 is external
    stoich = torch.tensor([[1.0, -1.0]])  # (1, 2)
    dep = torch.tensor([[1.0, 1.0]])  # (1, 2)
    is_external = torch.tensor([True, False])

    edges = BipartiteGraphBuilder(stoich, dep, is_external).build()

    # rxn_to_species: species 0 is external → excluded; only species 1
    r2s_species = edges.rxn_to_species_index[1]
    assert 0 not in r2s_species.tolist()
    assert 1 in r2s_species.tolist()

    # species_to_rxn: species 0 still sends messages
    s2r_species = edges.species_to_rxn_index[0]
    assert 0 in s2r_species.tolist()


# ── merge_bipartite_edges ─────────────────────────────────────────────────────


def test_merge_two_graphs_offsets(birth_death_repr: CRNTensorRepr) -> None:
    edges1 = birth_death_repr.bipartite_edges
    edges2 = birth_death_repr.bipartite_edges
    ns1 = birth_death_repr.n_species
    nr1 = birth_death_repr.n_reactions

    merged = merge_bipartite_edges([edges1, edges2], [ns1, ns1], [nr1, nr1])

    # Second graph reaction indices should be offset by nr1
    r2s_rxn = merged.rxn_to_species_index[0]
    assert r2s_rxn.max().item() >= nr1

    # Second graph species indices should be offset by ns1
    r2s_spe = merged.rxn_to_species_index[1]
    assert r2s_spe.max().item() >= ns1


def test_merge_preserves_features(birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    merged = merge_bipartite_edges(
        [edges, edges],
        [birth_death_repr.n_species, birth_death_repr.n_species],
        [birth_death_repr.n_reactions, birth_death_repr.n_reactions],
    )
    # Features should be the original features repeated twice
    E = edges.rxn_to_species_feat.shape[0]
    assert merged.rxn_to_species_feat.shape[0] == 2 * E
    # First E rows should match original
    assert torch.allclose(merged.rxn_to_species_feat[:E], edges.rxn_to_species_feat)


def test_merge_total_edge_count(birth_death_repr: CRNTensorRepr) -> None:
    edges = birth_death_repr.bipartite_edges
    E1 = edges.rxn_to_species_index.shape[1]
    merged = merge_bipartite_edges(
        [edges, edges],
        [birth_death_repr.n_species, birth_death_repr.n_species],
        [birth_death_repr.n_reactions, birth_death_repr.n_reactions],
    )
    assert merged.rxn_to_species_index.shape[1] == 2 * E1
