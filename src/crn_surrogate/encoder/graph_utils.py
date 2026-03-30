"""Bipartite graph edge construction for species-reaction message passing.

Provides BipartiteEdges, EdgeFeature, EDGE_FEAT_DIM, build_bipartite_edges,
and _scatter_max, used by the message-passing encoder to represent
species-reaction connectivity.

Design decision: no reaction-type embedding
-------------------------------------------
We deliberately omit explicit reaction-type annotations (e.g., "phosphorylation",
"degradation") from the encoder. The information is available through structure:
stoichiometry patterns and the dependency/stoichiometry edge-feature flags give
the GNN enough signal to distinguish reaction roles without hard-coded labels.
Adding category labels would couple the CRN representation to domain-specific
knowledge and risk the model short-cutting structural learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class EdgeFeature(IntEnum):
    """Edge feature channels for the bipartite species-reaction graph.

    This enum is the single source of truth for the edge feature schema.
    Both build_bipartite_edges (which constructs features) and
    MessagePassingLayer (which consumes them) derive their dimensions
    from this definition.

    To add a new edge feature:
    1. Add an entry here.
    2. Compute it in build_bipartite_edges.
    That's it. Everything else adapts automatically.
    """

    NET_CHANGE = 0
    IS_STOICHIOMETRIC = 1
    IS_DEPENDENCY = 2


EDGE_FEAT_DIM: int = len(EdgeFeature)


@dataclass(frozen=True)
class BipartiteEdges:
    """Edge indices and features for bipartite message passing.

    Each edge (reaction r, species s) has a feature vector of dimension
    EDGE_FEAT_DIM with channels defined by EdgeFeature:
        [NET_CHANGE, IS_STOICHIOMETRIC, IS_DEPENDENCY]

    - NET_CHANGE: S[r, s] (float, can be negative)
    - IS_STOICHIOMETRIC: 1.0 if |S[r, s]| > 0, else 0.0
    - IS_DEPENDENCY: 1.0 if D[r, s] > 0, else 0.0

    A catalytic species has IS_STOICHIOMETRIC=0, IS_DEPENDENCY=1.
    A consumed/produced species has IS_STOICHIOMETRIC=1.

    Attributes:
        rxn_to_species_index: (2, E) — row 0: reaction indices, row 1: species indices.
        rxn_to_species_feat: (E, EDGE_FEAT_DIM) edge features.
        species_to_rxn_index: (2, E) — row 0: species indices, row 1: reaction indices.
        species_to_rxn_feat: (E, EDGE_FEAT_DIM) edge features.
    """

    rxn_to_species_index: torch.Tensor
    rxn_to_species_feat: torch.Tensor
    species_to_rxn_index: torch.Tensor
    species_to_rxn_feat: torch.Tensor

    @property
    def edge_feat_dim(self) -> int:
        """Dimension of edge feature vectors."""
        return int(self.rxn_to_species_feat.shape[-1])


def build_bipartite_edges(
    stoichiometry: torch.Tensor,
    dependency_matrix: torch.Tensor,
) -> BipartiteEdges:
    """Extract bipartite edges from stoichiometry and dependency matrices.

    An edge exists between reaction r and species s wherever |S[r,s]| > 0
    (stoichiometric involvement) OR D[r,s] > 0 (propensity dependency,
    e.g. a catalytic enzyme).

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        dependency_matrix: (n_reactions, n_species) binary dependency matrix.

    Returns:
        BipartiteEdges with EDGE_FEAT_DIM-dimensional features for both
        message directions.
    """
    mask = (stoichiometry.abs() > 0) | (dependency_matrix > 0)
    rxn_idx, species_idx = mask.nonzero(as_tuple=True)

    features: list[torch.Tensor] = [torch.empty(0)] * len(EdgeFeature)
    features[EdgeFeature.NET_CHANGE] = stoichiometry[rxn_idx, species_idx].float()
    features[EdgeFeature.IS_STOICHIOMETRIC] = (
        stoichiometry[rxn_idx, species_idx].abs() > 0
    ).float()
    features[EdgeFeature.IS_DEPENDENCY] = (
        dependency_matrix[rxn_idx, species_idx] > 0
    ).float()

    edge_feat = torch.stack(features, dim=1)  # (E, EDGE_FEAT_DIM)
    assert edge_feat.shape[1] == EDGE_FEAT_DIM

    return BipartiteEdges(
        rxn_to_species_index=torch.stack([rxn_idx, species_idx], dim=0),
        rxn_to_species_feat=edge_feat,
        species_to_rxn_index=torch.stack([species_idx, rxn_idx], dim=0),
        species_to_rxn_feat=edge_feat,
    )


def _scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Compute per-group maximum for scatter softmax stability.

    Args:
        src: (E,) values.
        index: (E,) group assignments.
        num_groups: Number of groups.

    Returns:
        (num_groups,) per-group max values.
    """
    out = torch.full((num_groups,), float("-inf"), device=src.device, dtype=src.dtype)
    return out.scatter_reduce(0, index, src, reduce="amax")
