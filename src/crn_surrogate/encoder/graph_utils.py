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
    is_external: torch.Tensor | None = None,
) -> BipartiteEdges:
    """Extract bipartite edges from stoichiometry and dependency matrices.

    An edge exists between reaction r and species s wherever |S[r,s]| > 0
    (stoichiometric involvement) OR D[r,s] > 0 (propensity dependency,
    e.g. a catalytic enzyme).

    For externally controlled species (is_external[s] == True):
    - Species-to-reaction edges ARE created: external species influence reaction
      propensities via dependency edges.
    - Reaction-to-species edges are NOT created: nothing flows back into external
      species (their dynamics are prescribed by the input protocol, not CRN kinetics).

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        dependency_matrix: (n_reactions, n_species) binary dependency matrix.
        is_external: (n_species,) boolean tensor; True for externally controlled
            species. Defaults to all-False if not provided.

    Returns:
        BipartiteEdges with EDGE_FEAT_DIM-dimensional features for both
        message directions.
    """
    n_reactions, n_species = stoichiometry.shape
    if is_external is None:
        is_external = torch.zeros(n_species, dtype=torch.bool, device=stoichiometry.device)

    # All edges (both directions use the same connectivity mask)
    mask = (stoichiometry.abs() > 0) | (dependency_matrix > 0)
    rxn_idx, species_idx = mask.nonzero(as_tuple=True)

    def _edge_features(r_idx: torch.Tensor, s_idx: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = [torch.empty(0)] * len(EdgeFeature)
        features[EdgeFeature.NET_CHANGE] = stoichiometry[r_idx, s_idx].float()
        features[EdgeFeature.IS_STOICHIOMETRIC] = (
            stoichiometry[r_idx, s_idx].abs() > 0
        ).float()
        features[EdgeFeature.IS_DEPENDENCY] = (
            dependency_matrix[r_idx, s_idx] > 0
        ).float()
        feat = torch.stack(features, dim=1)  # (E, EDGE_FEAT_DIM)
        assert feat.shape[1] == EDGE_FEAT_DIM
        return feat

    # Species-to-reaction: keep all edges (external species may affect propensities)
    s2r_feat = _edge_features(rxn_idx, species_idx)

    # Reaction-to-species: exclude edges where the target species is external
    internal_mask = ~is_external[species_idx]
    r2s_rxn_idx = rxn_idx[internal_mask]
    r2s_species_idx = species_idx[internal_mask]
    r2s_feat = _edge_features(r2s_rxn_idx, r2s_species_idx)

    return BipartiteEdges(
        rxn_to_species_index=torch.stack([r2s_rxn_idx, r2s_species_idx], dim=0),
        rxn_to_species_feat=r2s_feat,
        species_to_rxn_index=torch.stack([species_idx, rxn_idx], dim=0),
        species_to_rxn_feat=s2r_feat,
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
