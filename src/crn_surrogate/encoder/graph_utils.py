"""Bipartite graph edge construction for species-reaction message passing.

Provides BipartiteEdges and build_bipartite_edges, used by the
message-passing encoder to represent species-reaction connectivity.
These utilities are derived from CRNTensorRepr fields but have no
knowledge of CRN serialization or propensity types.

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

import torch


@dataclass(frozen=True)
class BipartiteEdges:
    """Edge indices and features for bipartite message passing.

    Each edge (reaction r, species s) has a 3-dimensional feature:
        [net_change, is_stoichiometric, is_dependency]

    - net_change: S[r, s] (float, can be negative)
    - is_stoichiometric: 1.0 if |S[r, s]| > 0, else 0.0
    - is_dependency: 1.0 if D[r, s] > 0, else 0.0

    A catalytic species has is_stoichiometric=0, is_dependency=1.
    A consumed/produced species has is_stoichiometric=1.

    Attributes:
        rxn_to_species_index: (2, E) — row 0: reaction indices, row 1: species indices.
        rxn_to_species_feat: (E, 3) — [net_change, is_stoichiometric, is_dependency].
        species_to_rxn_index: (2, E) — row 0: species indices, row 1: reaction indices.
        species_to_rxn_feat: (E, 3) — [net_change, is_stoichiometric, is_dependency].
    """

    rxn_to_species_index: torch.Tensor
    rxn_to_species_feat: torch.Tensor
    species_to_rxn_index: torch.Tensor
    species_to_rxn_feat: torch.Tensor


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
        BipartiteEdges with 3D features for both message directions.
    """
    mask = (stoichiometry.abs() > 0) | (dependency_matrix > 0)
    rxn_idx, species_idx = mask.nonzero(as_tuple=True)

    net_changes = stoichiometry[rxn_idx, species_idx].float().unsqueeze(1)
    is_stoich = (stoichiometry[rxn_idx, species_idx].abs() > 0).float().unsqueeze(1)
    is_dep = (dependency_matrix[rxn_idx, species_idx] > 0).float().unsqueeze(1)
    edge_feat = torch.cat([net_changes, is_stoich, is_dep], dim=1)  # (E, 3)

    return BipartiteEdges(
        rxn_to_species_index=torch.stack([rxn_idx, species_idx], dim=0),
        rxn_to_species_feat=edge_feat,
        species_to_rxn_index=torch.stack([species_idx, rxn_idx], dim=0),
        species_to_rxn_feat=edge_feat,
    )
