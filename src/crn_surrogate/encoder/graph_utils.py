"""Bipartite graph edge construction for species-reaction message passing.

Provides BipartiteEdges and build_bipartite_edges, used by the
message-passing encoder to represent species-reaction connectivity.
These utilities are derived from CRNTensorRepr fields but have no
knowledge of CRN serialization or propensity types.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BipartiteEdges:
    """Edge indices and features for bipartite message passing.

    Attributes:
        rxn_to_species_index: (2, E) — row 0: reaction indices, row 1: species indices.
        rxn_to_species_feat: (E, 2) — [reactant_count, net_change] per edge.
        species_to_rxn_index: (2, E) — row 0: species indices, row 1: reaction indices.
        species_to_rxn_feat: (E, 2) — [reactant_count, net_change] per edge.
    """

    rxn_to_species_index: torch.Tensor
    rxn_to_species_feat: torch.Tensor
    species_to_rxn_index: torch.Tensor
    species_to_rxn_feat: torch.Tensor


def build_bipartite_edges(
    stoichiometry: torch.Tensor,
    reactant_matrix: torch.Tensor,
) -> BipartiteEdges:
    """Extract bipartite edges from stoichiometry and reactant matrices.

    An edge exists between reaction r and species s wherever |S[r,s]| > 0
    or R[r,s] > 0.

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        reactant_matrix: (n_reactions, n_species) consumption matrix.

    Returns:
        BipartiteEdges with indices and features for both message directions.
    """
    mask = (stoichiometry.abs() > 0) | (reactant_matrix > 0)
    rxn_idx, species_idx = mask.nonzero(as_tuple=True)

    reactant_counts = reactant_matrix[rxn_idx, species_idx].float().unsqueeze(1)
    net_changes = stoichiometry[rxn_idx, species_idx].float().unsqueeze(1)
    edge_feat = torch.cat([reactant_counts, net_changes], dim=1)

    return BipartiteEdges(
        rxn_to_species_index=torch.stack([rxn_idx, species_idx], dim=0),
        rxn_to_species_feat=edge_feat,
        species_to_rxn_index=torch.stack([species_idx, rxn_idx], dim=0),
        species_to_rxn_feat=edge_feat,
    )
