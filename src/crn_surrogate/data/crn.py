from __future__ import annotations
from dataclasses import dataclass, field
import torch
from crn_surrogate.data.propensities import PropensityType


@dataclass(frozen=True)
class CRNDefinition:
    """Immutable definition of a Chemical Reaction Network.

    Attributes:
        stoichiometry: Net change matrix, shape (n_reactions, n_species).
            stoichiometry[r, s] is the net change in species s when reaction r fires.
        reactant_matrix: Consumption matrix, shape (n_reactions, n_species).
            reactant_matrix[r, s] is the number of molecules of species s consumed
            by reaction r.
        propensity_types: One PropensityType per reaction.
        propensity_params: Shape (n_reactions, max_params), padded with zeros.
        species_names: Optional tuple of species name strings for debugging.
    """

    stoichiometry: torch.Tensor
    reactant_matrix: torch.Tensor
    propensity_types: tuple[PropensityType, ...]
    propensity_params: torch.Tensor
    species_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_species(self) -> int:
        """Number of species in the CRN."""
        return self.stoichiometry.shape[1]

    @property
    def n_reactions(self) -> int:
        """Number of reactions in the CRN."""
        return self.stoichiometry.shape[0]

    def __repr__(self) -> str:
        return (
            f"CRNDefinition(n_species={self.n_species}, "
            f"n_reactions={self.n_reactions}, "
            f"species={self.species_names})"
        )


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

    rxn_to_species_index = torch.stack([rxn_idx, species_idx], dim=0)
    species_to_rxn_index = torch.stack([species_idx, rxn_idx], dim=0)

    return BipartiteEdges(
        rxn_to_species_index=rxn_to_species_index,
        rxn_to_species_feat=edge_feat,
        species_to_rxn_index=species_to_rxn_index,
        species_to_rxn_feat=edge_feat,
    )
