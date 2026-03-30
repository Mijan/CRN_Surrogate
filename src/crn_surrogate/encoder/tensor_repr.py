"""Conversion between symbolic CRN objects and flat tensor representations.

This module defines the boundary between the symbolic CRN world and the tensor
world consumed by the neural network encoder.

Flow: symbolic CRN → CRNTensorRepr → BipartiteGNNEncoder → embeddings.

Also contains BipartiteEdges and build_bipartite_edges, which are derived from
CRNTensorRepr and used by the message-passing encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from crn_surrogate.crn.propensities import (
    HillParams,
    MassActionParams,
    _HillClosure,
    _MassActionClosure,
)

if TYPE_CHECKING:
    from crn_surrogate.crn.crn import CRN


class PropensityType(Enum):
    """Integer type identifier for propensity functions in the tensor representation."""

    MASS_ACTION = 0
    HILL = 1


@dataclass(frozen=True)
class CRNTensorRepr:
    """Flat tensor representation of a CRN for neural network input.

    All arrays are aligned on the reaction axis (row = reaction).

    Attributes:
        stoichiometry: (n_reactions, n_species) net change matrix.
        reactant_matrix: (n_reactions, n_species) consumption matrix.
        propensity_type_ids: (n_reactions,) integer type IDs (see PropensityType).
        propensity_params: (n_reactions, max_params) kinetic parameters.
    """

    stoichiometry: torch.Tensor
    reactant_matrix: torch.Tensor
    propensity_type_ids: torch.Tensor
    propensity_params: torch.Tensor

    @property
    def n_species(self) -> int:
        """Number of species."""
        return int(self.stoichiometry.shape[1])

    @property
    def n_reactions(self) -> int:
        """Number of reactions."""
        return int(self.stoichiometry.shape[0])


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


def crn_to_tensor_repr(crn: "CRN", max_params: int = 4) -> CRNTensorRepr:
    """Convert a CRN to its flat tensor representation.

    Only propensities that implement the SerializablePropensity protocol
    (i.e. _MassActionClosure and _HillClosure) are supported. Custom callables
    must be wrapped in an inspectable class with a .params property.

    Args:
        crn: The CRN to convert.
        max_params: Length of the parameter row for each reaction.

    Returns:
        CRNTensorRepr suitable for the BipartiteGNNEncoder.

    Raises:
        ValueError: If any propensity type is not serializable.
    """
    n_reactions = crn.n_reactions
    n_species = crn.n_species

    stoichiometry = crn.stoichiometry_matrix  # (n_reactions, n_species)
    reactant_rows = []
    type_id_rows = []
    param_rows = []

    for r, rxn in enumerate(crn.reactions):
        prop = rxn.propensity
        if isinstance(prop, _MassActionClosure):
            type_id_rows.append(PropensityType.MASS_ACTION.value)
            param_rows.append(prop.params.to_tensor(max_params))
            reactant_rows.append(prop.params.reactant_stoichiometry.float())
        elif isinstance(prop, _HillClosure):
            type_id_rows.append(PropensityType.HILL.value)
            param_rows.append(prop.params.to_tensor(max_params))
            reactant_rows.append(torch.zeros(n_species))
        else:
            raise ValueError(
                f"Reaction {r} ('{rxn.name}') has a non-serializable propensity "
                f"of type {type(prop).__name__}. To serialize, wrap it in a class "
                f"with a .params property implementing to_tensor() / from_tensor()."
            )

    return CRNTensorRepr(
        stoichiometry=stoichiometry,
        reactant_matrix=torch.stack(reactant_rows, dim=0),
        propensity_type_ids=torch.tensor(type_id_rows, dtype=torch.long),
        propensity_params=torch.stack(param_rows, dim=0),
    )


def tensor_repr_to_crn(tensor_repr: CRNTensorRepr) -> "CRN":
    """Reconstruct a CRN from its flat tensor representation.

    Args:
        tensor_repr: The tensor representation to reconstruct from.

    Returns:
        CRN with the same structure and kinetics as the original.

    Raises:
        ValueError: If an unknown propensity type ID is encountered.
    """
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import hill, mass_action
    from crn_surrogate.crn.reaction import Reaction

    reactions = []
    for r in range(tensor_repr.n_reactions):
        type_id = int(tensor_repr.propensity_type_ids[r].item())
        params = tensor_repr.propensity_params[r]
        reactant_row = tensor_repr.reactant_matrix[r]
        stoich = tensor_repr.stoichiometry[r]

        if type_id == PropensityType.MASS_ACTION.value:
            ma_params = MassActionParams.from_tensor(params, reactant_row)
            propensity_fn = mass_action(
                rate_constant=ma_params.rate_constant,
                reactant_stoichiometry=ma_params.reactant_stoichiometry,
            )
        elif type_id == PropensityType.HILL.value:
            hill_params = HillParams.from_tensor(params)
            propensity_fn = hill(
                v_max=hill_params.v_max,
                k_m=hill_params.k_m,
                hill_coefficient=hill_params.hill_coefficient,
                species_index=hill_params.species_index,
            )
        else:
            raise ValueError(
                f"Unknown propensity type ID {type_id} at reaction {r}. "
                f"Known IDs: {[pt.value for pt in PropensityType]}"
            )

        reactions.append(Reaction(stoichiometry=stoich, propensity=propensity_fn))

    return CRN(reactions=reactions)
