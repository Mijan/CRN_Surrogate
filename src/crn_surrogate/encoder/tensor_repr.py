"""Conversion between symbolic CRN objects and flat tensor representations.

This module defines the boundary between the symbolic CRN world and the tensor
world consumed by the neural network encoder.

Flow: symbolic CRN → CRNTensorRepr → BipartiteGNNEncoder → embeddings.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from crn_surrogate.crn.propensities import (
    ConstantRateParams,
    EnzymeMichaelisMentenParams,
    HillParams,
    MassActionParams,
)
from crn_surrogate.encoder.graph_utils import BipartiteEdges, build_bipartite_edges

if TYPE_CHECKING:
    from crn_surrogate.crn.crn import CRN


class PropensityType(Enum):
    """Integer type identifier for propensity functions in the tensor representation."""

    MASS_ACTION = 0
    HILL = 1
    CONSTANT_RATE = 2
    ENZYME_MICHAELIS_MENTEN = 3


# Registry mapping parameter dataclass type → PropensityType integer value.
# To support a new propensity type, add its Params class here and handle it
# in tensor_repr_to_crn.
_PARAMS_TO_TYPE_ID: dict[type, int] = {
    MassActionParams: PropensityType.MASS_ACTION.value,
    HillParams: PropensityType.HILL.value,
    ConstantRateParams: PropensityType.CONSTANT_RATE.value,
    EnzymeMichaelisMentenParams: PropensityType.ENZYME_MICHAELIS_MENTEN.value,
}


@dataclass(frozen=True)
class CRNTensorRepr:
    """Flat tensor representation of a CRN for neural network input.

    All arrays are aligned on the reaction axis (row = reaction).

    Attributes:
        stoichiometry: (n_reactions, n_species) net change matrix.
        dependency_matrix: (n_reactions, n_species) binary matrix indicating
            which species influence each reaction's propensity.
        propensity_type_ids: (n_reactions,) integer type IDs (see PropensityType).
        propensity_params: (n_reactions, max_params) kinetic parameters.
    """

    stoichiometry: torch.Tensor
    dependency_matrix: torch.Tensor
    propensity_type_ids: torch.Tensor
    propensity_params: torch.Tensor
    species_names: tuple[str, ...] = ()
    reaction_names: tuple[str, ...] = ()
    name: str = ""

    @property
    def n_species(self) -> int:
        """Number of species."""
        return int(self.stoichiometry.shape[1])

    @property
    def n_reactions(self) -> int:
        """Number of reactions."""
        return int(self.stoichiometry.shape[0])

    @property
    def bipartite_edges(self) -> BipartiteEdges:
        """Lazily computed and cached bipartite edges for this CRN's graph structure."""
        if not hasattr(self, "_cached_edges"):
            edges = build_bipartite_edges(self.stoichiometry, self.dependency_matrix)
            object.__setattr__(self, "_cached_edges", edges)
        return self._cached_edges  # type: ignore[attr-defined]


def crn_to_tensor_repr(crn: "CRN", max_params: int = 4) -> CRNTensorRepr:
    """Convert a CRN to its flat tensor representation.

    Only propensities that expose a `.params` property whose type is
    registered in _PARAMS_TO_TYPE_ID are supported. Custom propensity
    callables must implement a `.params` property returning a registered
    Params dataclass.

    Propensities that also expose `.species_dependencies` are used to
    construct the dependency matrix. If `.species_dependencies` is absent,
    all species are assumed to be dependencies and a warning is issued.

    Args:
        crn: The CRN to convert.
        max_params: Length of the parameter row for each reaction.

    Returns:
        CRNTensorRepr suitable for the BipartiteGNNEncoder.

    Raises:
        ValueError: If any propensity type is not serializable or not registered.
    """
    type_id_rows: list[int] = []
    param_rows: list[torch.Tensor] = []
    dep_rows: list[torch.Tensor] = []

    for r, rxn in enumerate(crn.reactions):
        prop = rxn.propensity
        if not hasattr(prop, "params"):
            raise ValueError(
                f"Reaction {r} ('{rxn.name}') has a non-serializable propensity "
                f"of type {type(prop).__name__}. Wrap it in a callable class "
                f"with a .params property implementing to_tensor()."
            )
        params = prop.params
        param_type = type(params)
        if param_type not in _PARAMS_TO_TYPE_ID:
            raise ValueError(
                f"Reaction {r} ('{rxn.name}') has propensity params of type "
                f"{param_type.__name__}, which is not registered for "
                f"serialization. Known types: {list(_PARAMS_TO_TYPE_ID.keys())}"
            )
        type_id_rows.append(_PARAMS_TO_TYPE_ID[param_type])
        param_rows.append(params.to_tensor(max_params))

        dep_row = torch.zeros(crn.n_species)
        if hasattr(prop, "species_dependencies"):
            for s in prop.species_dependencies:
                dep_row[s] = 1.0
        else:
            warnings.warn(
                f"Reaction {r} ('{rxn.name}') propensity {type(prop).__name__!r} "
                f"does not declare species_dependencies; assuming all species.",
                stacklevel=2,
            )
            dep_row = torch.ones(crn.n_species)
        dep_rows.append(dep_row)

    return CRNTensorRepr(
        stoichiometry=crn.stoichiometry_matrix,
        dependency_matrix=torch.stack(dep_rows, dim=0),
        propensity_type_ids=torch.tensor(type_id_rows, dtype=torch.long),
        propensity_params=torch.stack(param_rows, dim=0),
        species_names=crn.species_names,
        reaction_names=tuple(rxn.name for rxn in crn.reactions),
        name=crn.name if hasattr(crn, 'name') else ""
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
    from crn_surrogate.crn.propensities import (
        constant_rate,
        enzyme_michaelis_menten,
        hill,
        mass_action,
    )
    from crn_surrogate.crn.reaction import Reaction

    reactions = []
    for r in range(tensor_repr.n_reactions):
        type_id = int(tensor_repr.propensity_type_ids[r].item())
        params = tensor_repr.propensity_params[r]
        stoich = tensor_repr.stoichiometry[r]

        if type_id == PropensityType.MASS_ACTION.value:
            k = MassActionParams.from_tensor(params).rate_constant
            # For standard (non-autocatalytic) mass-action, reactant stoichiometry
            # equals the negative part of the net stoichiometry.
            reactant_stoich = (-stoich).clamp(min=0).float()
            propensity_fn = mass_action(
                rate_constant=k,
                reactant_stoichiometry=reactant_stoich,
            )
        elif type_id == PropensityType.HILL.value:
            hill_params = HillParams.from_tensor(params)
            propensity_fn = hill(
                v_max=hill_params.v_max,
                k_m=hill_params.k_m,
                hill_coefficient=hill_params.hill_coefficient,
                species_index=hill_params.species_index,
            )
        elif type_id == PropensityType.CONSTANT_RATE.value:
            cr_params = ConstantRateParams.from_tensor(params)
            propensity_fn = constant_rate(k=cr_params.rate)
        elif type_id == PropensityType.ENZYME_MICHAELIS_MENTEN.value:
            emm_params = EnzymeMichaelisMentenParams.from_tensor(params)
            propensity_fn = enzyme_michaelis_menten(
                k_cat=emm_params.k_cat,
                k_m=emm_params.k_m,
                enzyme_index=emm_params.enzyme_index,
                substrate_index=emm_params.substrate_index,
            )
        else:
            raise ValueError(
                f"Unknown propensity type ID {type_id} at reaction {r}. "
                f"Known IDs: {[pt.value for pt in PropensityType]}"
            )

        reactions.append(Reaction(stoichiometry=stoich, propensity=propensity_fn))

    return CRN(reactions=reactions)
