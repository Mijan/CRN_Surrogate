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
    HillActivationRepressionParams,
    HillParams,
    HillRepressionParams,
    MassActionParams,
    SubstrateInhibitionParams,
)
from crn_surrogate.encoder.graph_utils import BipartiteEdges, BipartiteGraphBuilder

if TYPE_CHECKING:
    from crn_surrogate.crn.crn import CRN


class PropensityType(Enum):
    """Integer type identifier for propensity functions in the tensor representation."""

    MASS_ACTION = 0
    HILL = 1
    CONSTANT_RATE = 2
    ENZYME_MICHAELIS_MENTEN = 3
    HILL_REPRESSION = 4
    HILL_ACTIVATION_REPRESSION = 5
    SUBSTRATE_INHIBITION = 6


# Registry mapping parameter dataclass type → PropensityType integer value.
# To support a new propensity type, add its Params class here and handle it
# in tensor_repr_to_crn.
_PARAMS_TO_TYPE_ID: dict[type, int] = {
    MassActionParams: PropensityType.MASS_ACTION.value,
    HillParams: PropensityType.HILL.value,
    ConstantRateParams: PropensityType.CONSTANT_RATE.value,
    EnzymeMichaelisMentenParams: PropensityType.ENZYME_MICHAELIS_MENTEN.value,
    HillRepressionParams: PropensityType.HILL_REPRESSION.value,
    HillActivationRepressionParams: PropensityType.HILL_ACTIVATION_REPRESSION.value,
    SubstrateInhibitionParams: PropensityType.SUBSTRATE_INHIBITION.value,
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
        is_external: (n_species,) boolean tensor; True for externally controlled
            species. External species receive no reaction-to-species messages
            in the bipartite graph (nothing flows back into them).
    """

    stoichiometry: torch.Tensor
    dependency_matrix: torch.Tensor
    propensity_type_ids: torch.Tensor
    propensity_params: torch.Tensor
    # Default None for backward compatibility; __post_init__ sets all-False tensor
    is_external: torch.Tensor | None = None
    species_names: tuple[str, ...] = ()
    reaction_names: tuple[str, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        # Default is_external to all-False if not provided
        if self.is_external is None:
            object.__setattr__(
                self,
                "is_external",
                torch.zeros(self.stoichiometry.shape[1], dtype=torch.bool),
            )

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
            edges = BipartiteGraphBuilder(
                self.stoichiometry, self.dependency_matrix, self.is_external
            ).build()
            object.__setattr__(self, "_cached_edges", edges)
        return self._cached_edges  # type: ignore[attr-defined]

    def to(self, device: torch.device) -> CRNTensorRepr:
        """Return a copy with all tensors moved to the given device.

        The cached bipartite edges are NOT carried over; they are lazily
        recomputed on the new device when first accessed.

        Returns self if already on the target device (avoids unnecessary copies).
        """
        if self.stoichiometry.device == device:
            return self
        return CRNTensorRepr(
            stoichiometry=self.stoichiometry.to(device),
            dependency_matrix=self.dependency_matrix.to(device),
            propensity_type_ids=self.propensity_type_ids.to(device),
            propensity_params=self.propensity_params.to(device),
            is_external=(
                self.is_external.to(device) if self.is_external is not None else None
            ),
            species_names=self.species_names,
            reaction_names=self.reaction_names,
            name=self.name,
        )


def crn_to_tensor_repr(crn: "CRN", max_params: int = 8) -> CRNTensorRepr:
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

    is_external = torch.tensor(crn.is_external, dtype=torch.bool)
    return CRNTensorRepr(
        stoichiometry=crn.stoichiometry_matrix,
        dependency_matrix=torch.stack(dep_rows, dim=0),
        propensity_type_ids=torch.tensor(type_id_rows, dtype=torch.long),
        propensity_params=torch.stack(param_rows, dim=0),
        is_external=is_external,
        species_names=crn.species_names,
        reaction_names=tuple(rxn.name for rxn in crn.reactions),
        name=crn.name if hasattr(crn, "name") else "",
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
        HillActivationRepressionParams,
        HillRepressionParams,
        SubstrateInhibitionParams,
        constant_rate,
        enzyme_michaelis_menten,
        hill,
        hill_activation_repression,
        hill_repression,
        mass_action,
        substrate_inhibition,
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
        elif type_id == PropensityType.HILL_REPRESSION.value:
            hr_params = HillRepressionParams.from_tensor(params)
            propensity_fn = hill_repression(
                k_max=hr_params.k_max,
                k_half=hr_params.k_half,
                hill_coefficient=hr_params.hill_coefficient,
                species_index=hr_params.species_index,
            )
        elif type_id == PropensityType.HILL_ACTIVATION_REPRESSION.value:
            har_params = HillActivationRepressionParams.from_tensor(params)
            propensity_fn = hill_activation_repression(
                k_max=har_params.k_max,
                k_act=har_params.k_act,
                n_act=har_params.n_act,
                activator_index=har_params.activator_index,
                k_rep=har_params.k_rep,
                n_rep=har_params.n_rep,
                repressor_index=har_params.repressor_index,
            )
        elif type_id == PropensityType.SUBSTRATE_INHIBITION.value:
            si_params = SubstrateInhibitionParams.from_tensor(params)
            propensity_fn = substrate_inhibition(
                v_max=si_params.v_max,
                k_m=si_params.k_m,
                k_i=si_params.k_i,
                species_index=si_params.species_index,
            )
        else:
            raise ValueError(
                f"Unknown propensity type ID {type_id} at reaction {r}. "
                f"Known IDs: {[pt.value for pt in PropensityType]}"
            )

        reactions.append(Reaction(stoichiometry=stoich, propensity=propensity_fn))

    return CRN(reactions=reactions)
