"""CRN composition: merge two CRNs by identifying shared species."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motifs.base import MotifFactory


@dataclass(frozen=True)
class CompositionSpec:
    """Specification for coupling two CRNs by identifying species.

    Attributes:
        upstream_factory: Factory that created the upstream CRN.
        downstream_factory: Factory that created the downstream CRN.
        coupling_map: Maps upstream species names to the corresponding
            downstream species names that should be identified (merged).
    """

    upstream_factory: MotifFactory
    downstream_factory: MotifFactory
    coupling_map: dict[str, str]


class CRNComposer:
    """Merges two CRNs into one by identifying shared species.

    The resulting CRN contains all species from both CRNs, with shared species
    counted once. Reactions from both CRNs are included with stoichiometry and
    propensity species indices updated to the merged species ordering.
    """

    def compose(
        self,
        upstream_crn: CRN,
        downstream_crn: CRN,
        spec: CompositionSpec,
    ) -> CRN:
        """Merge two CRNs by identifying shared species per coupling_map.

        Args:
            upstream_crn: The upstream CRN.
            downstream_crn: The downstream CRN.
            spec: Composition specification including the coupling map.

        Returns:
            A new CRN containing all species and reactions from both inputs,
            with shared species merged and indices updated.

        Raises:
            ValueError: If coupling_map references species not present in either CRN.
        """
        self._validate_coupling_map(spec.coupling_map, upstream_crn, downstream_crn)

        merged_names, down_to_merged_idx = self._build_merged_species(
            upstream_crn, downstream_crn, spec.coupling_map
        )
        n_merged = len(merged_names)
        up_to_merged_idx = {
            name: i for i, name in enumerate(upstream_crn.species_names)
        }

        upstream_reactions = self._reindex_upstream_reactions(
            upstream_crn, up_to_merged_idx, n_merged
        )
        downstream_reactions = self._reindex_downstream_reactions(
            downstream_crn, down_to_merged_idx, n_merged
        )

        return CRN(
            reactions=upstream_reactions + downstream_reactions,
            species_names=merged_names,
        )

    def _validate_coupling_map(
        self,
        coupling_map: dict[str, str],
        upstream_crn: CRN,
        downstream_crn: CRN,
    ) -> None:
        """Raise ValueError if coupling_map references unknown species.

        Args:
            coupling_map: Upstream-to-downstream species name mapping.
            upstream_crn: Upstream CRN for validation.
            downstream_crn: Downstream CRN for validation.

        Raises:
            ValueError: If any referenced species name is absent.
        """
        for up_name, down_name in coupling_map.items():
            if up_name not in upstream_crn.species_names:
                raise ValueError(
                    f"Upstream species {up_name!r} not found in upstream CRN. "
                    f"Available: {upstream_crn.species_names}"
                )
            if down_name not in downstream_crn.species_names:
                raise ValueError(
                    f"Downstream species {down_name!r} not found in downstream CRN. "
                    f"Available: {downstream_crn.species_names}"
                )

    def _build_merged_species(
        self,
        upstream_crn: CRN,
        downstream_crn: CRN,
        coupling_map: dict[str, str],
    ) -> tuple[list[str], dict[str, int]]:
        """Build the merged species list and a mapping for downstream species.

        Args:
            upstream_crn: Upstream CRN.
            downstream_crn: Downstream CRN.
            coupling_map: Maps upstream names to downstream names.

        Returns:
            Tuple of (merged_names, down_to_merged_idx) where down_to_merged_idx
            maps each downstream species name to its index in the merged list.
        """
        merged_names = list(upstream_crn.species_names)
        reverse_map = {v: k for k, v in coupling_map.items()}

        down_to_merged_idx: dict[str, int] = {}
        for name in downstream_crn.species_names:
            if name in reverse_map:
                up_name = reverse_map[name]
                merged_idx = merged_names.index(up_name)
                down_to_merged_idx[name] = merged_idx
            else:
                merged_idx = len(merged_names)
                merged_names.append(name)
                down_to_merged_idx[name] = merged_idx

        return merged_names, down_to_merged_idx

    def _reindex_upstream_reactions(
        self,
        upstream_crn: CRN,
        up_to_merged_idx: dict[str, int],
        n_merged: int,
    ) -> list[Reaction]:
        """Expand upstream reaction stoichiometries to the merged species count.

        Args:
            upstream_crn: Upstream CRN.
            up_to_merged_idx: Maps upstream species names to merged indices.
            n_merged: Total number of species in merged CRN.

        Returns:
            List of Reaction objects with expanded stoichiometry vectors.
        """
        reactions = []
        for rxn in upstream_crn.reactions:
            new_stoich = torch.zeros(n_merged)
            for s_idx, s_name in enumerate(upstream_crn.species_names):
                new_stoich[up_to_merged_idx[s_name]] = rxn.stoichiometry[s_idx].item()
            reactions.append(
                Reaction(
                    stoichiometry=new_stoich,
                    propensity=rxn.propensity,
                    name=rxn.name,
                )
            )
        return reactions

    def _reindex_downstream_reactions(
        self,
        downstream_crn: CRN,
        down_to_merged_idx: dict[str, int],
        n_merged: int,
    ) -> list[Reaction]:
        """Expand downstream reaction stoichiometries and re-index propensity species.

        Args:
            downstream_crn: Downstream CRN.
            down_to_merged_idx: Maps downstream species names to merged indices.
            n_merged: Total number of species in merged CRN.

        Returns:
            List of Reaction objects with expanded stoichiometry and re-indexed propensities.
        """
        reactions = []
        for rxn in downstream_crn.reactions:
            new_stoich = torch.zeros(n_merged)
            for s_idx, s_name in enumerate(downstream_crn.species_names):
                merged_idx = down_to_merged_idx[s_name]
                new_stoich[merged_idx] = rxn.stoichiometry[s_idx].item()
            new_propensity = self._reindex_propensity(
                rxn.propensity, downstream_crn.species_names, down_to_merged_idx
            )
            reactions.append(
                Reaction(
                    stoichiometry=new_stoich,
                    propensity=new_propensity,
                    name=rxn.name,
                )
            )
        return reactions

    def _reindex_propensity(
        self,
        propensity: Callable[[Any, float], Any],
        old_species_names: tuple[str, ...],
        down_to_merged: dict[str, int],
    ) -> Callable[[Any, float], Any]:
        """Return a new propensity with species indices updated to merged indexing.

        Args:
            propensity: Original propensity callable.
            old_species_names: Species names in the original (downstream) CRN.
            down_to_merged: Maps downstream species names to merged indices.

        Returns:
            A new propensity callable with updated species indices, or the
            original propensity if it has no species index fields to update.
        """
        from crn_surrogate.crn.propensities import (
            ConstantRateParams,
            EnzymeMichaelisMentenParams,
            HillActivationRepressionParams,
            HillParams,
            HillRepressionParams,
            MassActionParams,
            SubstrateInhibitionParams,
            enzyme_michaelis_menten,
            hill,
            hill_activation_repression,
            hill_repression,
            substrate_inhibition,
        )

        if not hasattr(propensity, "params"):
            return propensity

        params = propensity.params

        if isinstance(params, HillParams):
            new_idx = down_to_merged[old_species_names[params.species_index]]
            return hill(params.v_max, params.k_m, params.hill_coefficient, new_idx)
        elif isinstance(params, HillRepressionParams):
            new_idx = down_to_merged[old_species_names[params.species_index]]
            return hill_repression(
                params.k_max, params.k_half, params.hill_coefficient, new_idx
            )
        elif isinstance(params, HillActivationRepressionParams):
            new_act = down_to_merged[old_species_names[params.activator_index]]
            new_rep = down_to_merged[old_species_names[params.repressor_index]]
            return hill_activation_repression(
                params.k_max,
                params.k_act,
                params.n_act,
                new_act,
                params.k_rep,
                params.n_rep,
                new_rep,
            )
        elif isinstance(params, SubstrateInhibitionParams):
            new_idx = down_to_merged[old_species_names[params.species_index]]
            return substrate_inhibition(params.v_max, params.k_m, params.k_i, new_idx)
        elif isinstance(params, EnzymeMichaelisMentenParams):
            new_enz = down_to_merged[old_species_names[params.enzyme_index]]
            new_sub = down_to_merged[old_species_names[params.substrate_index]]
            return enzyme_michaelis_menten(params.k_cat, params.k_m, new_enz, new_sub)
        elif isinstance(params, (MassActionParams, ConstantRateParams)):
            return propensity
        else:
            return propensity
