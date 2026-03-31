"""CRN composition: merge two CRNs by identifying shared species."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import InitialStateRange, MotifFactory


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


@dataclass(frozen=True)
class ComposedParams:
    """Parameters for a composed motif: upstream + downstream sub-params.

    Fields do NOT use param_field() since ranges are derived from the
    sub-factories. The ParameterSampler handles composed params specially
    by sampling upstream and downstream independently.

    Attributes:
        upstream_params: Typed params for the upstream sub-factory.
        downstream_params: Typed params for the downstream sub-factory.
    """

    upstream_params: object
    downstream_params: object


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
        index_map = {
            old_idx: down_to_merged_idx[name]
            for old_idx, name in enumerate(downstream_crn.species_names)
        }
        reactions = []
        for rxn in downstream_crn.reactions:
            new_stoich = torch.zeros(n_merged)
            for s_idx, s_name in enumerate(downstream_crn.species_names):
                merged_idx = down_to_merged_idx[s_name]
                new_stoich[merged_idx] = rxn.stoichiometry[s_idx].item()
            new_propensity = self._reindex_propensity(rxn.propensity, index_map, n_merged)
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
        index_map: dict[int, int],
        n_merged: int,
    ) -> Callable[[Any, float], Any]:
        """Return a new propensity with species indices updated to merged indexing.

        Delegates to the propensity's own reindex_species() method, which each
        closure implements without any isinstance dispatch.

        Args:
            propensity: Original propensity callable.
            index_map: Maps old species index to new (merged) species index.
            n_merged: Total number of species in the merged CRN.

        Returns:
            A new propensity callable with updated species indices, or the
            original propensity if it does not implement reindex_species.
        """
        if hasattr(propensity, "reindex_species"):
            return propensity.reindex_species(index_map, n_merged)
        return propensity


class ComposedMotifFactory(MotifFactory[ComposedParams]):
    """Factory that composes two sub-motifs into a single CRN.

    Implements the MotifFactory interface so the pipeline can treat it
    identically to elementary factories. Parameters are sampled by
    ParameterSampler.sample_composed(), which delegates to the sub-factories.

    Args:
        spec: CompositionSpec defining the upstream/downstream factories
            and coupling map.
        species_names: Optional override for the composed species names.
    """

    def __init__(
        self,
        spec: CompositionSpec,
        *,
        species_names: tuple[str, ...] | None = None,
    ) -> None:
        self._spec = spec
        self._composer = CRNComposer()
        merged = self._compute_merged_species()
        self._n_species = len(merged)
        self._n_reactions = (
            spec.upstream_factory.n_reactions + spec.downstream_factory.n_reactions
        )
        actual_names = species_names or tuple(merged)
        super().__init__(species_names=actual_names)

    def _compute_merged_species(self) -> list[str]:
        """Determine merged species names without running a full simulation.

        Returns:
            Ordered list of species names in the composed CRN.
        """
        up_names = list(self._spec.upstream_factory.species_names)
        reverse_map = {v: k for k, v in self._spec.coupling_map.items()}
        merged = list(up_names)
        for name in self._spec.downstream_factory.species_names:
            if name not in reverse_map:
                merged.append(name)
        return merged

    def _default_species_names(self) -> tuple[str, ...]:
        """Return the merged species names as defaults."""
        return tuple(self._compute_merged_species())

    @property
    def n_species(self) -> int:
        """Total number of species in the composed CRN."""
        return self._n_species

    @property
    def n_reactions(self) -> int:
        """Total number of reactions in the composed CRN."""
        return self._n_reactions

    @property
    def motif_type(self) -> MotifType:
        """Composed factories always return MotifType.COMPOSED."""
        return MotifType.COMPOSED

    @property
    def params_type(self) -> type[ComposedParams]:
        """Parameter type for composed motifs."""
        return ComposedParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Merge initial state ranges from both sub-factories.

        For shared species (in coupling_map), the upstream range is used.

        Returns:
            Dict mapping merged species name to InitialStateRange.
        """
        reverse_map = {v: k for k, v in self._spec.coupling_map.items()}
        ranges: dict[str, InitialStateRange] = {}
        for name, rng in self._spec.upstream_factory.initial_state_ranges().items():
            ranges[name] = rng
        for name, rng in self._spec.downstream_factory.initial_state_ranges().items():
            if name not in reverse_map:
                ranges[name] = rng
        return ranges

    def create(self, params: ComposedParams) -> CRN:
        """Create the composed CRN from upstream and downstream sub-params.

        Args:
            params: ComposedParams with upstream_params and downstream_params.

        Returns:
            Composed CRN with merged species and reactions.
        """
        up_crn = self._spec.upstream_factory.create(params.upstream_params)  # type: ignore[arg-type]
        down_crn = self._spec.downstream_factory.create(params.downstream_params)  # type: ignore[arg-type]
        return self._composer.compose(up_crn, down_crn, self._spec)
