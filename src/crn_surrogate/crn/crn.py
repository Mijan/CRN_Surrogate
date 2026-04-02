"""Chemical Reaction Network: an ordered collection of reactions over a fixed species set."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from crn_surrogate.crn.reaction import Reaction


class CRN:
    """An ordered collection of reactions over a fixed set of species.

    The CRN is defined by its reactions and optional species names. Simulation
    concerns (initial conditions, time span) are not part of this object.

    Args:
        reactions: Ordered sequence of Reaction objects. All stoichiometry
            vectors must have the same length (n_species).
        species_names: Optional names for each species. If omitted, defaults
            to "S0", "S1", etc.
        external_species: Set of species indices that are externally controlled
            (pseudo-species driven by an InputProtocol rather than CRN kinetics).
            External species must have zero net stoichiometric change in all
            reactions; they can only appear as propensity dependencies.

    Raises:
        ValueError: If reactions is empty, stoichiometry lengths mismatch,
            species_names length does not match n_species, external_species
            indices are out of range, or any external species has nonzero net
            stoichiometric change in any reaction.
    """

    def __init__(
        self,
        reactions: Sequence[Reaction],
        species_names: Sequence[str] | None = None,
        external_species: frozenset[int] = frozenset(),
    ) -> None:
        if not reactions:
            raise ValueError("CRN must have at least one reaction")

        n_species = reactions[0].stoichiometry.shape[0]
        for i, rxn in enumerate(reactions):
            if rxn.stoichiometry.shape[0] != n_species:
                raise ValueError(
                    f"Reaction {i} stoichiometry length {rxn.stoichiometry.shape[0]} "
                    f"does not match n_species={n_species}"
                )

        self._reactions = tuple(reactions)
        self._n_species = n_species

        if species_names is not None:
            if len(species_names) != n_species:
                raise ValueError(
                    f"species_names length {len(species_names)} must equal "
                    f"n_species={n_species}"
                )
            self._species_names = tuple(species_names)
        else:
            self._species_names = tuple(f"S{i}" for i in range(n_species))

        # Validate external_species indices
        invalid = frozenset(idx for idx in external_species if idx not in range(n_species))
        if invalid:
            raise ValueError(
                f"external_species indices {sorted(invalid)} are out of range "
                f"for a CRN with {n_species} species (valid range: 0..{n_species - 1})"
            )

        # Validate that external species have zero net stoichiometric change
        for idx in external_species:
            for r, rxn in enumerate(reactions):
                net_change = rxn.stoichiometry[idx].item()
                if net_change != 0:
                    raise ValueError(
                        f"External species {idx} ('{self._species_names[idx]}') has "
                        f"nonzero net stoichiometric change ({net_change:+g}) in "
                        f"reaction {r} ('{rxn.name}'). External species must not be "
                        f"products or reactants — only propensity dependencies."
                    )

        self._external_species = external_species
        self._stoichiometry_matrix: torch.Tensor | None = None

    @property
    def n_species(self) -> int:
        """Number of species in the CRN."""
        return self._n_species

    @property
    def n_reactions(self) -> int:
        """Number of reactions in the CRN."""
        return len(self._reactions)

    @property
    def species_names(self) -> tuple[str, ...]:
        """Names of species, one per species."""
        return self._species_names

    @property
    def reactions(self) -> tuple[Reaction, ...]:
        """All reactions as an immutable tuple."""
        return self._reactions

    @property
    def stoichiometry_matrix(self) -> torch.Tensor:
        """Net change matrix, shape (n_reactions, n_species). Computed lazily and cached."""
        if self._stoichiometry_matrix is None:
            self._stoichiometry_matrix = torch.stack(
                [rxn.stoichiometry.float() for rxn in self._reactions], dim=0
            )
        return self._stoichiometry_matrix

    @property
    def dependency_matrix(self) -> torch.Tensor:
        """(n_reactions, n_species) binary matrix of propensity dependencies.

        Entry [r, s] is 1.0 if species s influences the rate of reaction r,
        derived from each reaction's propensity.species_dependencies. If a
        propensity does not declare dependencies, all species are assumed to
        be dependencies and a warning is issued.
        """
        rows = []
        for rxn in self._reactions:
            prop = rxn.propensity
            row = torch.zeros(self._n_species)
            if hasattr(prop, "species_dependencies"):
                for s in prop.species_dependencies:
                    row[s] = 1.0
            else:
                warnings.warn(
                    f"Propensity {type(prop).__name__!r} does not declare "
                    f"species_dependencies; assuming all species are dependencies.",
                    stacklevel=2,
                )
                row = torch.ones(self._n_species)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def evaluate_propensities(
        self, state: torch.Tensor, t: float = 0.0
    ) -> torch.Tensor:
        """Evaluate all reaction propensities at the given state and time.

        Args:
            state: (n_species,) current state vector.
            t: Current time (for time-varying propensities).

        Returns:
            (n_reactions,) propensity values, clamped to non-negative.
        """
        return torch.stack([rxn.propensity(state, t) for rxn in self._reactions]).clamp(
            min=0.0
        )

    def reaction(self, index: int) -> Reaction:
        """Access a reaction by index.

        Args:
            index: Zero-based reaction index.

        Returns:
            The Reaction at the given index.
        """
        return self._reactions[index]

    @property
    def external_species(self) -> frozenset[int]:
        """Set of species indices that are externally controlled."""
        return self._external_species

    @property
    def n_external_species(self) -> int:
        """Number of externally controlled species."""
        return len(self._external_species)

    @property
    def is_external(self) -> np.ndarray:
        """Boolean array of shape (n_species,): True for externally controlled species."""
        mask = np.zeros(self._n_species, dtype=bool)
        for idx in self._external_species:
            mask[idx] = True
        return mask

    @property
    def internal_species_mask(self) -> np.ndarray:
        """Boolean array of shape (n_species,): True for non-external (kinetic) species."""
        return ~self.is_external

    def __repr__(self) -> str:
        return (
            f"CRN(n_species={self.n_species}, n_reactions={self.n_reactions}, "
            f"species={self.species_names})"
        )
