"""Abstract base class for CRN motif factories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.motif_type import MotifType


@dataclass(frozen=True)
class MotifParameterRanges:
    """Specification of valid parameter ranges for a motif.

    Attributes:
        rate_ranges: Maps parameter name to (low, high) bounds. Sampling is
            log-uniform over these ranges.
        hill_coefficient_ranges: Maps Hill exponent names to (low, high) bounds.
            Sampling is uniform (not log-uniform) over these ranges.
        initial_state_ranges: Maps species name to (low, high) integer bounds
            for the initial population. Sampling is uniform integer.
    """

    rate_ranges: dict[str, tuple[float, float]]
    hill_coefficient_ranges: dict[str, tuple[float, float]]
    initial_state_ranges: dict[str, tuple[int, int]]


class MotifFactory(ABC):
    """Abstract factory that creates CRN instances for a specific motif type.

    Subclasses define the topology (stoichiometry and propensity structure) and
    the valid parameter ranges for that topology. The ParameterSampler draws
    random parameter dicts and passes them to create().
    """

    @abstractmethod
    def motif_type(self) -> MotifType:
        """Return the MotifType enum value for this factory."""
        ...

    @abstractmethod
    def parameter_ranges(self) -> MotifParameterRanges:
        """Return the valid parameter ranges for this motif.

        Returns:
            MotifParameterRanges with rate, Hill coefficient, and initial-state bounds.
        """
        ...

    @abstractmethod
    def create(self, params: dict[str, float]) -> CRN:
        """Instantiate a CRN from a sampled parameter dict.

        Args:
            params: Dict mapping parameter name to sampled value. Must contain
                all keys declared in parameter_ranges().

        Returns:
            CRN with stoichiometry and propensities set from params.
        """
        ...

    @abstractmethod
    def species_names(self) -> tuple[str, ...]:
        """Return the ordered species names for this motif.

        Returns:
            Tuple of species name strings in the same order as stoichiometry rows.
        """
        ...

    @property
    def n_species(self) -> int:
        """Number of species in this motif."""
        return len(self.species_names())

    @abstractmethod
    def n_reactions(self) -> int:
        """Number of reactions in this motif.

        Returns:
            Integer count of reactions.
        """
        ...

    def check_constraints(self, params: dict[str, float]) -> bool:
        """Check motif-specific parameter constraints beyond simple range bounds.

        Override in subclasses that have inter-parameter constraints. The default
        implementation accepts any parameter dict.

        Args:
            params: Dict of sampled parameter values.

        Returns:
            True if the parameter dict satisfies all constraints.
        """
        return True
