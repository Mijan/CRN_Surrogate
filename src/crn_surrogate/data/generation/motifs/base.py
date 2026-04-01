"""Abstract base class for CRN motif factories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Generic, TypeVar

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.motif_type import MotifType

P = TypeVar("P")


@dataclass(frozen=True)
class ParameterRange:
    """Range for a single kinetic parameter.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log_uniform: If True, sample in log-space. If False, sample uniform.
    """

    low: float
    high: float
    log_uniform: bool = True


@dataclass(frozen=True)
class InitialStateRange:
    """Range for a single species' initial count.

    Attributes:
        low: Minimum initial count (inclusive).
        high: Maximum initial count (inclusive).
    """

    low: int
    high: int


def param_field(
    low: float,
    high: float,
    *,
    log_uniform: bool = True,
) -> float:
    """Declare a parameter field with its sampling range as co-located metadata.

    Use this instead of a bare default on params dataclass fields. The range
    metadata is extracted by extract_parameter_ranges() and used by the
    ParameterSampler. This ensures the parameter name, type, and valid range
    are defined in exactly one place.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log_uniform: If True, sample in log-space. If False, sample uniform.

    Returns:
        A dataclass field descriptor with range metadata attached.
    """
    return field(  # type: ignore[return-value]
        metadata={"range": ParameterRange(low, high, log_uniform=log_uniform)},
    )


def extract_parameter_ranges(params_type: type) -> dict[str, ParameterRange]:
    """Extract ParameterRange metadata from a params dataclass type.

    Args:
        params_type: A frozen dataclass class whose fields were declared
            with param_field().

    Returns:
        Dict mapping field name to ParameterRange, in field declaration order.

    Raises:
        ValueError: If any field lacks range metadata.
        TypeError: If any field's range metadata is not a ParameterRange.
    """
    ranges: dict[str, ParameterRange] = {}
    for f in dc_fields(params_type):
        range_info = f.metadata.get("range")
        if range_info is None:
            raise ValueError(
                f"Field '{f.name}' on {params_type.__name__} has no range "
                f"metadata. Use param_field() instead of a bare annotation."
            )
        if not isinstance(range_info, ParameterRange):
            raise TypeError(
                f"Field '{f.name}' range metadata must be a ParameterRange, "
                f"got {type(range_info).__name__}"
            )
        ranges[f.name] = range_info
    return ranges


class MotifFactory(ABC, Generic[P]):
    """Abstract factory for constructing CRNs from a specific motif template.

    Type parameter P is the frozen params dataclass for this motif. Each field
    of P must be declared with param_field() so that the ParameterSampler can
    discover ranges without duplicating them in the factory.

    Args:
        species_names: Optional override for species names. If None, uses
            the motif's default names. Length must match n_species.
    """

    def __init__(
        self,
        *,
        species_names: tuple[str, ...] | None = None,
    ) -> None:
        names = species_names or self._default_species_names()
        if len(names) != self.n_species:
            raise ValueError(
                f"{type(self).__name__} requires {self.n_species} species, "
                f"got {len(names)}: {names}"
            )
        self._species_names = names

    # --- Abstract interface ---

    @abstractmethod
    def _default_species_names(self) -> tuple[str, ...]:
        """Return the default species names for this motif."""
        ...

    @property
    @abstractmethod
    def n_species(self) -> int:
        """Number of species in this motif."""
        ...

    @property
    @abstractmethod
    def n_reactions(self) -> int:
        """Number of reactions in this motif."""
        ...

    @property
    @abstractmethod
    def motif_type(self) -> MotifType:
        """The MotifType enum value for this motif."""
        ...

    @property
    @abstractmethod
    def params_type(self) -> type[P]:
        """Return the params dataclass class (not an instance).

        Used by the ParameterSampler to construct typed params from sampled
        values via extract_parameter_ranges(factory.params_type).
        """
        ...

    @abstractmethod
    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return sampling ranges for each species' initial state.

        Keys must exactly match the species names (self.species_names).

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        ...

    @abstractmethod
    def create(self, params: P) -> CRN:
        """Construct a CRN from typed parameters.

        Args:
            params: A frozen dataclass instance of type P.

        Returns:
            A fully specified CRN.
        """
        ...

    # --- Concrete interface ---

    @property
    def species_names(self) -> tuple[str, ...]:
        """The species names (configurable via constructor)."""
        return self._species_names

    def validate_params(self, params: P) -> None:
        """Validate parameter values. Override to add motif-specific checks.

        Base implementation checks that params is an instance of params_type.
        Subclasses call super() then add domain-specific constraints.

        Args:
            params: The params instance to validate.

        Raises:
            TypeError: If params is not the expected type.
            ValueError: If any parameter-specific constraint is violated.
        """
        if not isinstance(params, self.params_type):
            raise TypeError(
                f"Expected {self.params_type.__name__}, got {type(params).__name__}"
            )

    def params_from_dict(self, d: dict[str, float]) -> P:
        """Construct a typed params instance from a dict.

        This is the ONLY place where string-keyed dicts are converted to
        typed params. The ParameterSampler uses this.

        Args:
            d: Dict with keys matching params_type field names.

        Returns:
            An instance of P.

        Raises:
            TypeError: If keys don't match params_type fields.
        """
        return self.params_type(**d)
