"""Birth-death motif factory: constitutive production and first-order degradation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.mass_action_topology import (
    MassActionTopology,
    birth_death_topology,
)
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    MotifParams,
    param_field,
)


@dataclass(frozen=True)
class BirthDeathParams(MotifParams):
    """Parameters for the birth-death motif.

    Attributes:
        k_prod: Constitutive production rate.
        k_deg: First-order degradation rate.
    """

    k_prod: float = param_field(0.05, 200.0)
    k_deg: float = param_field(0.005, 5.0)


class BirthDeathFactory(MotifFactory[BirthDeathParams]):
    """Factory for the birth-death motif.

    Topology:
        R1: empty -> A  (constitutive production at rate k_prod)
        R2: A -> empty  (first-order degradation at rate k_deg)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A",)
    _N_SPECIES: ClassVar[int] = 1
    _N_REACTIONS: ClassVar[int] = 2
    TOPOLOGY: ClassVar[MassActionTopology] = birth_death_topology()

    def _default_species_names(self) -> tuple[str, ...]:
        return self._DEFAULT_SPECIES

    @property
    def n_species(self) -> int:
        return self._N_SPECIES

    @property
    def n_reactions(self) -> int:
        return self._N_REACTIONS

    @property
    def motif_type(self) -> MotifType:
        return MotifType.BIRTH_DEATH

    @property
    def params_type(self) -> type[BirthDeathParams]:
        return BirthDeathParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the birth-death motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[0]: InitialStateRange(0, 20),
        }

    def create(self, params: BirthDeathParams) -> CRN:
        """Create a birth-death CRN from the given parameters.

        Args:
            params: BirthDeathParams with k_prod and k_deg.

        Returns:
            CRN with constitutive production and first-order degradation.
        """
        self.validate_params(params)
        # Rate order matches topology reaction order: [birth (k_prod), death (k_deg)]
        crn = self.TOPOLOGY.to_crn([params.k_prod, params.k_deg])
        if self._species_names != self.TOPOLOGY.species_names:
            return CRN(reactions=crn.reactions, species_names=list(self._species_names))
        return crn
