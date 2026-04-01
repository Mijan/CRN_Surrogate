"""Birth-death motif factory: constitutive production and first-order degradation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
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

    k_prod: float = param_field(1e-4, 1e3)
    k_deg: float = param_field(1e-4, 1e3)


class BirthDeathFactory(MotifFactory[BirthDeathParams]):
    """Factory for the birth-death motif.

    Topology:
        R1: empty -> A  (constitutive production at rate k_prod)
        R2: A -> empty  (first-order degradation at rate k_deg)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A",)
    _N_SPECIES: ClassVar[int] = 1
    _N_REACTIONS: ClassVar[int] = 2

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
        a_name = self._species_names[0]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(params.k_prod),
                name=f"{a_name}_birth",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(params.k_deg, torch.tensor([1.0])),
                name=f"{a_name}_death",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
