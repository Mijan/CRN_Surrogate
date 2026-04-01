"""Negative autoregulation motif factory: species represses its own production via Hill kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import hill_repression, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    param_field,
)


@dataclass(frozen=True)
class NegativeAutoregulationParams:
    """Parameters for the negative autoregulation motif.

    Attributes:
        k_max: Maximum production rate.
        k_half: Half-maximal repression concentration.
        n_hill: Hill exponent for repression.
        k_deg: First-order degradation rate.
    """

    k_max: float = param_field(1e-4, 1e3)  # param_field(1.0, 100.0)
    k_half: float = param_field(1e-4, 1e3)  # param_field(5.0, 50.0)
    n_hill: float = param_field(1e-4, 1e3)  # param_field(1.0, 4.0, log_uniform=False)
    k_deg: float = param_field(1e-4, 1e3)  # param_field(0.01, 0.5)


class NegativeAutoregulationFactory(MotifFactory[NegativeAutoregulationParams]):
    """Factory for the negative autoregulation motif.

    Topology:
        R1: empty -> A  (Hill-repression by A itself)
        R2: A -> empty  (first-order degradation)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A",)
    _N_SPECIES: ClassVar[int] = 1
    _N_REACTIONS: ClassVar[int] = 2
    _IDX_A: ClassVar[int] = 0

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
        return MotifType.NEGATIVE_AUTOREGULATION

    @property
    def params_type(self) -> type[NegativeAutoregulationParams]:
        return NegativeAutoregulationParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the negative autoregulation motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[0]: InitialStateRange(0, 30),
        }

    def create(self, params: NegativeAutoregulationParams) -> CRN:
        """Create a negative autoregulation CRN from the given parameters.

        Args:
            params: NegativeAutoregulationParams with k_max, k_half, n_hill, k_deg.

        Returns:
            CRN where A represses its own production via Hill kinetics.
        """
        self.validate_params(params)
        a_name = self._species_names[0]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=hill_repression(
                    k_max=params.k_max,
                    k_half=params.k_half,
                    hill_coefficient=params.n_hill,
                    species_index=self._IDX_A,
                ),
                name=f"{a_name}_hill_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(params.k_deg, torch.tensor([1.0])),
                name=f"{a_name}_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
