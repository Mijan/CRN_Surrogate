"""Toggle switch motif factory: two mutually repressing species."""

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
class ToggleSwitchParams:
    """Parameters for the toggle switch motif.

    Attributes:
        k_max_A: Maximum production rate of A.
        k_max_B: Maximum production rate of B.
        k_half_A: Half-maximal repression concentration for A production (by B).
        k_half_B: Half-maximal repression concentration for B production (by A).
        n_A: Hill exponent for B-mediated repression of A.
        n_B: Hill exponent for A-mediated repression of B.
        k_deg_A: First-order degradation rate of A.
        k_deg_B: First-order degradation rate of B.
    """

    k_max_A: float = param_field(1e-4, 1e3)  # param_field(10.0, 100.0)
    k_max_B: float = param_field(1e-4, 1e3)  # param_field(10.0, 100.0)
    k_half_A: float = param_field(1e-4, 1e3)  # param_field(10.0, 50.0)
    k_half_B: float = param_field(1e-4, 1e3)  # param_field(10.0, 50.0)
    n_A: float = param_field(
        1.0, 1e2, log_uniform=False
    )  # param_field(2.0, 4.0, log_uniform=False)
    n_B: float = param_field(
        1.0, 1e2, log_uniform=False
    )  # param_field(2.0, 4.0, log_uniform=False)
    k_deg_A: float = param_field(1e-4, 1e3)  # param_field(0.05, 0.5)
    k_deg_B: float = param_field(1e-4, 1e3)  # param_field(0.05, 0.5)


class ToggleSwitchFactory(MotifFactory[ToggleSwitchParams]):
    """Factory for the toggle switch motif.

    Topology:
        R1: empty -> A  (B represses A via Hill kinetics)
        R2: A -> empty  (first-order degradation of A)
        R3: empty -> B  (A represses B via Hill kinetics)
        R4: B -> empty  (first-order degradation of B)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A", "B")
    _N_SPECIES: ClassVar[int] = 2
    _N_REACTIONS: ClassVar[int] = 4
    _IDX_A: ClassVar[int] = 0
    _IDX_B: ClassVar[int] = 1

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
        return MotifType.TOGGLE_SWITCH

    @property
    def params_type(self) -> type[ToggleSwitchParams]:
        return ToggleSwitchParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the toggle switch motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[self._IDX_A]: InitialStateRange(0, 50),
            self._species_names[self._IDX_B]: InitialStateRange(0, 50),
        }

    def create(self, params: ToggleSwitchParams) -> CRN:
        """Create a toggle switch CRN from the given parameters.

        Args:
            params: ToggleSwitchParams with all rate and Hill exponent fields.

        Returns:
            CRN where A and B mutually repress each other.
        """
        self.validate_params(params)
        a_name = self._species_names[self._IDX_A]
        b_name = self._species_names[self._IDX_B]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0]),
                propensity=hill_repression(
                    k_max=params.k_max_A,
                    k_half=params.k_half_A,
                    hill_coefficient=params.n_A,
                    species_index=self._IDX_B,
                ),
                name=f"{a_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0]),
                propensity=mass_action(params.k_deg_A, torch.tensor([1.0, 0.0])),
                name=f"{a_name}_degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0]),
                propensity=hill_repression(
                    k_max=params.k_max_B,
                    k_half=params.k_half_B,
                    hill_coefficient=params.n_B,
                    species_index=self._IDX_A,
                ),
                name=f"{b_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0]),
                propensity=mass_action(params.k_deg_B, torch.tensor([0.0, 1.0])),
                name=f"{b_name}_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
