"""Repressilator motif factory: three-species cyclic repression network."""

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
    MotifParams,
    param_field,
)


@dataclass(frozen=True)
class RepressilatorParams(MotifParams):
    """Parameters for the repressilator motif.

    Attributes:
        k_max_A: Maximum production rate of A.
        k_max_B: Maximum production rate of B.
        k_max_C: Maximum production rate of C.
        k_half_A: Half-maximal repression concentration for A (by C).
        k_half_B: Half-maximal repression concentration for B (by A).
        k_half_C: Half-maximal repression concentration for C (by B).
        n_A: Hill exponent for C-mediated repression of A.
        n_B: Hill exponent for A-mediated repression of B.
        n_C: Hill exponent for B-mediated repression of C.
        k_deg_A: Degradation rate of A.
        k_deg_B: Degradation rate of B.
        k_deg_C: Degradation rate of C.
    """

    k_max_A: float = param_field(1e-4, 1e3)  # param_field(20.0, 200.0)
    k_max_B: float = param_field(1e-4, 1e3)  # param_field(20.0, 200.0)
    k_max_C: float = param_field(1e-4, 1e3)  # param_field(20.0, 200.0)
    k_half_A: float = param_field(1e-4, 1e3)  # param_field(10.0, 50.0)
    k_half_B: float = param_field(1e-4, 1e3)  # param_field(10.0, 50.0)
    k_half_C: float = param_field(1e-4, 1e3)  # param_field(10.0, 50.0)
    n_A: float = param_field(
        1, 1e2, log_uniform=False
    )  # param_field(2.0, 5.0, log_uniform=False)
    n_B: float = param_field(
        1, 1e2, log_uniform=False
    )  # param_field(2.0, 5.0, log_uniform=False)
    n_C: float = param_field(
        1, 1e2, log_uniform=False
    )  # param_field(2.0, 5.0, log_uniform=False)
    k_deg_A: float = param_field(1e-4, 1e3)  # param_field(0.05, 0.5)
    k_deg_B: float = param_field(1e-4, 1e3)  # param_field(0.05, 0.5)
    k_deg_C: float = param_field(1e-4, 1e3)  # param_field(0.05, 0.5)


class RepressilatorFactory(MotifFactory[RepressilatorParams]):
    """Factory for the repressilator motif.

    Topology (species order: A, B, C):
        R1: empty -> A  (C represses A via Hill kinetics)
        R2: A -> empty  (first-order degradation of A)
        R3: empty -> B  (A represses B via Hill kinetics)
        R4: B -> empty  (first-order degradation of B)
        R5: empty -> C  (B represses C via Hill kinetics)
        R6: C -> empty  (first-order degradation of C)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A", "B", "C")
    _N_SPECIES: ClassVar[int] = 3
    _N_REACTIONS: ClassVar[int] = 6
    _IDX_A: ClassVar[int] = 0
    _IDX_B: ClassVar[int] = 1
    _IDX_C: ClassVar[int] = 2

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
        return MotifType.REPRESSILATOR

    @property
    def params_type(self) -> type[RepressilatorParams]:
        return RepressilatorParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the repressilator motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[self._IDX_A]: InitialStateRange(10, 50),
            self._species_names[self._IDX_B]: InitialStateRange(0, 0),
            self._species_names[self._IDX_C]: InitialStateRange(0, 0),
        }

    def create(self, params: RepressilatorParams) -> CRN:
        """Create a repressilator CRN from the given parameters.

        Args:
            params: RepressilatorParams with all rate and Hill exponent fields.

        Returns:
            CRN implementing the three-species cyclic repressilator.
        """
        self.validate_params(params)
        a_name = self._species_names[self._IDX_A]
        b_name = self._species_names[self._IDX_B]
        c_name = self._species_names[self._IDX_C]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
                propensity=hill_repression(
                    k_max=params.k_max_A,
                    k_half=params.k_half_A,
                    hill_coefficient=params.n_A,
                    species_index=self._IDX_C,
                ),
                name=f"{a_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0, 0.0]),
                propensity=mass_action(params.k_deg_A, torch.tensor([1.0, 0.0, 0.0])),
                name=f"{a_name}_degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, 0.0]),
                propensity=hill_repression(
                    k_max=params.k_max_B,
                    k_half=params.k_half_B,
                    hill_coefficient=params.n_B,
                    species_index=self._IDX_A,
                ),
                name=f"{b_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0, 0.0]),
                propensity=mass_action(params.k_deg_B, torch.tensor([0.0, 1.0, 0.0])),
                name=f"{b_name}_degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 1.0]),
                propensity=hill_repression(
                    k_max=params.k_max_C,
                    k_half=params.k_half_C,
                    hill_coefficient=params.n_C,
                    species_index=self._IDX_B,
                ),
                name=f"{c_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, -1.0]),
                propensity=mass_action(params.k_deg_C, torch.tensor([0.0, 0.0, 1.0])),
                name=f"{c_name}_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
