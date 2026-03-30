"""Enzymatic catalysis motif factory: Michaelis-Menten enzyme kinetics with explicit complex."""

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
    param_field,
)


@dataclass(frozen=True)
class EnzymaticCatalysisParams:
    """Parameters for the enzymatic catalysis motif.

    Attributes:
        k_on: Substrate-enzyme binding rate.
        k_off: Complex dissociation rate.
        k_cat: Catalytic turnover rate.
        k_prod: Substrate input (constitutive production) rate.
        k_deg_P: Product degradation rate.
    """

    k_on: float = param_field(0.001, 0.1)
    k_off: float = param_field(0.01, 1.0)
    k_cat: float = param_field(0.01, 1.0)
    k_prod: float = param_field(0.5, 20.0)
    k_deg_P: float = param_field(0.01, 0.5)


class EnzymaticCatalysisFactory(MotifFactory[EnzymaticCatalysisParams]):
    """Factory for the enzymatic catalysis motif.

    Topology (species order: S, E, C, P):
        R1: S + E -> C         (binding, rate k_on)
        R2: C -> S + E         (unbinding, rate k_off)
        R3: C -> E + P         (catalysis, rate k_cat)
        R4: empty -> S         (substrate input, rate k_prod)
        R5: P -> empty         (product degradation, rate k_deg_P)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("S", "E", "C", "P")
    _N_SPECIES: ClassVar[int] = 4
    _N_REACTIONS: ClassVar[int] = 5
    _IDX_S: ClassVar[int] = 0
    _IDX_E: ClassVar[int] = 1
    _IDX_C: ClassVar[int] = 2
    _IDX_P: ClassVar[int] = 3

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
        return MotifType.ENZYMATIC_CATALYSIS

    @property
    def params_type(self) -> type[EnzymaticCatalysisParams]:
        return EnzymaticCatalysisParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the enzymatic catalysis motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[self._IDX_S]: InitialStateRange(10, 100),
            self._species_names[self._IDX_E]: InitialStateRange(5, 50),
            self._species_names[self._IDX_C]: InitialStateRange(0, 0),
            self._species_names[self._IDX_P]: InitialStateRange(0, 0),
        }

    def create(self, params: EnzymaticCatalysisParams) -> CRN:
        """Create an enzymatic catalysis CRN from the given parameters.

        Args:
            params: EnzymaticCatalysisParams with k_on, k_off, k_cat, k_prod, k_deg_P.

        Returns:
            CRN with explicit enzyme-substrate complex formation and catalysis.
        """
        self.validate_params(params)
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([-1.0, -1.0, 1.0, 0.0]),
                propensity=mass_action(
                    params.k_on,
                    torch.tensor([1.0, 1.0, 0.0, 0.0]),
                ),
                name="binding",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0, 1.0, -1.0, 0.0]),
                propensity=mass_action(
                    params.k_off,
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                ),
                name="unbinding",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, -1.0, 1.0]),
                propensity=mass_action(
                    params.k_cat,
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                ),
                name="catalysis",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                propensity=constant_rate(params.k_prod),
                name="substrate_input",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 0.0, -1.0]),
                propensity=mass_action(
                    params.k_deg_P,
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                ),
                name="product_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
