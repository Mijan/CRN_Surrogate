"""Enzymatic catalysis motif factory: Michaelis-Menten enzyme kinetics with explicit complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.mass_action_topology import (
    MassActionTopology,
    enzymatic_catalysis_topology,
)
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    MotifParams,
    param_field,
)


@dataclass(frozen=True)
class EnzymaticCatalysisParams(MotifParams):
    """Parameters for the enzymatic catalysis motif.

    Attributes:
        k_on: Substrate-enzyme binding rate.
        k_off: Complex dissociation rate.
        k_cat: Catalytic turnover rate.
        k_prod: Substrate input (constitutive production) rate.
        k_deg_P: Product degradation rate.
    """

    k_on: float = param_field(5e-5, 0.5)
    k_off: float = param_field(0.005, 5.0)
    k_cat: float = param_field(0.005, 5.0)
    k_prod: float = param_field(0.1, 100.0)
    k_deg_P: float = param_field(0.005, 2.0)


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
    TOPOLOGY: ClassVar[MassActionTopology] = enzymatic_catalysis_topology()

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
        # Rate order matches topology reaction order: [binding, unbinding, catalysis, substrate_input, product_degradation]
        crn = self.TOPOLOGY.to_crn([
            params.k_on, params.k_off, params.k_cat, params.k_prod, params.k_deg_P
        ])
        if self._species_names != self.TOPOLOGY.species_names:
            return CRN(reactions=crn.reactions, species_names=list(self._species_names))
        return crn
