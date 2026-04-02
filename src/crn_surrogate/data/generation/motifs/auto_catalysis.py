"""Auto-catalysis motif factory: species promotes its own production."""

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
class AutoCatalysisParams(MotifParams):
    """Parameters for the auto-catalysis motif.

    Attributes:
        k_basal: Basal (constitutive) production rate.
        k_cat: Autocatalytic amplification rate.
        k_deg: First-order degradation rate. Must exceed k_cat for bounded dynamics.
    """

    k_basal: float = param_field(0.01, 50.0)
    k_cat: float = param_field(0.0005, 1.0)
    k_deg: float = param_field(0.005, 5.0)


class AutoCatalysisFactory(MotifFactory[AutoCatalysisParams]):
    """Factory for the auto-catalysis motif.

    Topology:
        R1: empty -> A  (basal production at rate k_basal)
        R2: A -> 2A     (autocatalytic amplification at rate k_cat, net +1)
        R3: A -> empty  (first-order degradation at rate k_deg)

    Constraint: k_deg > k_cat (required for bounded steady state).
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("A",)
    _N_SPECIES: ClassVar[int] = 1
    _N_REACTIONS: ClassVar[int] = 3

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
        return MotifType.AUTO_CATALYSIS

    @property
    def params_type(self) -> type[AutoCatalysisParams]:
        return AutoCatalysisParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the auto-catalysis motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[0]: InitialStateRange(1, 30),
        }

    def validate_params(self, params: AutoCatalysisParams) -> None:
        """Enforce k_deg > k_cat for bounded steady state.

        Args:
            params: AutoCatalysisParams to validate.

        Raises:
            TypeError: If params is not AutoCatalysisParams.
            ValueError: If k_cat >= k_deg.
        """
        super().validate_params(params)
        if params.k_cat >= params.k_deg:
            raise ValueError(
                f"Auto-catalysis requires k_deg > k_cat for bounded dynamics, "
                f"got k_cat={params.k_cat}, k_deg={params.k_deg}"
            )

    def create(self, params: AutoCatalysisParams) -> CRN:
        """Create an auto-catalysis CRN from the given parameters.

        Args:
            params: AutoCatalysisParams with k_basal, k_cat, and k_deg.

        Returns:
            CRN with basal production, autocatalytic amplification, and degradation.
        """
        self.validate_params(params)
        a_name = self._species_names[0]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(params.k_basal),
                name=f"{a_name}_basal_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=mass_action(params.k_cat, torch.tensor([1.0])),
                name=f"{a_name}_autocatalysis",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(params.k_deg, torch.tensor([1.0])),
                name=f"{a_name}_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
