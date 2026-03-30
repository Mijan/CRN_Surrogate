"""Birth-death motif factory: constitutive production and first-order degradation."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("A",)
_N_REACTIONS: int = 2


class BirthDeathFactory(MotifFactory):
    """Factory for the birth-death motif.

    Topology:
        R1: empty -> A  (constitutive production at rate k_prod)
        R2: A -> empty  (first-order degradation at rate k_deg)
    """

    def motif_type(self) -> MotifType:
        """Return BIRTH_DEATH motif type."""
        return MotifType.BIRTH_DEATH

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the birth-death motif.

        Returns:
            MotifParameterRanges with rate and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_prod": (0.5, 50.0),
                "k_deg": (0.01, 1.0),
            },
            hill_coefficient_ranges={},
            initial_state_ranges={"A": (0, 20)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create a birth-death CRN from the given parameters.

        Args:
            params: Must contain "k_prod" and "k_deg".

        Returns:
            CRN with constitutive production and first-order degradation.
        """
        k_prod = params["k_prod"]
        k_deg = params["k_deg"]

        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(k_prod),
                name="birth",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(k_deg, torch.tensor([1.0])),
                name="death",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
