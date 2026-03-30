"""Auto-catalysis motif factory: species promotes its own production."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("A",)
_N_REACTIONS: int = 3


class AutoCatalysisFactory(MotifFactory):
    """Factory for the auto-catalysis motif.

    Topology:
        R1: empty -> A  (basal production at rate k_basal)
        R2: A -> 2A     (autocatalytic amplification at rate k_cat, net +1)
        R3: A -> empty  (first-order degradation at rate k_deg)

    Constraint: k_deg > k_cat (required for bounded steady state).
    """

    def motif_type(self) -> MotifType:
        """Return AUTO_CATALYSIS motif type."""
        return MotifType.AUTO_CATALYSIS

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the auto-catalysis motif.

        Returns:
            MotifParameterRanges with rate and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_basal": (0.1, 10.0),
                "k_cat": (0.001, 0.1),
                "k_deg": (0.01, 0.5),
            },
            hill_coefficient_ranges={},
            initial_state_ranges={"A": (1, 30)},
        )

    def check_constraints(self, params: dict[str, float]) -> bool:
        """Enforce k_deg > k_cat for bounded steady state.

        Args:
            params: Sampled parameter dict.

        Returns:
            True only if k_deg strictly exceeds k_cat.
        """
        return params["k_deg"] > params["k_cat"]

    def create(self, params: dict[str, float]) -> CRN:
        """Create an auto-catalysis CRN from the given parameters.

        Args:
            params: Must contain "k_basal", "k_cat", and "k_deg".

        Returns:
            CRN with basal production, autocatalytic amplification, and degradation.
        """
        k_basal = params["k_basal"]
        k_cat = params["k_cat"]
        k_deg = params["k_deg"]

        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(k_basal),
                name="basal_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=mass_action(k_cat, torch.tensor([1.0])),
                name="autocatalysis",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(k_deg, torch.tensor([1.0])),
                name="degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
