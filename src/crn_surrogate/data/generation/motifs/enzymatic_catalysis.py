"""Enzymatic catalysis motif factory: Michaelis-Menten enzyme kinetics with explicit complex."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("S", "E", "C", "P")
_N_REACTIONS: int = 5

# Species indices
_IDX_S: int = 0
_IDX_E: int = 1
_IDX_C: int = 2
_IDX_P: int = 3


class EnzymaticCatalysisFactory(MotifFactory):
    """Factory for the enzymatic catalysis motif.

    Topology (species order: S, E, C, P):
        R1: S + E -> C         (binding, rate k_on)
        R2: C -> S + E         (unbinding, rate k_off)
        R3: C -> E + P         (catalysis, rate k_cat)
        R4: empty -> S         (substrate input, rate k_prod)
        R5: P -> empty         (product degradation, rate k_deg_P)
    """

    def motif_type(self) -> MotifType:
        """Return ENZYMATIC_CATALYSIS motif type."""
        return MotifType.ENZYMATIC_CATALYSIS

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the enzymatic catalysis motif.

        Returns:
            MotifParameterRanges with rate and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_on": (0.001, 0.1),
                "k_off": (0.01, 1.0),
                "k_cat": (0.01, 1.0),
                "k_prod": (0.5, 20.0),
                "k_deg_P": (0.01, 0.5),
            },
            hill_coefficient_ranges={},
            initial_state_ranges={
                "S": (10, 100),
                "E": (5, 50),
                "C": (0, 0),
                "P": (0, 0),
            },
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create an enzymatic catalysis CRN from the given parameters.

        Args:
            params: Must contain k_on, k_off, k_cat, k_prod, k_deg_P.

        Returns:
            CRN with explicit enzyme-substrate complex formation and catalysis.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([-1.0, -1.0, 1.0, 0.0]),
                propensity=mass_action(
                    params["k_on"],
                    torch.tensor([1.0, 1.0, 0.0, 0.0]),
                ),
                name="binding",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0, 1.0, -1.0, 0.0]),
                propensity=mass_action(
                    params["k_off"],
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                ),
                name="unbinding",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, -1.0, 1.0]),
                propensity=mass_action(
                    params["k_cat"],
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                ),
                name="catalysis",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                propensity=constant_rate(params["k_prod"]),
                name="substrate_input",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 0.0, -1.0]),
                propensity=mass_action(
                    params["k_deg_P"],
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                ),
                name="product_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
