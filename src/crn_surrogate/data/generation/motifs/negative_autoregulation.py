"""Negative autoregulation motif factory: species represses its own production via Hill kinetics."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import hill_repression, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("A",)
_N_REACTIONS: int = 2
_SPECIES_INDEX_A: int = 0


class NegativeAutoregulationFactory(MotifFactory):
    """Factory for the negative autoregulation motif.

    Topology:
        R1: empty -> A  (Hill-repression by A itself)
        R2: A -> empty  (first-order degradation)
    """

    def motif_type(self) -> MotifType:
        """Return NEGATIVE_AUTOREGULATION motif type."""
        return MotifType.NEGATIVE_AUTOREGULATION

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the negative autoregulation motif.

        Returns:
            MotifParameterRanges with rate, Hill coefficient, and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_max": (1.0, 100.0),
                "k_half": (5.0, 50.0),
                "k_deg": (0.01, 0.5),
            },
            hill_coefficient_ranges={"n_hill": (1.0, 4.0)},
            initial_state_ranges={"A": (0, 30)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create a negative autoregulation CRN from the given parameters.

        Args:
            params: Must contain "k_max", "k_half", "n_hill", and "k_deg".

        Returns:
            CRN where A represses its own production via Hill kinetics.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=hill_repression(
                    k_max=params["k_max"],
                    k_half=params["k_half"],
                    hill_coefficient=params["n_hill"],
                    species_index=_SPECIES_INDEX_A,
                ),
                name="hill_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(params["k_deg"], torch.tensor([1.0])),
                name="degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
