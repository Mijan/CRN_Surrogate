"""Repressilator motif factory: three-species cyclic repression network."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import hill_repression, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("A", "B", "C")
_N_REACTIONS: int = 6

# Species indices
_IDX_A: int = 0
_IDX_B: int = 1
_IDX_C: int = 2


class RepressilatorFactory(MotifFactory):
    """Factory for the repressilator motif.

    Topology (species order: A, B, C):
        R1: empty -> A  (C represses A via Hill kinetics)
        R2: A -> empty  (first-order degradation of A)
        R3: empty -> B  (A represses B via Hill kinetics)
        R4: B -> empty  (first-order degradation of B)
        R5: empty -> C  (B represses C via Hill kinetics)
        R6: C -> empty  (first-order degradation of C)
    """

    def motif_type(self) -> MotifType:
        """Return REPRESSILATOR motif type."""
        return MotifType.REPRESSILATOR

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the repressilator motif.

        Returns:
            MotifParameterRanges with rate, Hill coefficient, and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_max_A": (20.0, 200.0),
                "k_max_B": (20.0, 200.0),
                "k_max_C": (20.0, 200.0),
                "k_half_A": (10.0, 50.0),
                "k_half_B": (10.0, 50.0),
                "k_half_C": (10.0, 50.0),
                "k_deg_A": (0.05, 0.5),
                "k_deg_B": (0.05, 0.5),
                "k_deg_C": (0.05, 0.5),
            },
            hill_coefficient_ranges={
                "n_A": (2.0, 5.0),
                "n_B": (2.0, 5.0),
                "n_C": (2.0, 5.0),
            },
            initial_state_ranges={"A": (10, 50), "B": (0, 0), "C": (0, 0)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create a repressilator CRN from the given parameters.

        Args:
            params: Must contain all keys declared in parameter_ranges().

        Returns:
            CRN implementing the three-species cyclic repressilator.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
                propensity=hill_repression(
                    k_max=params["k_max_A"],
                    k_half=params["k_half_A"],
                    hill_coefficient=params["n_A"],
                    species_index=_IDX_C,
                ),
                name="production_A",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0, 0.0]),
                propensity=mass_action(
                    params["k_deg_A"], torch.tensor([1.0, 0.0, 0.0])
                ),
                name="degradation_A",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, 0.0]),
                propensity=hill_repression(
                    k_max=params["k_max_B"],
                    k_half=params["k_half_B"],
                    hill_coefficient=params["n_B"],
                    species_index=_IDX_A,
                ),
                name="production_B",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0, 0.0]),
                propensity=mass_action(
                    params["k_deg_B"], torch.tensor([0.0, 1.0, 0.0])
                ),
                name="degradation_B",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 1.0]),
                propensity=hill_repression(
                    k_max=params["k_max_C"],
                    k_half=params["k_half_C"],
                    hill_coefficient=params["n_C"],
                    species_index=_IDX_B,
                ),
                name="production_C",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, -1.0]),
                propensity=mass_action(
                    params["k_deg_C"], torch.tensor([0.0, 0.0, 1.0])
                ),
                name="degradation_C",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
