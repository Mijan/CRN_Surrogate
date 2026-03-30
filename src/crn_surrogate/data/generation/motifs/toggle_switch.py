"""Toggle switch motif factory: two mutually repressing species."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import hill_repression, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("A", "B")
_N_REACTIONS: int = 4
_SPECIES_INDEX_A: int = 0
_SPECIES_INDEX_B: int = 1


class ToggleSwitchFactory(MotifFactory):
    """Factory for the toggle switch motif.

    Topology:
        R1: empty -> A  (B represses A via Hill kinetics)
        R2: A -> empty  (first-order degradation of A)
        R3: empty -> B  (A represses B via Hill kinetics)
        R4: B -> empty  (first-order degradation of B)
    """

    def motif_type(self) -> MotifType:
        """Return TOGGLE_SWITCH motif type."""
        return MotifType.TOGGLE_SWITCH

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the toggle switch motif.

        Returns:
            MotifParameterRanges with rate, Hill coefficient, and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_max_A": (10.0, 100.0),
                "k_max_B": (10.0, 100.0),
                "k_half_A": (10.0, 50.0),
                "k_half_B": (10.0, 50.0),
                "k_deg_A": (0.05, 0.5),
                "k_deg_B": (0.05, 0.5),
            },
            hill_coefficient_ranges={
                "n_A": (2.0, 4.0),
                "n_B": (2.0, 4.0),
            },
            initial_state_ranges={"A": (0, 50), "B": (0, 50)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create a toggle switch CRN from the given parameters.

        Args:
            params: Must contain k_max_A, k_max_B, k_half_A, k_half_B,
                n_A, n_B, k_deg_A, k_deg_B.

        Returns:
            CRN where A and B mutually repress each other.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0]),
                propensity=hill_repression(
                    k_max=params["k_max_A"],
                    k_half=params["k_half_A"],
                    hill_coefficient=params["n_A"],
                    species_index=_SPECIES_INDEX_B,
                ),
                name="production_A",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0]),
                propensity=mass_action(params["k_deg_A"], torch.tensor([1.0, 0.0])),
                name="degradation_A",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0]),
                propensity=hill_repression(
                    k_max=params["k_max_B"],
                    k_half=params["k_half_B"],
                    hill_coefficient=params["n_B"],
                    species_index=_SPECIES_INDEX_A,
                ),
                name="production_B",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0]),
                propensity=mass_action(params["k_deg_B"], torch.tensor([0.0, 1.0])),
                name="degradation_B",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
