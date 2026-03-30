"""Incoherent feedforward loop motif factory: X activates Z directly and represses Z via Y."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import (
    constant_rate,
    hill,
    hill_activation_repression,
    mass_action,
)
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("X", "Y", "Z")
_N_REACTIONS: int = 6

# Species indices
_IDX_X: int = 0
_IDX_Y: int = 1
_IDX_Z: int = 2


class IncoherentFeedforwardFactory(MotifFactory):
    """Factory for the incoherent type-1 feedforward loop (I1-FFL).

    Topology (species order: X, Y, Z):
        R1: empty -> X  (constitutive production of X, rate k_X)
        R2: X -> empty  (degradation of X, rate k_deg_X)
        R3: empty -> Y  (X activates Y via Hill activation)
        R4: Y -> empty  (degradation of Y, rate k_deg_Y)
        R5: empty -> Z  (X activates and Y represses Z via Hill activation-repression)
        R6: Z -> empty  (degradation of Z, rate k_deg_Z)
    """

    def motif_type(self) -> MotifType:
        """Return INCOHERENT_FEEDFORWARD motif type."""
        return MotifType.INCOHERENT_FEEDFORWARD

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the incoherent feedforward loop.

        Returns:
            MotifParameterRanges with rate, Hill coefficient, and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_X": (1.0, 20.0),
                "k_deg_X": (0.05, 0.3),
                "k_max_Y": (5.0, 50.0),
                "K_act_Y": (5.0, 30.0),
                "k_deg_Y": (0.05, 0.3),
                "k_max_Z": (5.0, 50.0),
                "K_act_Z": (5.0, 30.0),
                "K_rep_Z": (5.0, 30.0),
                "k_deg_Z": (0.05, 0.3),
            },
            hill_coefficient_ranges={
                "n_act_Y": (1.5, 3.0),
                "n_act_Z": (1.5, 3.0),
                "n_rep_Z": (1.5, 3.0),
            },
            initial_state_ranges={"X": (0, 10), "Y": (0, 0), "Z": (0, 0)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create an incoherent feedforward CRN from the given parameters.

        Args:
            params: Must contain all keys declared in parameter_ranges().

        Returns:
            CRN implementing the I1-FFL motif.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
                propensity=constant_rate(params["k_X"]),
                name="production_X",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0, 0.0]),
                propensity=mass_action(
                    params["k_deg_X"], torch.tensor([1.0, 0.0, 0.0])
                ),
                name="degradation_X",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, 0.0]),
                propensity=hill(
                    v_max=params["k_max_Y"],
                    k_m=params["K_act_Y"],
                    hill_coefficient=params["n_act_Y"],
                    species_index=_IDX_X,
                ),
                name="production_Y",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0, 0.0]),
                propensity=mass_action(
                    params["k_deg_Y"], torch.tensor([0.0, 1.0, 0.0])
                ),
                name="degradation_Y",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 1.0]),
                propensity=hill_activation_repression(
                    k_max=params["k_max_Z"],
                    k_act=params["K_act_Z"],
                    n_act=params["n_act_Z"],
                    activator_index=_IDX_X,
                    k_rep=params["K_rep_Z"],
                    n_rep=params["n_rep_Z"],
                    repressor_index=_IDX_Y,
                ),
                name="production_Z",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, -1.0]),
                propensity=mass_action(
                    params["k_deg_Z"], torch.tensor([0.0, 0.0, 1.0])
                ),
                name="degradation_Z",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
