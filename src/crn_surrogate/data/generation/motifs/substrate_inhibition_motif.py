"""Substrate inhibition motif factory: enzyme kinetics with substrate self-inhibition at high concentrations."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import (
    constant_rate,
    mass_action,
    substrate_inhibition,
)
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges

_SPECIES: tuple[str, ...] = ("S", "P")
_N_REACTIONS: int = 3

# Species indices
_IDX_S: int = 0
_IDX_P: int = 1


class SubstrateInhibitionMotifFactory(MotifFactory):
    """Factory for the substrate inhibition motif.

    Topology (species order: S, P):
        R1: empty -> S        (substrate input, rate k_in)
        R2: S -> P            (substrate-inhibited conversion, V_max, K_m, K_i)
        R3: P -> empty        (product degradation, rate k_deg)
    """

    def motif_type(self) -> MotifType:
        """Return SUBSTRATE_INHIBITION motif type."""
        return MotifType.SUBSTRATE_INHIBITION

    def species_names(self) -> tuple[str, ...]:
        """Return species names."""
        return _SPECIES

    def n_reactions(self) -> int:
        """Return number of reactions."""
        return _N_REACTIONS

    def parameter_ranges(self) -> MotifParameterRanges:
        """Return parameter ranges for the substrate inhibition motif.

        Returns:
            MotifParameterRanges with rate and initial-state bounds.
        """
        return MotifParameterRanges(
            rate_ranges={
                "k_in": (0.5, 20.0),
                "V_max": (1.0, 50.0),
                "K_m": (5.0, 50.0),
                "K_i": (20.0, 200.0),
                "k_deg": (0.01, 0.3),
            },
            hill_coefficient_ranges={},
            initial_state_ranges={"S": (0, 20), "P": (0, 0)},
        )

    def create(self, params: dict[str, float]) -> CRN:
        """Create a substrate inhibition CRN from the given parameters.

        Args:
            params: Must contain k_in, V_max, K_m, K_i, k_deg.

        Returns:
            CRN with substrate-inhibited conversion kinetics.
        """
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0]),
                propensity=constant_rate(params["k_in"]),
                name="substrate_input",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 1.0]),
                propensity=substrate_inhibition(
                    v_max=params["V_max"],
                    k_m=params["K_m"],
                    k_i=params["K_i"],
                    species_index=_IDX_S,
                ),
                name="conversion",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0]),
                propensity=mass_action(params["k_deg"], torch.tensor([0.0, 1.0])),
                name="product_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(_SPECIES))
