"""Substrate inhibition motif factory: enzyme kinetics with substrate self-inhibition at high concentrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import (
    constant_rate,
    mass_action,
    substrate_inhibition,
)
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    param_field,
)


@dataclass(frozen=True)
class SubstrateInhibitionParams:
    """Parameters for the substrate inhibition motif.

    Attributes:
        k_in: Substrate input (constitutive production) rate.
        V_max: Maximum conversion rate.
        K_m: Michaelis constant (substrate affinity).
        K_i: Substrate inhibition constant. Must exceed K_m.
        k_deg: Product degradation rate.
    """

    k_in: float = param_field(1e-4, 1e3) # param_field(0.5, 20.0)
    V_max: float = param_field(1e-4, 1e3) # param_field(1.0, 50.0)
    K_m: float = param_field(1e-4, 1e3) # param_field(5.0, 50.0)
    K_i: float = param_field(1e-4, 1e3) # param_field(20.0, 200.0)
    k_deg: float = param_field(1e-4, 1e3) # param_field(0.01, 0.3)


class SubstrateInhibitionMotifFactory(MotifFactory[SubstrateInhibitionParams]):
    """Factory for the substrate inhibition motif.

    Topology (species order: S, P):
        R1: empty -> S        (substrate input, rate k_in)
        R2: S -> P            (substrate-inhibited conversion, V_max, K_m, K_i)
        R3: P -> empty        (product degradation, rate k_deg)

    Constraint: K_i > K_m (inhibition constant must exceed Michaelis constant).
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("S", "P")
    _N_SPECIES: ClassVar[int] = 2
    _N_REACTIONS: ClassVar[int] = 3
    _IDX_S: ClassVar[int] = 0
    _IDX_P: ClassVar[int] = 1

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
        return MotifType.SUBSTRATE_INHIBITION

    @property
    def params_type(self) -> type[SubstrateInhibitionParams]:
        return SubstrateInhibitionParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the substrate inhibition motif.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[self._IDX_S]: InitialStateRange(0, 20),
            self._species_names[self._IDX_P]: InitialStateRange(0, 0),
        }

    def validate_params(self, params: SubstrateInhibitionParams) -> None:
        """Enforce K_i > K_m for valid substrate inhibition kinetics.

        Args:
            params: SubstrateInhibitionParams to validate.

        Raises:
            TypeError: If params is not SubstrateInhibitionParams.
            ValueError: If K_i <= K_m.
        """
        super().validate_params(params)
        if params.K_i <= params.K_m:
            raise ValueError(
                f"Substrate inhibition requires K_i > K_m, "
                f"got K_m={params.K_m}, K_i={params.K_i}"
            )

    def create(self, params: SubstrateInhibitionParams) -> CRN:
        """Create a substrate inhibition CRN from the given parameters.

        Args:
            params: SubstrateInhibitionParams with k_in, V_max, K_m, K_i, k_deg.

        Returns:
            CRN with substrate-inhibited conversion kinetics.
        """
        self.validate_params(params)
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0]),
                propensity=constant_rate(params.k_in),
                name="substrate_input",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 1.0]),
                propensity=substrate_inhibition(
                    v_max=params.V_max,
                    k_m=params.K_m,
                    k_i=params.K_i,
                    species_index=self._IDX_S,
                ),
                name="conversion",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0]),
                propensity=mass_action(params.k_deg, torch.tensor([0.0, 1.0])),
                name="product_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
