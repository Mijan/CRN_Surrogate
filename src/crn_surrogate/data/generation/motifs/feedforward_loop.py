"""Incoherent feedforward loop motif factory: X activates Z directly and represses Z via Y."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

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
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    param_field,
)


@dataclass(frozen=True)
class IncoherentFeedforwardParams:
    """Parameters for the incoherent feedforward loop motif.

    Attributes:
        k_X: Constitutive production rate of X.
        k_deg_X: Degradation rate of X.
        k_max_Y: Maximum production rate of Y (activated by X).
        K_act_Y: Half-activation concentration for X-mediated Y production.
        n_act_Y: Hill exponent for X-mediated Y activation.
        k_deg_Y: Degradation rate of Y.
        k_max_Z: Maximum production rate of Z.
        K_act_Z: Half-activation concentration for X-mediated Z activation.
        n_act_Z: Hill exponent for X-mediated Z activation.
        K_rep_Z: Half-repression concentration for Y-mediated Z repression.
        n_rep_Z: Hill exponent for Y-mediated Z repression.
        k_deg_Z: Degradation rate of Z.
    """

    k_X: float = param_field(1e-4, 1e3) # param_field(1.0, 20.0)
    k_deg_X: float = param_field(1e-4, 1e3) # param_field(0.05, 0.3)
    k_max_Y: float = param_field(1e-4, 1e3) # param_field(5.0, 50.0)
    K_act_Y: float = param_field(1e-4, 1e3) # param_field(5.0, 30.0)
    n_act_Y: float = param_field(1e-4, 1e3) # param_field(1.5, 3.0, log_uniform=False)
    k_deg_Y: float = param_field(1e-4, 1e3) # param_field(0.05, 0.3)
    k_max_Z: float = param_field(1e-4, 1e3) # param_field(5.0, 50.0)
    K_act_Z: float = param_field(1e-4, 1e3) # param_field(5.0, 30.0)
    n_act_Z: float = param_field(1e-4, 1e3) # param_field(1.5, 3.0, log_uniform=False)
    K_rep_Z: float = param_field(1e-4, 1e3) # param_field(5.0, 30.0)
    n_rep_Z: float = param_field(1e-4, 1e3) # param_field(1.5, 3.0, log_uniform=False)
    k_deg_Z: float = param_field(1e-4, 1e3) # param_field(0.05, 0.3)


class IncoherentFeedforwardFactory(MotifFactory[IncoherentFeedforwardParams]):
    """Factory for the incoherent type-1 feedforward loop (I1-FFL).

    Topology (species order: X, Y, Z):
        R1: empty -> X  (constitutive production of X, rate k_X)
        R2: X -> empty  (degradation of X, rate k_deg_X)
        R3: empty -> Y  (X activates Y via Hill activation)
        R4: Y -> empty  (degradation of Y, rate k_deg_Y)
        R5: empty -> Z  (X activates and Y represses Z via Hill activation-repression)
        R6: Z -> empty  (degradation of Z, rate k_deg_Z)
    """

    _DEFAULT_SPECIES: ClassVar[tuple[str, ...]] = ("X", "Y", "Z")
    _N_SPECIES: ClassVar[int] = 3
    _N_REACTIONS: ClassVar[int] = 6
    _IDX_X: ClassVar[int] = 0
    _IDX_Y: ClassVar[int] = 1
    _IDX_Z: ClassVar[int] = 2

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
        return MotifType.INCOHERENT_FEEDFORWARD

    @property
    def params_type(self) -> type[IncoherentFeedforwardParams]:
        return IncoherentFeedforwardParams

    def initial_state_ranges(self) -> dict[str, InitialStateRange]:
        """Return initial state ranges for the incoherent feedforward loop.

        Returns:
            Dict mapping species name to InitialStateRange.
        """
        return {
            self._species_names[self._IDX_X]: InitialStateRange(0, 10),
            self._species_names[self._IDX_Y]: InitialStateRange(0, 0),
            self._species_names[self._IDX_Z]: InitialStateRange(0, 0),
        }

    def create(self, params: IncoherentFeedforwardParams) -> CRN:
        """Create an incoherent feedforward CRN from the given parameters.

        Args:
            params: IncoherentFeedforwardParams with all kinetic fields.

        Returns:
            CRN implementing the I1-FFL motif.
        """
        self.validate_params(params)
        x_name = self._species_names[self._IDX_X]
        y_name = self._species_names[self._IDX_Y]
        z_name = self._species_names[self._IDX_Z]
        reactions = [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
                propensity=constant_rate(params.k_X),
                name=f"{x_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0, 0.0]),
                propensity=mass_action(params.k_deg_X, torch.tensor([1.0, 0.0, 0.0])),
                name=f"{x_name}_degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 1.0, 0.0]),
                propensity=hill(
                    v_max=params.k_max_Y,
                    k_m=params.K_act_Y,
                    hill_coefficient=params.n_act_Y,
                    species_index=self._IDX_X,
                ),
                name=f"{y_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0, 0.0]),
                propensity=mass_action(params.k_deg_Y, torch.tensor([0.0, 1.0, 0.0])),
                name=f"{y_name}_degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, 1.0]),
                propensity=hill_activation_repression(
                    k_max=params.k_max_Z,
                    k_act=params.K_act_Z,
                    n_act=params.n_act_Z,
                    activator_index=self._IDX_X,
                    k_rep=params.K_rep_Z,
                    n_rep=params.n_rep_Z,
                    repressor_index=self._IDX_Y,
                ),
                name=f"{z_name}_production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, -1.0]),
                propensity=mass_action(params.k_deg_Z, torch.tensor([0.0, 0.0, 1.0])),
                name=f"{z_name}_degradation",
            ),
        ]
        return CRN(reactions=reactions, species_names=list(self._species_names))
