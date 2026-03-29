from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch


class PropensityType(Enum):
    """Categorical propensity function type."""

    MASS_ACTION = 0
    HILL = 1


class PropensityFunction(ABC):
    """Abstract base class for propensity functions."""

    @abstractmethod
    def evaluate(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute propensity value given current state and parameters.

        Args:
            state: (n_species,) current molecule counts.
            params: (max_params,) function-specific parameters.

        Returns:
            Scalar propensity value.
        """


class MassActionPropensity(PropensityFunction):
    """Mass-action propensity: a(X) = k * prod_s X_s^{R[r,s]}.

    Uses the continuous relaxation X^R for differentiability.
    params[0] is the rate constant k.
    """

    def __init__(self, reactant_stoich: torch.Tensor) -> None:
        """Args:
        reactant_stoich: (n_species,) reactant stoichiometry for this reaction.
        """
        self._reactant_stoich = reactant_stoich

    def evaluate(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute mass-action propensity."""
        k = params[0]
        # continuous relaxation: X^R (elementwise power)
        powers = torch.pow(state.clamp(min=0.0), self._reactant_stoich.float())
        return k * powers.prod()


class HillPropensity(PropensityFunction):
    """Hill-type propensity: a(X) = V_max * X_s^n / (K_m^n + X_s^n).

    params = [V_max, K_m, n, species_index].
    """

    def evaluate(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute Hill propensity."""
        v_max = params[0]
        k_m = params[1]
        n = params[2]
        s = int(params[3].item())
        x = state[s].clamp(min=0.0)
        x_n = torch.pow(x, n)
        k_n = torch.pow(k_m, n)
        return v_max * x_n / (k_n + x_n + 1e-8)


def make_propensity(
    ptype: PropensityType, reactant_stoich: torch.Tensor
) -> PropensityFunction:
    """Factory: instantiate the correct PropensityFunction for a reaction.

    Args:
        ptype: The propensity type enum value.
        reactant_stoich: (n_species,) reactant stoichiometry row for this reaction.

    Returns:
        Instantiated PropensityFunction.
    """
    if ptype == PropensityType.MASS_ACTION:
        return MassActionPropensity(reactant_stoich)
    if ptype == PropensityType.HILL:
        return HillPropensity()
    raise ValueError(f"Unknown propensity type: {ptype}")
