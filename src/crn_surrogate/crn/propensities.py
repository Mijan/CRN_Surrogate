"""Standard propensity function factories for mass-action, Hill, and constant-rate kinetics.

Each factory returns a PropensityFn callable that captures kinetic parameters at
construction time. Callables are implemented as classes (not lambdas) so that
their parameters are inspectable via a `.params` property for serialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from crn_surrogate.crn.reaction import PropensityFn


# ── Parameter dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MassActionParams:
    """Named parameters for mass-action kinetics.

    Only the rate constant is stored here. The reactant stoichiometry is a
    structural property encoded in the CRN's stoichiometry matrix and is
    not a kinetic parameter.

    Attributes:
        rate_constant: Rate constant k.
    """

    rate_constant: float

    def to_tensor(self, max_params: int = 4) -> torch.Tensor:
        """Serialize to a flat tensor of length max_params.

        Layout: [rate_constant, 0, 0, ...].

        Args:
            max_params: Length of the output tensor.

        Returns:
            Flat parameter tensor.
        """
        t = torch.zeros(max_params)
        t[0] = self.rate_constant
        return t

    @classmethod
    def from_tensor(cls, params: torch.Tensor) -> "MassActionParams":
        """Reconstruct from flat parameter tensor.

        Args:
            params: Flat tensor with rate_constant at index 0.

        Returns:
            MassActionParams instance.
        """
        return cls(rate_constant=params[0].item())


@dataclass(frozen=True)
class HillParams:
    """Named parameters for Hill-type activation kinetics.

    Attributes:
        v_max: Maximum rate V.
        k_m: Half-saturation constant K.
        hill_coefficient: Hill exponent n.
        species_index: Index of the species driving the reaction.
    """

    v_max: float
    k_m: float
    hill_coefficient: float
    species_index: int

    def to_tensor(self, max_params: int = 4) -> torch.Tensor:
        """Serialize to a flat tensor of length max_params.

        Layout: [v_max, k_m, hill_coefficient, species_index].

        Args:
            max_params: Length of the output tensor (must be >= 4).

        Returns:
            Flat parameter tensor.
        """
        t = torch.zeros(max_params)
        t[0] = self.v_max
        t[1] = self.k_m
        t[2] = self.hill_coefficient
        t[3] = self.species_index
        return t

    @classmethod
    def from_tensor(cls, params: torch.Tensor) -> HillParams:
        """Reconstruct from flat parameter tensor.

        Args:
            params: Flat tensor with layout [v_max, k_m, hill_coefficient, species_index].

        Returns:
            HillParams instance.
        """
        return cls(
            v_max=params[0].item(),
            k_m=params[1].item(),
            hill_coefficient=params[2].item(),
            species_index=int(params[3].item()),
        )


# ── Callable closure classes ──────────────────────────────────────────────────


class _MassActionClosure:
    """Callable mass-action propensity: a(X,t) = k * prod_s X_s^{R_s}."""

    def __init__(
        self, params: MassActionParams, reactant_stoichiometry: torch.Tensor
    ) -> None:
        self._params = params
        self._reactant_stoichiometry = reactant_stoichiometry

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor:
        k = self._params.rate_constant
        rs = self._reactant_stoichiometry.float()
        return k * torch.pow(state.clamp(min=0.0), rs).prod()

    @property
    def params(self) -> MassActionParams:
        """Kinetic parameter dataclass (rate constant only)."""
        return self._params

    @property
    def reactant_stoichiometry(self) -> torch.Tensor:
        """(n_species,) consumption counts for this reaction.

        Separate from params because reactant stoichiometry is structural
        information, not a kinetic parameter. Needed for serialization and
        propensity evaluation, but not part of the learnable parameter set.
        """
        return self._reactant_stoichiometry

    def __repr__(self) -> str:
        return f"MassAction(k={self._params.rate_constant})"


class _HillClosure:
    """Callable Hill activation propensity: a(X,t) = V * X_s^n / (K^n + X_s^n)."""

    def __init__(self, params: HillParams) -> None:
        self._params = params

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor:
        p = self._params
        x = state[p.species_index].clamp(min=0.0)
        x_n = torch.pow(x, p.hill_coefficient)
        k_n = p.k_m**p.hill_coefficient
        return p.v_max * x_n / (k_n + x_n + 1e-8)

    @property
    def params(self) -> HillParams:
        """Inspectable parameter dataclass."""
        return self._params

    def __repr__(self) -> str:
        return (
            f"Hill(v_max={self._params.v_max}, k_m={self._params.k_m}, "
            f"n={self._params.hill_coefficient})"
        )


class _ConstantRateClosure:
    """Callable constant propensity: a(X,t) = k. For zero-order reactions."""

    def __init__(self, k: float) -> None:
        self._k = k

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor:
        return torch.tensor(float(self._k))

    def __repr__(self) -> str:
        return f"ConstantRate(k={self._k})"


# ── Public factory functions ──────────────────────────────────────────────────


def mass_action(
    rate_constant: float,
    reactant_stoichiometry: torch.Tensor,
) -> PropensityFn:
    """Create a mass-action propensity: a(X,t) = k * prod_s X_s^{R_s}.

    The returned callable captures rate_constant and reactant_stoichiometry
    in its closure. The callable ignores t (mass-action is autonomous).

    Args:
        rate_constant: Rate constant k.
        reactant_stoichiometry: (n_species,) how many molecules of each
            species are consumed (NOT the net stoichiometry).

    Returns:
        Callable (state, t) → scalar propensity.
    """
    return _MassActionClosure(
        MassActionParams(rate_constant=rate_constant),
        reactant_stoichiometry=reactant_stoichiometry,
    )


def hill(
    v_max: float,
    k_m: float,
    hill_coefficient: float,
    species_index: int,
) -> PropensityFn:
    """Create a Hill activation propensity: a(X,t) = V * X_s^n / (K^n + X_s^n).

    The returned callable captures all parameters in its closure. It ignores t.

    Args:
        v_max: Maximum rate V.
        k_m: Half-saturation constant K.
        hill_coefficient: Hill exponent n.
        species_index: Index of the species driving the reaction.

    Returns:
        Callable (state, t) → scalar propensity.
    """
    return _HillClosure(
        HillParams(
            v_max=v_max,
            k_m=k_m,
            hill_coefficient=hill_coefficient,
            species_index=species_index,
        )
    )


def constant_rate(k: float) -> PropensityFn:
    """Create a constant propensity: a(X,t) = k. For zero-order (creation) reactions.

    Args:
        k: Constant rate.

    Returns:
        Callable (state, t) → scalar propensity.
    """
    return _ConstantRateClosure(k)


# ── Serialization protocols ───────────────────────────────────────────────────


class PropensityParams(Protocol):
    """Protocol for parameter dataclasses that support tensor serialization."""

    def to_tensor(self, max_params: int = 4) -> torch.Tensor: ...


@runtime_checkable
class SerializablePropensity(Protocol):
    """Protocol for propensity callables that expose their parameters.

    Callables implementing this protocol can be serialized to and from
    flat tensor representations via crn_to_tensor_repr / tensor_repr_to_crn.
    """

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor: ...

    @property
    def params(self) -> PropensityParams: ...
