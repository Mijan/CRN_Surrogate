"""Standard propensity function factories for mass-action, Hill, enzyme-catalyzed MM, and constant-rate kinetics.

Each factory returns a PropensityFn callable that captures kinetic parameters at
construction time. Callables are implemented as classes (not lambdas) so that
their parameters are inspectable via a `.params` property and their species
dependencies are declared via `.species_dependencies`.
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

    Only the rate constant is stored. The reactant stoichiometry is a
    structural property captured in the closure for evaluation.

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
    def from_tensor(cls, params: torch.Tensor) -> "HillParams":
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


@dataclass(frozen=True)
class ConstantRateParams:
    """Parameters for constant-rate (zero-order) kinetics.

    Attributes:
        rate: Constant propensity value (independent of state).
    """

    rate: float

    def to_tensor(self, max_params: int = 4) -> torch.Tensor:
        """Serialize to a flat tensor of length max_params.

        Layout: [rate, 0, 0, ...].

        Args:
            max_params: Length of the output tensor.

        Returns:
            Flat parameter tensor.
        """
        t = torch.zeros(max_params)
        t[0] = self.rate
        return t

    @classmethod
    def from_tensor(cls, params: torch.Tensor) -> "ConstantRateParams":
        """Reconstruct from flat parameter tensor.

        Args:
            params: Flat tensor with rate at index 0.

        Returns:
            ConstantRateParams instance.
        """
        return cls(rate=params[0].item())


@dataclass(frozen=True)
class EnzymeMichaelisMentenParams:
    """Parameters for enzyme-catalyzed Michaelis-Menten kinetics.

    Attributes:
        k_cat: Catalytic rate constant.
        k_m: Michaelis constant.
        enzyme_index: Index of the enzyme species.
        substrate_index: Index of the substrate species.
    """

    k_cat: float
    k_m: float
    enzyme_index: int
    substrate_index: int

    def to_tensor(self, max_params: int = 4) -> torch.Tensor:
        """Serialize to a flat tensor of length max_params.

        Layout: [k_cat, k_m, enzyme_index, substrate_index].

        Args:
            max_params: Length of the output tensor (must be >= 4).

        Returns:
            Flat parameter tensor.
        """
        t = torch.zeros(max_params)
        t[0] = self.k_cat
        t[1] = self.k_m
        t[2] = self.enzyme_index
        t[3] = self.substrate_index
        return t

    @classmethod
    def from_tensor(cls, params: torch.Tensor) -> "EnzymeMichaelisMentenParams":
        """Reconstruct from flat parameter tensor.

        Args:
            params: Flat tensor with layout [k_cat, k_m, enzyme_index, substrate_index].

        Returns:
            EnzymeMichaelisMentenParams instance.
        """
        return cls(
            k_cat=params[0].item(),
            k_m=params[1].item(),
            enzyme_index=int(params[2].item()),
            substrate_index=int(params[3].item()),
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
    def species_dependencies(self) -> frozenset[int]:
        """Indices of species that influence this propensity (nonzero reactant order)."""
        return frozenset(
            int(i)
            for i, r in enumerate(self._reactant_stoichiometry.tolist())
            if r != 0.0
        )

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

    @property
    def species_dependencies(self) -> frozenset[int]:
        """Index of the species driving this Hill propensity."""
        return frozenset({self._params.species_index})

    def __repr__(self) -> str:
        return (
            f"Hill(v_max={self._params.v_max}, k_m={self._params.k_m}, "
            f"n={self._params.hill_coefficient})"
        )


class _ConstantRateClosure:
    """Callable constant propensity: a(X,t) = k. For zero-order reactions."""

    def __init__(self, params: ConstantRateParams) -> None:
        self._params = params

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor:
        return torch.tensor(self._params.rate)

    @property
    def params(self) -> ConstantRateParams:
        """Inspectable parameter dataclass."""
        return self._params

    @property
    def species_dependencies(self) -> frozenset[int]:
        """No species influence a constant-rate propensity."""
        return frozenset()

    def __repr__(self) -> str:
        return f"ConstantRate(k={self._params.rate})"


class _EnzymeMichaelisMentenClosure:
    """Callable enzyme-catalyzed Michaelis-Menten propensity.

    a(X, t) = k_cat * X_enzyme * X_substrate / (K_m + X_substrate)
    """

    def __init__(self, params: EnzymeMichaelisMentenParams) -> None:
        self._params = params
        self._species_dependencies = frozenset(
            {params.enzyme_index, params.substrate_index}
        )

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor:
        p = self._params
        e = state[p.enzyme_index].clamp(min=0.0)
        s = state[p.substrate_index].clamp(min=0.0)
        return p.k_cat * e * s / (p.k_m + s + 1e-8)

    @property
    def params(self) -> EnzymeMichaelisMentenParams:
        """Inspectable parameter dataclass."""
        return self._params

    @property
    def species_dependencies(self) -> frozenset[int]:
        """Enzyme and substrate both influence this propensity."""
        return self._species_dependencies

    def __repr__(self) -> str:
        p = self._params
        return (
            f"EnzymeMM(k_cat={p.k_cat}, K_m={p.k_m}, "
            f"E={p.enzyme_index}, S={p.substrate_index})"
        )


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
    """Create a constant propensity: a(X,t) = k.

    For zero-order reactions (constitutive production). The propensity is
    independent of all species and of time.

    Args:
        k: Constant rate.

    Returns:
        Callable (state, t) → scalar propensity.
    """
    return _ConstantRateClosure(ConstantRateParams(rate=k))


def enzyme_michaelis_menten(
    k_cat: float,
    k_m: float,
    enzyme_index: int,
    substrate_index: int,
) -> PropensityFn:
    """Create an enzyme-catalyzed Michaelis-Menten propensity.

    a(X, t) = k_cat * X_enzyme * X_substrate / (K_m + X_substrate)

    The enzyme participates catalytically (zero net stoichiometry for the
    enzyme) but influences the rate. Both enzyme_index and substrate_index
    are declared as species dependencies.

    Args:
        k_cat: Catalytic rate constant.
        k_m: Michaelis constant.
        enzyme_index: Index of the enzyme species.
        substrate_index: Index of the substrate species.

    Returns:
        Callable (state, t) → scalar propensity.
    """
    return _EnzymeMichaelisMentenClosure(
        EnzymeMichaelisMentenParams(
            k_cat=k_cat,
            k_m=k_m,
            enzyme_index=enzyme_index,
            substrate_index=substrate_index,
        )
    )


# ── Serialization protocols ───────────────────────────────────────────────────


class PropensityParams(Protocol):
    """Protocol for parameter dataclasses that support tensor serialization."""

    def to_tensor(self, max_params: int = 4) -> torch.Tensor: ...


@runtime_checkable
class SerializablePropensity(Protocol):
    """Protocol for propensity callables that expose their parameters and dependencies.

    Callables implementing this protocol can be serialized to and from
    flat tensor representations via crn_to_tensor_repr / tensor_repr_to_crn.
    """

    def __call__(self, state: torch.Tensor, t: float) -> torch.Tensor: ...

    @property
    def params(self) -> PropensityParams: ...

    @property
    def species_dependencies(self) -> frozenset[int]: ...
