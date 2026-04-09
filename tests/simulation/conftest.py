"""Shared fixtures for simulation tests.

Provides lightweight CRN-like objects that satisfy the interface expected
by GillespieSSA and MassActionODE without importing the full CRN module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch


@dataclass
class StubCRN:
    """Minimal CRN-like object for simulation tests.

    Satisfies the interface: .stoichiometry_matrix, .evaluate_propensities(),
    .n_species, .n_reactions.
    """

    stoichiometry_matrix: torch.Tensor
    _propensity_fn: Callable[[torch.Tensor, float], torch.Tensor]
    n_species: int
    n_reactions: int

    def evaluate_propensities(self, state: torch.Tensor, t: float) -> torch.Tensor:
        return self._propensity_fn(state, t)


def _birth_death_crn(k_birth: float = 2.0, k_death: float = 0.5) -> StubCRN:
    """Birth-death process: ∅ -> X (rate k_birth), X -> ∅ (rate k_death*X).

    Analytical stationary mean = k_birth / k_death.
    """
    stoich = torch.tensor(
        [
            [1.0],  # birth: +1
            [-1.0],  # death: -1
        ]
    )

    def propensity_fn(state: torch.Tensor, t: float) -> torch.Tensor:
        x = state[0].clamp(min=0.0)
        return torch.tensor([k_birth, k_death * x])

    return StubCRN(
        stoichiometry_matrix=stoich,
        _propensity_fn=propensity_fn,
        n_species=1,
        n_reactions=2,
    )


def _decay_crn(k: float = 0.1) -> StubCRN:
    """Pure decay: X -> ∅ (rate k*X).

    Analytical ODE solution: x(t) = x0 * exp(-k*t).
    """
    stoich = torch.tensor([[-1.0]])

    def propensity_fn(state: torch.Tensor, t: float) -> torch.Tensor:
        return torch.tensor([k * state[0].clamp(min=0.0)])

    return StubCRN(
        stoichiometry_matrix=stoich,
        _propensity_fn=propensity_fn,
        n_species=1,
        n_reactions=1,
    )


def _two_species_crn() -> StubCRN:
    """A -> B (rate 0.5*A), B -> A (rate 0.3*B).

    Two species, two reactions. Useful for shape tests.
    """
    stoich = torch.tensor(
        [
            [-1.0, 1.0],  # A -> B
            [1.0, -1.0],  # B -> A
        ]
    )

    def propensity_fn(state: torch.Tensor, t: float) -> torch.Tensor:
        a, b = state[0].clamp(min=0.0), state[1].clamp(min=0.0)
        return torch.tensor([0.5 * a, 0.3 * b])

    return StubCRN(
        stoichiometry_matrix=stoich,
        _propensity_fn=propensity_fn,
        n_species=2,
        n_reactions=2,
    )


@pytest.fixture
def birth_death_crn() -> StubCRN:
    return _birth_death_crn()


@pytest.fixture
def decay_crn() -> StubCRN:
    return _decay_crn()


@pytest.fixture
def two_species_crn() -> StubCRN:
    return _two_species_crn()
