"""Atomic unit of a Chemical Reaction Network: stoichiometry paired with a propensity callable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


PropensityFn = Callable[[torch.Tensor, float], torch.Tensor]
"""Type alias for a propensity function: (state: Tensor(n_species,), t: float) → Tensor(scalar)."""


@dataclass(frozen=True)
class Reaction:
    """A single chemical reaction: stoichiometry vector paired with a propensity function.

    Attributes:
        stoichiometry: (n_species,) integer-valued net change vector.
        propensity: Callable (state, t) → scalar propensity value.
        name: Optional display name for debugging.
    """

    stoichiometry: torch.Tensor
    propensity: PropensityFn
    name: str = ""

    def __post_init__(self) -> None:
        if self.stoichiometry.dim() != 1:
            raise ValueError(
                f"stoichiometry must be 1D, got shape {self.stoichiometry.shape}"
            )
        if not callable(self.propensity):
            raise ValueError("propensity must be callable")

    def __repr__(self) -> str:
        return (
            f"Reaction(name={self.name!r}, "
            f"stoichiometry={self.stoichiometry.tolist()}, "
            f"propensity={self.propensity!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reaction):
            return NotImplemented
        return (
            self.name == other.name
            and self.propensity is other.propensity
            and torch.equal(self.stoichiometry, other.stoichiometry)
        )

    def __hash__(self) -> int:
        return id(self)
