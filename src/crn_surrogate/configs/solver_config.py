"""Configuration for numerical solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """Solver-specific settings, independent of neural network architecture.

    Attributes:
        clip_state: Clamp species counts to [0, inf) after each integration step.
    """

    clip_state: bool = True
