"""Analytical CLE drift and diffusion for known CRN motifs.

These functions return the exact Chemical Langevin Equation drift and
diffusion as callables, providing ground-truth references for dynamics
visualization and residual analysis.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "birth_death_analytical",
    "lotka_volterra_analytical",
]


def birth_death_analytical(
    k_birth: float = 1.0,
    k_death: float = 0.1,
) -> dict[str, Callable | float]:
    """Analytical CLE drift and diffusion for the birth-death process.

    The Chemical Langevin Equation for birth-death is:
        dX = (k_birth − k_death · X) dt + sqrt(k_birth + k_death · X) dW

    The stationary distribution is Poisson(k_birth / k_death), so the
    stationary mean equals the stationary variance.

    Args:
        k_birth: Zero-order birth rate.
        k_death: First-order death rate.

    Returns:
        Dict with keys:
            ``drift``: callable ``(x: Tensor) -> Tensor``, vectorized over x.
            ``diffusion``: callable ``(x: Tensor) -> Tensor``, noise amplitude.
            ``stationary_mean``: float, k_birth / k_death.
            ``stationary_var``: float, k_birth / k_death (Poisson identity).
    """
    return {
        "drift": lambda x: torch.as_tensor(k_birth, dtype=x.dtype) - k_death * x,
        "diffusion": lambda x: (
            (torch.as_tensor(k_birth, dtype=x.dtype) + k_death * x)
            .clamp(min=0.0)
            .sqrt()
        ),
        "stationary_mean": k_birth / k_death,
        "stationary_var": k_birth / k_death,
    }


def lotka_volterra_analytical(
    k_prey_birth: float = 1.0,
    k_predation: float = 0.01,
    k_predator_death: float = 0.5,
) -> dict[str, Callable]:
    """Analytical CLE drift for the Lotka-Volterra predator-prey system.

    The deterministic part of the CLE is:
        d(prey)/dt   = k_prey_birth · prey − k_predation · prey · pred
        d(pred)/dt   = k_predation · prey · pred − k_predator_death · pred

    No closed-form stationary distribution exists (the system oscillates).

    Args:
        k_prey_birth: First-order prey birth rate.
        k_predation: Second-order predation rate.
        k_predator_death: First-order predator death rate.

    Returns:
        Dict with key:
            ``drift``: callable ``(prey: Tensor, pred: Tensor) -> Tensor(2,)``.
    """
    return {
        "drift": lambda prey, pred: torch.stack(
            [
                k_prey_birth * prey - k_predation * prey * pred,
                k_predation * prey * pred - k_predator_death * pred,
            ]
        ),
    }
