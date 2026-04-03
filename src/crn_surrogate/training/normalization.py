"""Per-item trajectory normalization for scale-invariant training.

The TrajectoryNormalizer captures the min_scale policy at construction time
and exposes compute_scale / normalize / denormalize as a cohesive unit.
"""

from __future__ import annotations

import torch


class TrajectoryNormalizer:
    """Per-species, per-item trajectory normalizer.

    Captures the min_scale floor so callers never need to re-specify it.
    The three-step pattern is:

        scale = normalizer.compute_scale(ssa_trajs)
        normed = normalizer.normalize(trajs, scale)
        # ... model forward pass ...
        counts = normalizer.denormalize(normed, scale)

    Attributes:
        min_scale: Minimum allowed scale value. Species whose mean absolute
            count falls below this floor are not amplified.
    """

    def __init__(self, min_scale: float = 1.0) -> None:
        """Args:
        min_scale: Minimum scale per species. Default 1.0 prevents
            amplification of near-zero species.
        """
        self._min_scale = min_scale

    @property
    def min_scale(self) -> float:
        """Minimum scale floor applied by compute_scale."""
        return self._min_scale

    def compute_scale(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute per-species normalization scale from an SSA ensemble.

        The scale is the mean absolute value across all samples and time
        steps, clamped below by min_scale.

        Args:
            trajectories: (M, T, n_species) SSA ensemble.

        Returns:
            (n_species,) scale tensor.
        """
        return trajectories.abs().mean(dim=(0, 1)).clamp(min=self._min_scale)

    def normalize(
        self,
        trajectories: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Divide trajectories by per-species scale.

        Args:
            trajectories: (..., n_species) tensor.
            scale: (n_species,) scale from compute_scale.

        Returns:
            Normalized trajectories of the same shape.
        """
        return trajectories / scale

    def denormalize(
        self,
        trajectories: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply trajectories by per-species scale (inverse of normalize).

        Args:
            trajectories: (..., n_species) tensor in normalized space.
            scale: (n_species,) scale from compute_scale.

        Returns:
            Denormalized trajectories in count space.
        """
        return trajectories * scale
