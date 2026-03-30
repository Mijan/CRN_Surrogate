"""Viability filter for SSA trajectory ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crn_surrogate.data.generation.configs import CurationConfig


@dataclass(frozen=True)
class CurationResult:
    """Result of a viability check.

    Attributes:
        viable: True if the trajectory ensemble passes all criteria.
        rejection_reason: Empty string if viable; otherwise a short identifier
            for the first failed criterion.
    """

    viable: bool
    rejection_reason: str


class ViabilityFilter:
    """Applies a sequence of viability criteria to an SSA trajectory ensemble.

    Criteria checked in order:
        1. No NaN or Inf values.
        2. No population blowup (max value below threshold).
        3. Not stuck at zero (zero fraction below threshold).
        4. Minimum number of state transitions.
        5. Non-trivial dynamics (CV above threshold).
        6. Bounded final state (mean final population below threshold).
    """

    def __init__(self, config: CurationConfig) -> None:
        """Args:
        config: Curation configuration with thresholds for each criterion.
        """
        self._config = config

    def check(self, trajectories: torch.Tensor) -> CurationResult:
        """Check viability, returning a CurationResult with a rejection_reason.

        Args:
            trajectories: (M, T, n_species) SSA trajectory ensemble.

        Returns:
            CurationResult with viable=True if all criteria pass, otherwise
            viable=False and a short rejection_reason string.
        """
        cfg = self._config

        if not torch.isfinite(trajectories).all():
            return CurationResult(viable=False, rejection_reason="nan_or_inf")

        if trajectories.max() > cfg.blowup_threshold:
            return CurationResult(viable=False, rejection_reason="blowup")

        all_zero = (trajectories == 0).all(dim=-1)  # (M, T) bool
        zero_fraction = all_zero.float().mean().item()
        if zero_fraction > cfg.max_zero_fraction:
            return CurationResult(viable=False, rejection_reason="zero_stuck")

        diff = (trajectories[:, 1:, :] != trajectories[:, :-1, :]).any(
            dim=-1
        )  # (M, T-1)
        n_transitions = diff.sum().item()
        if n_transitions < cfg.min_reactions_fired:
            return CurationResult(viable=False, rejection_reason="low_activity")

        traj_mean = trajectories.mean(dim=1)  # (M, n_species)
        traj_std = trajectories.std(dim=1)  # (M, n_species)
        cv = (traj_std / (traj_mean.abs() + 1e-8)).mean(dim=0)  # (n_species,)
        max_cv = cv.max().item()
        if max_cv < cfg.min_coefficient_of_variation:
            return CurationResult(viable=False, rejection_reason="low_cv")

        final_mean = trajectories[:, -10:, :].mean().item()
        if final_mean > cfg.max_final_population:
            return CurationResult(viable=False, rejection_reason="unbounded_final")

        return CurationResult(viable=True, rejection_reason="")

    def is_viable(self, trajectories: torch.Tensor) -> bool:
        """Return True if the trajectory ensemble passes all viability criteria.

        Args:
            trajectories: (M, T, n_species) SSA trajectory ensemble.

        Returns:
            True if viable, False otherwise.
        """
        return self.check(trajectories).viable
