"""Forward Euler integrator for mass-action ODE systems.

Integrates the deterministic rate equations:
    dx/dt = S^T * a(x)
where S is the stoichiometry matrix and a(x) are the propensities.
"""

from __future__ import annotations

import numpy as np
import torch


class MassActionODE:
    """Deterministic ODE integrator for CRNs via forward Euler with substeps.

    Operates on the CRN's propensity function directly (no neural network).
    Intended for data generation and pre-screening, not for training.
    """

    def __init__(
        self,
        n_substeps: int = 10,
        blowup_threshold: float = 1e5,
    ) -> None:
        """Args:
            n_substeps: Euler substeps between consecutive time grid points.
            blowup_threshold: Abort and return None if any state exceeds this.
        """
        self._n_substeps = n_substeps
        self._blowup_threshold = blowup_threshold

    def integrate(
        self,
        crn,
        initial_state: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Integrate the mass-action ODE on a time grid.

        Args:
            crn: A CRN object with .stoichiometry_matrix and .evaluate_propensities().
            initial_state: (n_species,) initial molecule counts.
            time_grid: (T,) time points at which to record the state.

        Returns:
            (T, n_species) trajectory tensor, or None if blowup detected.
        """
        T = len(time_grid)
        n_species = initial_state.shape[0]
        stoich = crn.stoichiometry_matrix.numpy().T  # (n_species, n_reactions)

        x = initial_state.numpy().astype(np.float64).copy()
        recorded = np.zeros((T, n_species), dtype=np.float64)
        recorded[0] = x

        for t_idx in range(1, T):
            dt_segment = (time_grid[t_idx] - time_grid[t_idx - 1]).item()
            dt_sub = dt_segment / self._n_substeps

            for _ in range(self._n_substeps):
                x_clamped = np.maximum(x, 0.0)
                x_tensor = torch.tensor(x_clamped, dtype=torch.float32)
                props = crn.evaluate_propensities(x_tensor, 0.0).numpy()
                dx = stoich @ props
                x = x + dt_sub * dx
                x = np.maximum(x, 0.0)

                if np.any(x > self._blowup_threshold) or np.any(np.isnan(x)):
                    return None

            recorded[t_idx] = x

        return torch.tensor(recorded, dtype=torch.float32)
