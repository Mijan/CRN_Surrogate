"""Forward Euler integrator for mass-action ODE systems.

Integrates the deterministic rate equations:
    dx/dt = S^T * a(x)
where S is the stoichiometry matrix and a(x) are the propensities.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op decorator matching numba.njit's interface when numba is absent."""

        def decorator(fn):
            return fn

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _euler_mass_action_inner(
    stoichiometry: np.ndarray,
    reactant_matrix: np.ndarray,
    rate_constants: np.ndarray,
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    n_substeps: int,
    blowup_threshold: float,
) -> tuple[np.ndarray, bool]:
    """Euler-integrate mass-action ODE entirely in compiled code.

    Propensity computation is identical to _gillespie_mass_action_inner:
        a_r = k_r * prod_s max(x_s, 0)^R_{r,s}

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        reactant_matrix: (n_reactions, n_species) reactant stoichiometry.
        rate_constants: (n_reactions,) rate constants.
        initial_state: (n_species,) initial molecule counts.
        time_grid: (T,) time points to record states at.
        n_substeps: Euler substeps between consecutive grid points.
        blowup_threshold: Abort if any state exceeds this.

    Returns:
        ((T, n_species) array, success: bool). On blowup, the array
        contains partial results up to the failure point and success=False.
    """
    T = time_grid.shape[0]
    n_species = stoichiometry.shape[1]
    n_reactions = stoichiometry.shape[0]

    # Pre-transpose stoichiometry for efficient dx computation
    stoich_T = np.empty((n_species, n_reactions), dtype=np.float64)
    for s in range(n_species):
        for r in range(n_reactions):
            stoich_T[s, r] = stoichiometry[r, s]

    x = initial_state.copy()
    recorded = np.empty((T, n_species), dtype=np.float64)
    for s in range(n_species):
        recorded[0, s] = x[s]

    for t_idx in range(1, T):
        dt_sub = (time_grid[t_idx] - time_grid[t_idx - 1]) / n_substeps

        for _ in range(n_substeps):
            # Compute mass-action propensities
            props = np.empty(n_reactions, dtype=np.float64)
            for r in range(n_reactions):
                a = rate_constants[r]
                for s in range(n_species):
                    power = reactant_matrix[r, s]
                    if power > 0.0:
                        val = max(x[s], 0.0)
                        if val <= 0.0:
                            a = 0.0
                            break
                        a *= val**power
                props[r] = a

            # Euler step: x += dt * S^T @ props, clamped to zero
            for s in range(n_species):
                dx_s = 0.0
                for r in range(n_reactions):
                    dx_s += stoich_T[s, r] * props[r]
                x[s] = max(x[s] + dt_sub * dx_s, 0.0)

                # Blowup / NaN check
                if x[s] > blowup_threshold or x[s] != x[s]:
                    return recorded, False

        for s in range(n_species):
            recorded[t_idx, s] = x[s]

    return recorded, True


# Sentinel returned by _integrate_fast when the CRN is not mass-action.
_NOT_MASS_ACTION = object()


class MassActionODE:
    """Deterministic ODE integrator for CRNs via forward Euler with substeps.

    Uses a Numba-compiled inner loop for mass-action CRNs when numba is
    available, falling back to a pure-Python/NumPy loop for non-mass-action
    CRNs or when numba is absent. Intended for data generation and
    pre-screening, not for training.
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

        Uses the Numba fast path for mass-action CRNs when available;
        falls back to the Python path for non-mass-action kinetics.

        Args:
            crn: A CRN object with .stoichiometry_matrix and .evaluate_propensities().
            initial_state: (n_species,) initial molecule counts.
            time_grid: (T,) time points at which to record the state.

        Returns:
            (T, n_species) trajectory tensor, or None if blowup detected.
        """
        if NUMBA_AVAILABLE:
            result = self._integrate_fast(crn, initial_state, time_grid)
            if result is not _NOT_MASS_ACTION:
                return result

        return self._integrate_python(crn, initial_state, time_grid)

    def _integrate_fast(
        self,
        crn,
        initial_state: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None | object:
        """Try numba-accelerated integration.

        Args:
            crn: CRN object.
            initial_state: (n_species,) initial molecule counts.
            time_grid: (T,) time points.

        Returns:
            (T, n_species) tensor on success, None on blowup,
            or _NOT_MASS_ACTION sentinel if the CRN has non-mass-action kinetics.
        """
        try:
            from crn_surrogate.simulation.fast_ssa import FastMassActionSSA

            arrays = FastMassActionSSA.extract_topology_arrays(crn)
        except (ValueError, AttributeError, ImportError):
            return _NOT_MASS_ACTION

        recorded, success = _euler_mass_action_inner(
            stoichiometry=arrays["stoichiometry"],
            reactant_matrix=arrays["reactant_matrix"],
            rate_constants=arrays["rate_constants"],
            initial_state=initial_state.numpy().astype(np.float64),
            time_grid=time_grid.numpy().astype(np.float64),
            n_substeps=self._n_substeps,
            blowup_threshold=self._blowup_threshold,
        )

        if not success:
            return None
        return torch.tensor(recorded, dtype=torch.float32)

    def _integrate_python(
        self,
        crn,
        initial_state: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Pure Python/NumPy Euler integration. Handles any propensity type.

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
