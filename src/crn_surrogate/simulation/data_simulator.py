"""Simulation backends for dataset generation.

Provides a uniform interface for stochastic (SSA) and deterministic (ODE)
simulation of CRNs during data generation. These are NOT the neural
surrogate simulators (those live in crn_surrogate.simulator).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import torch

from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.mass_action_ode import MassActionODE
from crn_surrogate.simulation.trajectory import Trajectory


class DataSimulator(ABC):
    """Abstract interface for CRN simulation during data generation.

    All implementations return a (M, T, n_species) tensor of trajectories
    interpolated onto a shared time grid, or None on failure (timeout, blowup).
    """

    @abstractmethod
    def simulate(
        self,
        crn,
        initial_state: torch.Tensor,
        t_max: float,
        n_trajectories: int,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Simulate trajectories for a CRN.

        Args:
            crn: CRN object with stoichiometry_matrix and evaluate_propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time.
            n_trajectories: Number of independent trajectories (M).
            time_grid: (T,) time points to record states at.

        Returns:
            (M, T, n_species) tensor, or None on failure.
        """
        ...


class SSASimulator(DataSimulator):
    """Standard Gillespie SSA with optional wall-clock timeout."""

    def __init__(self, timeout: int = 30) -> None:
        """Args:
        timeout: Per-CRN wall-clock timeout in seconds. 0 disables.
        """
        self._ssa = GillespieSSA()
        self._timeout = timeout

    def simulate(
        self,
        crn,
        initial_state: torch.Tensor,
        t_max: float,
        n_trajectories: int,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Simulate SSA trajectories with optional timeout.

        Args:
            crn: CRN object with stoichiometry_matrix and evaluate_propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time.
            n_trajectories: Number of independent SSA trajectories.
            time_grid: (T,) time grid for interpolation.

        Returns:
            (M, T, n_species) tensor, or None on timeout.
        """
        trajs = self._run_with_timeout(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=initial_state,
            t_max=t_max,
            n_trajectories=n_trajectories,
        )
        if trajs is None:
            return None
        return Trajectory.stack_on_grid(trajs, time_grid)

    def _run_with_timeout(self, **kwargs) -> list[Trajectory] | None:
        """Run SSA batch with optional timeout.

        Args:
            **kwargs: Arguments forwarded to GillespieSSA.simulate_batch.

        Returns:
            List of Trajectory objects, or None on timeout.
        """
        if self._timeout <= 0:
            return self._ssa.simulate_batch(**kwargs, n_workers=1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._ssa.simulate_batch, **kwargs, n_workers=1)
            try:
                return future.result(timeout=self._timeout)
            except FuturesTimeout:
                return None


class FastSSASimulator(DataSimulator):
    """Numba-accelerated SSA with fallback to standard SSA.

    Handles JIT warmup at construction. Falls back to SSASimulator if the
    CRN has non-mass-action kinetics or if numba is unavailable.
    """

    def __init__(self, timeout: int = 30) -> None:
        """Args:
            timeout: Per-CRN wall-clock timeout in seconds. 0 disables.

        Raises:
            ImportError: If numba is not available.
        """
        import numpy as np

        from crn_surrogate.simulation.fast_ssa import (
            NUMBA_AVAILABLE,
            FastMassActionSSA,
            _gillespie_mass_action_inner,
        )

        if not NUMBA_AVAILABLE:
            raise ImportError("FastSSASimulator requires numba.")

        self._fast_ssa = FastMassActionSSA()
        self._fallback = SSASimulator(timeout=timeout)
        self._timeout = timeout

        # JIT warmup — compile on first call so it does not happen mid-generation
        _gillespie_mass_action_inner(
            np.zeros((2, 1), dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
            100,
            42,
        )

    def simulate(
        self,
        crn,
        initial_state: torch.Tensor,
        t_max: float,
        n_trajectories: int,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Simulate using Numba-accelerated SSA, falling back to standard SSA.

        Args:
            crn: CRN object. Must have mass-action kinetics for the fast path.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time.
            n_trajectories: Number of independent SSA trajectories.
            time_grid: (T,) time grid for interpolation.

        Returns:
            (M, T, n_species) tensor, or None on timeout.
        """
        from crn_surrogate.simulation.fast_ssa import FastMassActionSSA

        try:
            arrays = FastMassActionSSA.extract_topology_arrays(crn)
        except (AttributeError, ValueError):
            return self._fallback.simulate(
                crn, initial_state, t_max, n_trajectories, time_grid
            )

        if self._timeout <= 0:
            return self._fast_ssa.simulate_batch(
                stoichiometry=arrays["stoichiometry"],
                reactant_matrix=arrays["reactant_matrix"],
                rate_constants=arrays["rate_constants"],
                initial_state=initial_state,
                t_max=t_max,
                time_grid=time_grid,
                n_trajectories=n_trajectories,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._fast_ssa.simulate_batch,
                stoichiometry=arrays["stoichiometry"],
                reactant_matrix=arrays["reactant_matrix"],
                rate_constants=arrays["rate_constants"],
                initial_state=initial_state,
                t_max=t_max,
                time_grid=time_grid,
                n_trajectories=n_trajectories,
            )
            try:
                return future.result(timeout=self._timeout)
            except FuturesTimeout:
                return None


class ODESimulator(DataSimulator):
    """Deterministic ODE integration for data generation.

    Always produces M=1 trajectory (deterministic system has no
    stochastic variation). The n_trajectories argument is ignored.
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
        self._ode = MassActionODE(
            n_substeps=n_substeps,
            blowup_threshold=blowup_threshold,
        )

    def simulate(
        self,
        crn,
        initial_state: torch.Tensor,
        t_max: float,
        n_trajectories: int,
        time_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Integrate the mass-action ODE, returning a single trajectory.

        Args:
            crn: CRN object with stoichiometry_matrix and evaluate_propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time (unused; derived from time_grid).
            n_trajectories: Ignored — ODE produces exactly one trajectory.
            time_grid: (T,) time grid to record states at.

        Returns:
            (1, T, n_species) tensor, or None if blowup detected.
        """
        result = self._ode.integrate(crn, initial_state, time_grid)
        if result is None:
            return None
        # (T, n_species) -> (1, T, n_species) to match stochastic format
        return result.unsqueeze(0)
