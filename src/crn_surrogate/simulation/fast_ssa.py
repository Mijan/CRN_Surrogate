"""Numba-accelerated SSA for mass-action CRNs.

Provides a JIT-compiled Gillespie inner loop that operates on numpy arrays.
Falls back gracefully if numba is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from crn_surrogate.crn.crn import CRN

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
def _gillespie_mass_action_inner(
    stoichiometry: np.ndarray,
    reactant_matrix: np.ndarray,
    rate_constants: np.ndarray,
    initial_state: np.ndarray,
    t_max: float,
    max_reactions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact Gillespie SSA for a mass-action CRN.

    Propensities follow the macroscopic mass-action law:
        a_r = k_r * prod_s max(X_s, 0)^R_{r,s}

    This matches the formula used by the Python _MassActionClosure
    (k * prod(state.clamp(min=0) ** reactant_stoichiometry)).
    Zero-order reactions (all-zero reactant row) have a_r = k_r.

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        reactant_matrix: (n_reactions, n_species) reactant stoichiometry.
        rate_constants: (n_reactions,) rate constant per reaction.
        initial_state: (n_species,) initial molecule counts.
        t_max: Maximum simulation time.
        max_reactions: Safety cap on total events.
        seed: RNG seed for this trajectory.

    Returns:
        (times, states) where times is (n_events+1,) and states is
        (n_events+1, n_species). Includes the initial state at t=0.
    """
    np.random.seed(seed)
    n_reactions = stoichiometry.shape[0]
    n_species = stoichiometry.shape[1]

    # Pre-allocate output arrays; +2 for initial state and final entry.
    max_steps = max_reactions + 2
    times = np.empty(max_steps, dtype=np.float64)
    states = np.empty((max_steps, n_species), dtype=np.float64)

    state = initial_state.copy()
    t = 0.0
    times[0] = t
    states[0] = state.copy()
    step = 1

    propensities = np.empty(n_reactions, dtype=np.float64)

    for _ in range(max_reactions):
        if t >= t_max:
            break

        # Compute macroscopic mass-action propensities
        total_propensity = 0.0
        for r in range(n_reactions):
            a = rate_constants[r]
            for s in range(n_species):
                power = reactant_matrix[r, s]
                if power > 0.0:
                    if state[s] <= 0.0:
                        a = 0.0
                        break
                    a *= state[s] ** power
            propensities[r] = max(a, 0.0)
            total_propensity += propensities[r]

        if total_propensity <= 0.0:
            break

        # Sample waiting time ~ Exp(total_propensity)
        u1 = np.random.random()
        if u1 <= 0.0:
            u1 = 1e-10
        dt = -np.log(u1) / total_propensity

        t_next = t + dt
        if t_next > t_max:
            break

        # Sample which reaction fires
        u2 = np.random.random() * total_propensity
        cumsum = 0.0
        reaction = n_reactions - 1  # default to last
        for r in range(n_reactions):
            cumsum += propensities[r]
            if u2 <= cumsum:
                reaction = r
                break

        # Apply reaction; clamp to zero (no negative counts)
        t = t_next
        for s in range(n_species):
            state[s] += stoichiometry[reaction, s]
            if state[s] < 0.0:
                state[s] = 0.0

        times[step] = t
        states[step] = state.copy()
        step += 1

    # Record final state at t_max if not already there
    if times[step - 1] < t_max:
        times[step] = t_max
        states[step] = state.copy()
        step += 1

    return times[:step].copy(), states[:step].copy()


@njit(cache=True)
def _gillespie_batch_inner(
    stoichiometry: np.ndarray,
    reactant_matrix: np.ndarray,
    rate_constants: np.ndarray,
    initial_state: np.ndarray,
    t_max: float,
    max_reactions: int,
    seeds: np.ndarray,
    n_trajectories: int,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Run M independent SSA trajectories and interpolate to a common grid.

    Combines simulation and zero-order-hold interpolation in a single compiled
    function to avoid Python-level round trips per trajectory.

    Args:
        stoichiometry: (n_reactions, n_species) net change matrix.
        reactant_matrix: (n_reactions, n_species) reactant stoichiometry.
        rate_constants: (n_reactions,) rate constants.
        initial_state: (n_species,) initial molecule counts.
        t_max: Simulation end time.
        max_reactions: Safety cap per trajectory.
        seeds: (n_trajectories,) int64 RNG seeds, one per trajectory.
        n_trajectories: Number of independent runs.
        time_grid: (T,) time points to interpolate onto.

    Returns:
        (n_trajectories, T, n_species) array of interpolated trajectories.
    """
    n_grid = time_grid.shape[0]
    n_species = stoichiometry.shape[1]
    result = np.empty((n_trajectories, n_grid, n_species), dtype=np.float64)

    for m in range(n_trajectories):
        times, states = _gillespie_mass_action_inner(
            stoichiometry,
            reactant_matrix,
            rate_constants,
            initial_state,
            t_max,
            max_reactions,
            int(seeds[m]),
        )

        # Zero-order hold interpolation onto the time grid
        n_events = times.shape[0]
        event_idx = 0
        for t_idx in range(n_grid):
            t_query = time_grid[t_idx]
            while event_idx < n_events - 1 and times[event_idx + 1] <= t_query:
                event_idx += 1
            result[m, t_idx, :] = states[event_idx, :]

    return result


class FastMassActionSSA:
    """Numba-accelerated SSA for mass-action CRNs.

    Works with CRNs where ALL propensities are either constant_rate or
    mass_action. For CRNs with Hill, Michaelis-Menten, or other non-mass-action
    kinetics, use the standard GillespieSSA.

    Requires numba to be installed. Raises ImportError if numba is absent.
    """

    def __init__(self) -> None:
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "FastMassActionSSA requires numba. Install with: "
                "pip install 'crn-surrogate[fast]'  or  pip install numba"
            )

    def simulate_batch(
        self,
        stoichiometry: np.ndarray,
        reactant_matrix: np.ndarray,
        rate_constants: np.ndarray,
        initial_state: np.ndarray,
        t_max: float,
        time_grid: np.ndarray,
        n_trajectories: int,
        max_reactions: int = 100_000,
    ) -> torch.Tensor:
        """Run M SSA trajectories and return interpolated tensor.

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            reactant_matrix: (n_reactions, n_species) reactant stoichiometry.
            rate_constants: (n_reactions,) rate constants.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time.
            time_grid: (T,) time points to interpolate onto.
            n_trajectories: Number of independent runs.
            max_reactions: Safety cap per trajectory.

        Returns:
            (n_trajectories, T, n_species) torch.Tensor of float32.
        """
        import torch

        stoich_np = self._to_numpy(stoichiometry)
        reactant_np = self._to_numpy(reactant_matrix)
        rates_np = self._to_numpy(rate_constants)
        init_np = self._to_numpy(initial_state)
        grid_np = self._to_numpy(time_grid)

        # Generate seeds from torch RNG so results are reproducible under torch.manual_seed
        seeds = np.array(
            [torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(n_trajectories)],
            dtype=np.int64,
        )

        result_np = _gillespie_batch_inner(
            stoich_np,
            reactant_np,
            rates_np,
            init_np,
            t_max,
            max_reactions,
            seeds,
            n_trajectories,
            grid_np,
        )

        return torch.from_numpy(result_np).float()

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert a torch.Tensor or np.ndarray to a float64 C-contiguous array."""
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    @classmethod
    def extract_topology_arrays(cls, crn: CRN) -> dict[str, np.ndarray]:
        """Extract numpy arrays required by simulate_batch from a CRN.

        Only works when every propensity is constant_rate or mass_action.
        Raises ValueError if any reaction uses a different kinetic law.

        Args:
            crn: CRN whose propensities are all constant-rate or mass-action.

        Returns:
            Dict with keys 'stoichiometry', 'reactant_matrix', 'rate_constants'.

        Raises:
            ValueError: If any propensity type is not supported.
        """
        from crn_surrogate.crn.propensities import (
            ConstantRateParams,
            MassActionParams,
            SerializablePropensity,
        )

        n_reactions = crn.n_reactions
        n_species = crn.n_species
        stoich = crn.stoichiometry_matrix.numpy().astype(np.float64)
        reactant = np.zeros((n_reactions, n_species), dtype=np.float64)
        rates = np.zeros(n_reactions, dtype=np.float64)

        for r, rxn in enumerate(crn.reactions):
            if not isinstance(rxn.propensity, SerializablePropensity):
                raise ValueError(
                    f"Reaction {r} ({rxn}) propensity does not implement "
                    f"SerializablePropensity (missing .params). "
                    f"Only constant_rate and mass_action propensities are supported."
                )
            prop: SerializablePropensity = rxn.propensity
            if isinstance(prop.params, ConstantRateParams):
                rates[r] = prop.params.rate
                # reactant row stays all-zero (zero-order reaction)
            elif isinstance(prop.params, MassActionParams):
                rates[r] = prop.params.rate_constant
                # cast to Any: isinstance(prop.params, MassActionParams) guarantees
                # prop is a _MassActionClosure which exposes .reactant_stoichiometry,
                # but SerializablePropensity does not declare that attribute.
                reactant[r] = (
                    cast(Any, prop).reactant_stoichiometry.numpy().astype(np.float64)
                )
            else:
                raise ValueError(
                    f"Reaction {r} ({rxn}) has propensity type "
                    f"{type(prop).__name__}, which is not supported by "
                    f"FastMassActionSSA. Only constant_rate and mass_action "
                    f"propensities are supported."
                )

        return {
            "stoichiometry": stoich,
            "reactant_matrix": reactant,
            "rate_constants": rates,
        }
