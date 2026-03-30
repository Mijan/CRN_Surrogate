"""Gillespie direct method exact stochastic simulation algorithm (SSA).

The simulator is CRN-agnostic: it accepts a stoichiometry matrix and a
propensity callable, with no dependency on the CRN domain class.
"""
from __future__ import annotations

import warnings
from typing import Callable

import torch

from crn_surrogate.simulation.trajectory import Trajectory


class GillespieSSA:
    """Exact stochastic simulation algorithm (Gillespie 1977, direct method).

    The simulator is CRN-agnostic. It accepts a stoichiometry matrix and a
    propensity function and has no dependency on the CRN class.

    Usage with a CRN object::

        ssa = GillespieSSA()
        traj = ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=x0,
            t_max=100.0,
        )

    Usage with a hand-coded system::

        traj = ssa.simulate(
            stoichiometry=torch.tensor([[1], [-1]]),
            propensity_fn=lambda state, t: torch.tensor([1.0, 0.1 * state[0]]),
            initial_state=torch.tensor([10.0]),
            t_max=100.0,
        )
    """

    def simulate(
        self,
        stoichiometry: torch.Tensor,
        propensity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        initial_state: torch.Tensor,
        t_max: float,
        max_reactions: int = 100_000,
    ) -> Trajectory:
        """Run exact SSA and return a trajectory at reaction event times.

        Issues a warning if the propensity function appears to be time-varying
        (detected by comparing propensity_fn(state, 0.0) vs propensity_fn(state, 1.0)).
        The standard direct method may not be exact for time-varying rates.

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            propensity_fn: Callable (state, t) → (n_reactions,) propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Maximum simulation time.
            max_reactions: Safety cap on the number of reaction events.

        Returns:
            Trajectory with event times and states.
        """
        self._warn_if_time_varying(propensity_fn, initial_state)

        state = initial_state.float().clone()
        t = 0.0

        times = [t]
        states = [state.clone()]

        for _ in range(max_reactions):
            a = propensity_fn(state, t).clamp(min=0.0)
            a_total = a.sum()
            if a_total <= 0:
                break

            dt = self._sample_wait_time(a_total)
            t = t + dt.item()
            if t >= t_max:
                break

            r = self._select_reaction(a, a_total)
            state = (state + stoichiometry[r].float()).clamp(min=0.0)

            times.append(t)
            states.append(state.clone())

        times_tensor = torch.tensor(times)
        states_tensor = torch.stack(states, dim=0)
        return Trajectory(times=times_tensor, states=states_tensor)

    def _sample_wait_time(self, a_total: torch.Tensor) -> torch.Tensor:
        """Sample exponential waiting time ~ Exp(a_total)."""
        u = torch.rand(1).clamp(min=1e-10)
        return -torch.log(u) / a_total

    def _select_reaction(self, a: torch.Tensor, a_total: torch.Tensor) -> int:
        """Sample which reaction fires, proportional to propensities."""
        probs = a / a_total
        return int(torch.multinomial(probs, num_samples=1).item())

    def _warn_if_time_varying(
        self,
        propensity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
    ) -> None:
        """Issue a warning if the propensity function appears to vary with time."""
        try:
            a0 = propensity_fn(state, 0.0)
            a1 = propensity_fn(state, 1.0)
            if not torch.allclose(a0, a1, atol=1e-6):
                warnings.warn(
                    "The propensity function appears to be time-varying "
                    "(propensity_fn(state, 0.0) != propensity_fn(state, 1.0)). "
                    "The Gillespie direct method assumes propensities change only at "
                    "reaction events and may not be exact for time-varying rates. "
                    "Consider using a thinning-based method for correctness.",
                    UserWarning,
                    stacklevel=3,
                )
        except Exception:
            pass
