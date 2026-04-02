"""Gillespie direct method exact stochastic simulation algorithm (SSA).

The simulator is CRN-agnostic: it accepts a stoichiometry matrix and a
propensity callable, with no dependency on the CRN domain class.
"""

from __future__ import annotations

import bisect
import warnings
from typing import TYPE_CHECKING, Callable

import torch

from crn_surrogate.simulation.trajectory import Trajectory

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import InputProtocol


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

    Usage with an input protocol::

        from crn_surrogate.crn.inputs import single_pulse, InputProtocol
        protocol = InputProtocol(schedules={2: single_pulse(5.0, 15.0, amplitude=50.0)})
        traj = ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=x0,
            t_max=20.0,
            input_protocol=protocol,
            external_species=crn.external_species,
        )
    """

    def simulate(
        self,
        stoichiometry: torch.Tensor,
        propensity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        initial_state: torch.Tensor,
        t_max: float,
        max_reactions: int = 100_000,
        input_protocol: InputProtocol | None = None,
        external_species: frozenset[int] = frozenset(),
    ) -> Trajectory:
        """Run exact SSA and return a trajectory at reaction event times.

        When an input_protocol is provided, the simulation is breakpoint-aware:
        it runs standard Gillespie within each piecewise-constant interval and
        updates external species values at each breakpoint.

        Issues a warning if the propensity function appears to be time-varying
        and no input_protocol is provided (detected by comparing
        propensity_fn(state, 0.0) vs propensity_fn(state, 1.0)). The standard
        direct method may not be exact for time-varying rates.

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            propensity_fn: Callable (state, t) → (n_reactions,) propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Maximum simulation time.
            max_reactions: Safety cap on the total number of reaction events.
            input_protocol: Optional pulsatile input schedule. If provided,
                external species are updated at each protocol breakpoint.
            external_species: Set of species indices that are externally
                controlled. Their stoichiometric columns must be zero (enforced
                by CRN validation). Provided here for a runtime safety check.

        Returns:
            Trajectory with event times and states.

        Raises:
            ValueError: If a reaction's stoichiometry changes an external species.
        """
        # Lazy import to avoid circular dependencies at module load time
        from crn_surrogate.crn.inputs import EMPTY_PROTOCOL

        protocol = input_protocol if input_protocol is not None else EMPTY_PROTOCOL

        if not protocol.schedules:
            self._warn_if_time_varying(propensity_fn, initial_state)

        self._validate_external_stoichiometry(stoichiometry, external_species)

        state = initial_state.float().clone()

        # Set initial external species values
        for idx, value in protocol.evaluate(0.0).items():
            state[idx] = value

        # Build sorted list of breakpoints to partition the simulation
        breakpoints_list = sorted(set(protocol.breakpoints()) | {0.0, t_max})

        times = [0.0]
        states = [state.clone()]
        total_reactions = 0

        for bp_idx in range(len(breakpoints_list) - 1):
            t = breakpoints_list[bp_idx]
            next_bp = breakpoints_list[bp_idx + 1]

            # Standard Gillespie within [t, next_bp)
            while t < next_bp and total_reactions < max_reactions:
                a = (
                    propensity_fn(state, t)
                    .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    .clamp(min=0.0)
                )
                a_total = a.sum()
                if a_total <= 0:
                    break  # no reactions possible; jump to breakpoint

                dt = self._sample_wait_time(a_total)
                t_next = t + dt.item()
                if t_next >= next_bp:
                    break  # next event would cross breakpoint; stop here

                t = t_next
                r = self._select_reaction(a, a_total)
                state = (state + stoichiometry[r].float()).clamp(min=0.0)
                # Restore external species (stoichiometry should be zero,
                # but guard against floating-point drift)
                for ext_idx, value in protocol.evaluate(t).items():
                    state[ext_idx] = value

                times.append(t)
                states.append(state.clone())
                total_reactions += 1

            # At breakpoint: update external species
            t = next_bp
            for ext_idx, value in protocol.evaluate(t).items():
                state[ext_idx] = value

            if total_reactions >= max_reactions:
                break

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
            if not a0.isfinite().all() or not a1.isfinite().all():
                return
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

    def _validate_external_stoichiometry(
        self,
        stoichiometry: torch.Tensor,
        external_species: frozenset[int],
    ) -> None:
        """Raise ValueError if any external species has nonzero stoichiometric change.

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            external_species: Set of externally controlled species indices.

        Raises:
            ValueError: If any external species appears as a reactant or product.
        """
        for idx in external_species:
            col = stoichiometry[:, idx]
            if col.abs().max().item() > 0:
                raise ValueError(
                    f"External species {idx} has nonzero net stoichiometric change "
                    f"in the stoichiometry matrix. External species must not appear "
                    f"as reactants or products — only as propensity dependencies."
                )
