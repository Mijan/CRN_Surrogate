"""Gillespie direct method exact stochastic simulation algorithm (SSA).

The simulator is CRN-agnostic: it accepts a stoichiometry matrix and a
propensity callable, with no dependency on the CRN domain class.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import torch

from crn_surrogate.simulation.trajectory import Trajectory

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import InputProtocol


def _ssa_worker(args: dict) -> Trajectory:
    """Module-level worker for parallel SSA. Required for pickling.

    Each worker receives its own RNG seed to ensure independent trajectories.
    """
    torch.manual_seed(args["seed"])
    ssa = GillespieSSA()
    kwargs: dict = {}
    if args.get("input_protocol") is not None:
        kwargs["input_protocol"] = args["input_protocol"]
    if args.get("external_species") is not None:
        kwargs["external_species"] = args["external_species"]
    return ssa.simulate(
        stoichiometry=args["stoichiometry"],
        propensity_fn=args["propensity_fn"],
        initial_state=args["initial_state"].clone(),
        t_max=args["t_max"],
        **kwargs,
    )


class GillespieSSA:
    """Exact stochastic simulation algorithm (Gillespie 1977, direct method).

    The simulator is CRN-agnostic. It accepts a stoichiometry matrix and a
    propensity function and has no dependency on the CRN class.

    Single trajectory::

        ssa = GillespieSSA()
        traj = ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=x0,
            t_max=100.0,
        )

    Batch of independent trajectories::

        trajectories = ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=x0,
            t_max=100.0,
            n_trajectories=32,
            n_workers=4,
        )
        # Interpolate and stack for training:
        tensor = Trajectory.stack_on_grid(trajectories, time_grid)
        # tensor.shape == (32, len(time_grid), n_species)

    With an input protocol::

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

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            propensity_fn: Callable (state, t) -> (n_reactions,) propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Maximum simulation time.
            max_reactions: Safety cap on the total number of reaction events.
            input_protocol: Optional pulsatile input schedule.
            external_species: Set of externally controlled species indices.

        Returns:
            Trajectory with event times and states.

        Raises:
            ValueError: If a reaction's stoichiometry changes an external species.
        """
        from crn_surrogate.crn.inputs import EMPTY_PROTOCOL

        protocol = input_protocol if input_protocol is not None else EMPTY_PROTOCOL

        if not protocol.schedules:
            self._warn_if_time_varying(propensity_fn, initial_state)

        self._validate_external_stoichiometry(stoichiometry, external_species)

        state = initial_state.float().clone()

        for idx, value in protocol.evaluate(0.0).items():
            state[idx] = value

        breakpoints_list = sorted(set(protocol.breakpoints()) | {0.0, t_max})
        breakpoints_list = [bp for bp in breakpoints_list if bp <= t_max]

        times = [0.0]
        states = [state.clone()]
        total_reactions = 0

        for bp_idx in range(len(breakpoints_list) - 1):
            t = breakpoints_list[bp_idx]
            next_bp = breakpoints_list[bp_idx + 1]

            while t < next_bp and total_reactions < max_reactions:
                a = (
                    propensity_fn(state, t)
                    .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    .clamp(min=0.0)
                )
                a_total = a.sum()
                if a_total <= 0:
                    break

                dt = self._sample_wait_time(a_total)
                t_next = t + dt.item()
                if t_next >= next_bp:
                    break

                t = t_next
                r = self._select_reaction(a, a_total)
                state = (state + stoichiometry[r].float()).clamp(min=0.0)
                for ext_idx, value in protocol.evaluate(t).items():
                    state[ext_idx] = value

                times.append(t)
                states.append(state.clone())
                total_reactions += 1

            t = next_bp
            for ext_idx, value in protocol.evaluate(t).items():
                state[ext_idx] = value

            if t < t_max:
                times.append(t)
                states.append(state.clone())

            if total_reactions >= max_reactions:
                break

        if times[-1] < t_max:
            times.append(t_max)
            states.append(state.clone())

        return Trajectory(
            times=torch.tensor(times),
            states=torch.stack(states, dim=0),
        )

    def simulate_batch(
        self,
        stoichiometry: torch.Tensor,
        propensity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        initial_state: torch.Tensor,
        t_max: float,
        n_trajectories: int,
        input_protocol: InputProtocol | None = None,
        external_species: frozenset[int] = frozenset(),
        n_workers: int = 1,
    ) -> list[Trajectory]:
        """Run M independent SSA simulations.

        Each trajectory uses independent RNG state. When ``n_workers > 1``,
        simulations are distributed across processes. If pickling fails
        (common on macOS with ``spawn``), falls back to sequential execution
        with a warning.

        Args:
            stoichiometry: (n_reactions, n_species) net change matrix.
            propensity_fn: Callable (state, t) -> (n_reactions,) propensities.
            initial_state: (n_species,) initial molecule counts.
            t_max: Simulation end time.
            n_trajectories: Number of independent SSA runs (M).
            input_protocol: Optional input protocol for external species.
            external_species: Set of externally controlled species indices.
            n_workers: Number of parallel worker processes. Default 1.

        Returns:
            List of M Trajectory objects at event times.
        """
        seeds = [
            torch.randint(0, 2**62, (1,)).item() for _ in range(n_trajectories)
        ]

        if n_workers <= 1:
            return self._batch_sequential(
                stoichiometry, propensity_fn, initial_state, t_max,
                seeds, input_protocol, external_species,
            )
        return self._batch_parallel(
            stoichiometry, propensity_fn, initial_state, t_max,
            seeds, input_protocol, external_species, n_workers,
        )

    def _batch_sequential(
        self,
        stoichiometry: torch.Tensor,
        propensity_fn: Callable,
        initial_state: torch.Tensor,
        t_max: float,
        seeds: list[int],
        input_protocol: InputProtocol | None,
        external_species: frozenset[int],
    ) -> list[Trajectory]:
        kwargs: dict = {}
        if input_protocol is not None:
            kwargs["input_protocol"] = input_protocol
        if external_species:
            kwargs["external_species"] = external_species
        results = []
        for seed in seeds:
            torch.manual_seed(seed)
            traj = self.simulate(
                stoichiometry=stoichiometry,
                propensity_fn=propensity_fn,
                initial_state=initial_state.clone(),
                t_max=t_max,
                **kwargs,
            )
            results.append(traj)
        return results

    def _batch_parallel(
        self,
        stoichiometry: torch.Tensor,
        propensity_fn: Callable,
        initial_state: torch.Tensor,
        t_max: float,
        seeds: list[int],
        input_protocol: InputProtocol | None,
        external_species: frozenset[int],
        n_workers: int,
    ) -> list[Trajectory]:
        args_list = [
            {
                "stoichiometry": stoichiometry,
                "propensity_fn": propensity_fn,
                "initial_state": initial_state,
                "t_max": t_max,
                "seed": seed,
                "input_protocol": input_protocol,
                "external_species": external_species if external_species else None,
            }
            for seed in seeds
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                return list(pool.map(_ssa_worker, args_list))
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            warnings.warn(
                f"Parallel SSA failed ({exc!r}), falling back to sequential. "
                "On macOS, try: multiprocessing.set_start_method('fork').",
                RuntimeWarning,
                stacklevel=2,
            )
            return [_ssa_worker(a) for a in args_list]

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
        """Raise ValueError if any external species has nonzero stoichiometric change."""
        for idx in external_species:
            col = stoichiometry[:, idx]
            if col.abs().max().item() > 0:
                raise ValueError(
                    f"External species {idx} has nonzero net stoichiometric change "
                    f"in the stoichiometry matrix. External species must not appear "
                    f"as reactants or products — only as propensity dependencies."
                )
