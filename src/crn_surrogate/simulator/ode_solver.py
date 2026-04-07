"""Euler ODE solver for deterministic integration of the neural drift network.

Uses the same interface as EulerMaruyamaSolver but suppresses the diffusion
term entirely — no ``sde.diffusion()`` call, no noise sampling. Intended for
deterministic proof-of-concept runs where data was generated via ODE integration
and stochasticity is not desired at inference time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.state_transform import StateTransform

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import ResolvedProtocol


class EulerODESolver:
    """Pure Euler (ODE) integrator for the neural drift network.

    X(t+dt) = X(t) + f(X, t) * dt

    The diffusion network is never called. Two calls with identical inputs
    always produce identical trajectories.
    """

    def __init__(
        self, config: SDEConfig, state_transform: StateTransform | None = None
    ) -> None:
        """Args:
        config: SDE configuration (clip_state flag lives here).
        state_transform: Optional transform applied to state before/after integration.
        """
        self._config = config
        self._state_transform = state_transform

    def solve(
        self,
        sde: CRNNeuralSDE,
        initial_state: torch.Tensor,
        crn_context: CRNContext,
        t_span: torch.Tensor,
        dt: float,
        resolved_protocol: ResolvedProtocol | None = None,
    ) -> Trajectory:
        """Integrate the neural drift forward in time (no noise).

        Args:
            sde: The neural SDE module (only its drift network is used).
            initial_state: (n_species,) initial state.
            crn_context: CRN encoder output for conditioning.
            t_span: (T,) time points at which to record the state.
            dt: Integration step size.
            resolved_protocol: Optional bundle of the InputProtocol, its
                pre-computed protocol embedding, and the external species mask.
                When provided, external species are clamped at each step and
                the embedding conditions the drift via FiLM.

        Returns:
            Trajectory with states recorded at t_span time points.
        """
        input_protocol = (
            resolved_protocol.protocol if resolved_protocol is not None else None
        )

        t_start = t_span[0].item()
        t_end = t_span[-1].item()
        n_steps = max(1, int((t_end - t_start) / dt))
        time_grid = torch.linspace(
            t_start, t_end, n_steps + 1, device=initial_state.device
        )

        state = initial_state.clone().float()
        if self._state_transform is not None:
            state = self._state_transform.forward(state)

        if input_protocol is not None:
            for idx, value in input_protocol.evaluate(t_start).items():
                state[idx] = value

        recorded_states = []
        span_idx = 0

        for i in range(n_steps):
            t = time_grid[i]
            if span_idx < len(t_span) and t >= t_span[span_idx]:
                recorded_states.append(state.clone())
                span_idx += 1

            state = self._step(
                sde,
                state,
                t,
                dt,
                crn_context,
                resolved_protocol=resolved_protocol,
            )

        while span_idx < len(t_span):
            recorded_states.append(state.clone())
            span_idx += 1

        states = torch.stack(recorded_states, dim=0)
        if self._state_transform is not None:
            states = self._state_transform.inverse_trajectory(states)
        return Trajectory(times=t_span, states=states)

    def _step(
        self,
        sde: CRNNeuralSDE,
        state: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        crn_context: CRNContext,
        resolved_protocol: ResolvedProtocol | None = None,
    ) -> torch.Tensor:
        """Single Euler step (drift only) with optional external input clamping."""
        protocol_embedding = (
            resolved_protocol.embedding if resolved_protocol is not None else None
        )
        input_protocol = (
            resolved_protocol.protocol if resolved_protocol is not None else None
        )
        external_species_mask = (
            resolved_protocol.external_species_mask
            if resolved_protocol is not None
            else None
        )

        # 1. Set clamped species BEFORE computing drift.
        if input_protocol is not None:
            for idx, value in input_protocol.evaluate(t.item()).items():
                state[idx] = value

        # 2. Compute drift only — diffusion is not called.
        f = sde.drift(t, state, crn_context, protocol_embedding)

        # 3. Euler step (no noise term).
        new_state = state + f * dt

        # 4. Clip internal species only.
        if self._config.clip_state:
            if external_species_mask is not None:
                internal_mask = ~external_species_mask
                new_state[internal_mask] = new_state[internal_mask].clamp(min=0.0)
            else:
                new_state = new_state.clamp(min=0.0)

        # 5. Overwrite external species with protocol values at t + dt.
        if input_protocol is not None:
            t_next = t.item() + dt
            for idx, value in input_protocol.evaluate(t_next).items():
                new_state[idx] = value

        return new_state
