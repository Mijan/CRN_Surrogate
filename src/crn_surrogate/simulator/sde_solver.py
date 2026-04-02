from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import ResolvedProtocol


class EulerMaruyamaSolver:
    """Euler-Maruyama integrator for the neural SDE.

    X(t+dt) = X(t) + f(X, t) * dt + g(X, t) * sqrt(dt) * Z,  Z ~ N(0, I)
    """

    def __init__(self, config: SDEConfig) -> None:
        """Args:
        config: SDE configuration (clip_state flag lives here).
        """
        self._config = config

    def solve(
        self,
        sde: CRNNeuralSDE,
        initial_state: torch.Tensor,
        crn_context: CRNContext,
        t_span: torch.Tensor,
        dt: float,
        resolved_protocol: ResolvedProtocol | None = None,
    ) -> Trajectory:
        """Integrate the neural SDE forward in time.

        Args:
            sde: The neural SDE module.
            initial_state: (n_species,) initial state.
            crn_context: CRN encoder output for conditioning.
            t_span: (T,) time points at which to record the state.
            dt: Integration step size.
            resolved_protocol: Optional bundle of the InputProtocol, its
                pre-computed protocol embedding, and the external species mask.
                When provided, external species are clamped at each step and
                the embedding conditions the SDE drift/diffusion via FiLM.

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

        # Set initial external species values from the protocol.
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

        # Capture any remaining t_span points.
        while span_idx < len(t_span):
            recorded_states.append(state.clone())
            span_idx += 1

        states = torch.stack(recorded_states, dim=0)
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
        """Single Euler-Maruyama step with optional external input clamping.

        External species are overwritten from the protocol before computing drift
        and diffusion, ensuring the SDE always sees the correct input values.
        """
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

        # 1. Set clamped species BEFORE computing drift/diffusion.
        if input_protocol is not None:
            for idx, value in input_protocol.evaluate(t.item()).items():
                state[idx] = value

        # 2. Compute drift and diffusion on full state (including clamped species).
        f = sde.drift(t, state, crn_context, protocol_embedding)
        g = sde.diffusion(t, state, crn_context, protocol_embedding)

        # 3. Euler-Maruyama step.
        z = torch.randn(g.shape[-1], device=state.device)
        noise = (g * z.unsqueeze(0)).sum(dim=-1)
        new_state = state + f * dt + noise * (dt**0.5)

        # 4. Clip internal species only (external species are overwritten next step).
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
