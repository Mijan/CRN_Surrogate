from __future__ import annotations
import torch
from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.trajectory import Trajectory


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
    ) -> Trajectory:
        """Integrate the neural SDE forward in time.

        Args:
            sde: The neural SDE module.
            initial_state: (n_species,) initial state.
            crn_context: CRN encoder output for conditioning.
            t_span: (T,) time points at which to record the state.
            dt: Integration step size.

        Returns:
            Trajectory with states recorded at t_span time points.
        """
        t_start = t_span[0].item()
        t_end = t_span[-1].item()
        n_steps = max(1, int((t_end - t_start) / dt))
        time_grid = torch.linspace(t_start, t_end, n_steps + 1, device=initial_state.device)

        state = initial_state.clone().float()
        recorded_states = []
        span_idx = 0

        for i in range(n_steps):
            t = time_grid[i]
            if span_idx < len(t_span) and t >= t_span[span_idx]:
                recorded_states.append(state.clone())
                span_idx += 1

            state = self._step(sde, state, t, dt, crn_context)

        # Capture any remaining t_span points
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
    ) -> torch.Tensor:
        """Single Euler-Maruyama step."""
        f = sde.drift(t, state, crn_context)
        g = sde.diffusion(t, state, crn_context)

        z = torch.randn(g.shape[-1], device=state.device)
        noise = (g * z.unsqueeze(0)).sum(dim=-1)

        new_state = state + f * dt + noise * (dt ** 0.5)

        if self._config.clip_state:
            new_state = new_state.clamp(min=0.0)

        return new_state
