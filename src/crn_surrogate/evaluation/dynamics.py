"""Learned drift and diffusion visualization against analytical CLE reference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr
from crn_surrogate.simulator.base import StochasticSurrogate


@dataclass(frozen=True)
class DynamicsProfile:
    """Learned drift and diffusion evaluated over a state range.

    Attributes:
        state_values: (N,) swept state values for the target species.
        learned_drift: (N, n_species) drift F_θ(x) evaluated at each state.
        learned_diffusion: (N, n_species) effective noise amplitude ||G_θ[s,:]||₂.
        analytical_drift: (N,) analytical CLE drift for the target species, or None.
        analytical_diffusion: (N,) analytical CLE diffusion for the target species, or None.
    """

    state_values: torch.Tensor
    learned_drift: torch.Tensor
    learned_diffusion: torch.Tensor
    analytical_drift: torch.Tensor | None = None
    analytical_diffusion: torch.Tensor | None = None


class DynamicsVisualizer:
    """Compare learned SDE dynamics against analytical CLE.

    Encodes the CRN once at construction time and caches the context.
    All subsequent calls to evaluate/plot use the cached context.
    """

    def __init__(
        self,
        encoder: BipartiteGNNEncoder,
        sde: StochasticSurrogate,
        crn_repr: CRNTensorRepr,
        initial_state: torch.Tensor,
    ) -> None:
        """Encode the CRN once and cache the context.

        Args:
            encoder: Trained bipartite GNN encoder.
            sde: Trained neural SDE.
            crn_repr: Tensor representation of the CRN.
            initial_state: (n_species,) baseline state for evaluating
                drift/diffusion over a state range.
        """
        device = next(encoder.parameters()).device
        crn_repr = crn_repr.to(device)
        initial_state = initial_state.to(device)

        encoder.eval()
        sde.eval()
        with torch.no_grad():
            self._ctx: CRNContext = encoder(crn_repr)
        self._sde = sde
        self._device = device
        self._initial_state = initial_state

    def evaluate_over_state_range(
        self,
        state_range: torch.Tensor,
        *,
        species_index: int = 0,
        t: float = 0.0,
        analytical_drift_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        analytical_diffusion_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> DynamicsProfile:
        """Evaluate drift and diffusion across a range of states.

        Fixes all species to initial_state and varies species_index over
        state_range. Runs a single batched forward pass.

        Args:
            state_range: (N,) values to sweep for the target species.
            species_index: Which species dimension to vary.
            t: Time at which to evaluate (for future time-varying dynamics).
            analytical_drift_fn: Optional callable ``(x: Tensor(N,)) -> Tensor(N,)``
                returning the analytical CLE drift for the target species.
            analytical_diffusion_fn: Optional callable ``(x: Tensor(N,)) -> Tensor(N,)``
                returning the analytical CLE diffusion magnitude for the target species.

        Returns:
            DynamicsProfile with learned and optional analytical values.
        """
        state_range = state_range.to(self._device)
        N = len(state_range)
        states = (
            self._initial_state.unsqueeze(0).expand(N, -1).clone()
        )  # (N, n_species)
        states[:, species_index] = state_range
        t_batch = torch.full((N,), t, device=self._device)

        with torch.no_grad():
            learned_drift = self._sde.drift(
                t_batch, states, self._ctx
            )  # (N, n_species)
            G = self._sde.diffusion(
                t_batch, states, self._ctx
            )  # (N, n_species, n_noise)
            learned_diffusion = G.pow(2).sum(dim=-1).sqrt()  # (N, n_species)

        a_drift = (
            analytical_drift_fn(state_range)
            if analytical_drift_fn is not None
            else None
        )
        a_diff = (
            analytical_diffusion_fn(state_range)
            if analytical_diffusion_fn is not None
            else None
        )

        return DynamicsProfile(
            state_values=state_range,
            learned_drift=learned_drift,
            learned_diffusion=learned_diffusion,
            analytical_drift=a_drift,
            analytical_diffusion=a_diff,
        )

    def plot_drift(
        self,
        state_range: torch.Tensor,
        *,
        analytical_drift_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        species_index: int = 0,
        label: str = "Learned",
        color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot learned drift vs state, optionally with analytical CLE overlay.

        Args:
            state_range: (N,) state values for the x-axis.
            analytical_drift_fn: Optional callable ``(x: Tensor(N,)) -> Tensor(N,)``.
                For birth-death: ``lambda x: k_birth - k_death * x``.
            species_index: Which species dimension to plot.
            label: Legend label for the learned curve.
            color: Color for the learned curve.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        profile = self.evaluate_over_state_range(
            state_range,
            species_index=species_index,
            analytical_drift_fn=analytical_drift_fn,
        )
        x = profile.state_values.numpy()
        ax.plot(
            x,
            profile.learned_drift[:, species_index].numpy(),
            color=color,
            lw=2,
            label=label,
        )

        if profile.analytical_drift is not None:
            ax.plot(
                x,
                profile.analytical_drift.numpy(),
                "k--",
                lw=1.5,
                label="Analytical (CLE)",
            )

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(f"Species {species_index} count")
        ax.set_ylabel("Drift  F\u03b8(x)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_diffusion(
        self,
        state_range: torch.Tensor,
        *,
        analytical_diffusion_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        species_index: int = 0,
        label: str = "Learned",
        color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot learned diffusion magnitude ||G_θ[s,:]||₂ vs state.

        Args:
            state_range: (N,) state values for the x-axis.
            analytical_diffusion_fn: Optional callable ``(x: Tensor(N,)) -> Tensor(N,)``.
                For birth-death: ``lambda x: torch.sqrt(k_birth + k_death * x)``.
            species_index: Which species dimension to plot.
            label: Legend label for the learned curve.
            color: Color for the learned curve.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        profile = self.evaluate_over_state_range(
            state_range,
            species_index=species_index,
            analytical_diffusion_fn=analytical_diffusion_fn,
        )
        x = profile.state_values.numpy()
        ax.plot(
            x,
            profile.learned_diffusion[:, species_index].numpy(),
            color=color,
            lw=2,
            label=label,
        )

        if profile.analytical_diffusion is not None:
            ax.plot(
                x,
                profile.analytical_diffusion.numpy(),
                "k--",
                lw=1.5,
                label="Analytical (CLE)",
            )

        ax.set_xlabel(f"Species {species_index} count")
        ax.set_ylabel("\u2016G\u03b8[s,:]\u2016\u2082  (diffusion amplitude)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax
