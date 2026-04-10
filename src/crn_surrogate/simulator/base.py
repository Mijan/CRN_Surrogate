"""Abstract base classes for surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from crn_surrogate.encoder.bipartite_gnn import CRNContext


class SurrogateModel(ABC, torch.nn.Module):
    """Base class for all CRN surrogate models.

    Every surrogate model provides a drift function conditioned on a CRN
    context. Stochastic models additionally provide a diffusion function.
    """

    @property
    @abstractmethod
    def n_species(self) -> int:
        """State dimension."""
        ...

    @abstractmethod
    def drift(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drift f(X, t; ctx). Returns same shape as state."""
        ...

    @abstractmethod
    def drift_from_context(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drift from a pre-extracted context vector (for batched NLL)."""
        ...

    def predict_transition(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        dt: float,
        protocol_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (mu, process_variance) for one Euler step.

        Deterministic base implementation: variance is always zero.
        StochasticSurrogate overrides this to include diffusion variance.

        Args:
            t: (N,) or scalar time values.
            x: (N, n_species) current states.
            context_vector: (N, d_context) context vectors.
            dt: Time step size.
            protocol_embedding: Optional (N, d_protocol) embeddings.

        Returns:
            mu: (N, n_species) predicted next state.
            process_variance: (N, n_species) process variance (zero for
                deterministic models).
        """
        drift = self.drift_from_context(t, x, context_vector, protocol_embedding)
        mu = x + drift * dt
        return mu, torch.zeros_like(mu)


class StochasticSurrogate(SurrogateModel, ABC):
    """Surrogate model with both drift and diffusion."""

    @abstractmethod
    def diffusion(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion g(X, t; ctx).

        Returns:
            (n_species, n_noise_channels) or (B, n_species, n_noise_channels).
        """
        ...

    @abstractmethod
    def diffusion_from_context(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion from a pre-extracted context vector."""
        ...

    def predict_transition(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        dt: float,
        protocol_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (mu, process_variance) for one Euler step including diffusion.

        Args:
            t: (N,) or scalar time values.
            x: (N, n_species) current states.
            context_vector: (N, d_context) context vectors.
            dt: Time step size.
            protocol_embedding: Optional (N, d_protocol) embeddings.

        Returns:
            mu: (N, n_species) predicted next state.
            process_variance: (N, n_species) per-species variance from the
                diffusion matrix G, i.e. var_s = sum_j(G_{s,j}^2) * dt.
        """
        drift = self.drift_from_context(t, x, context_vector, protocol_embedding)
        mu = x + drift * dt
        G = self.diffusion_from_context(t, x, context_vector, protocol_embedding)
        variance = (G**2).sum(dim=-1) * dt
        return mu, variance
