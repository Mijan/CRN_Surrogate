"""Abstract base classes for surrogate models and simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulation.trajectory import Trajectory


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


class Simulator(ABC):
    """Abstract simulator that integrates a surrogate model forward in time."""

    @abstractmethod
    def solve(
        self,
        model: SurrogateModel,
        initial_state: torch.Tensor,
        crn_context: CRNContext,
        t_span: torch.Tensor,
        dt: float,
        resolved_protocol=None,
    ) -> Trajectory:
        """Integrate the model forward, recording states at t_span."""
        ...
