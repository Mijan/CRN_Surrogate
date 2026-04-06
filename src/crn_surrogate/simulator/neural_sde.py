"""Neural SDE whose drift and diffusion are conditioned on CRN embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulator.conditioned_mlp import ConditionedMLP


class CRNNeuralSDE(nn.Module):
    """Neural SDE whose drift and diffusion are conditioned on CRN embeddings.

    Mirrors the Chemical Langevin Equation structure:
      dX = f(X, t; ctx) dt + g(X, t; ctx) dW
    where f and g are ConditionedMLPs that apply FiLM modulation at every
    hidden layer using the CRN encoder output as context.
    """

    def __init__(self, config: SDEConfig, n_species: int) -> None:
        """Args:
        config: SDE configuration.
        n_species: Number of species (state dimension).
        """
        super().__init__()
        self._config = config
        self._n_species = n_species
        # context = CRN pool (2 * d_model) + optional protocol embedding (d_protocol)
        d_context = 2 * config.d_model + config.d_protocol

        self._drift_net = ConditionedMLP(
            d_in=n_species,
            d_hidden=config.d_hidden,
            d_out=n_species,
            d_context=d_context,
            n_hidden_layers=config.n_hidden_layers,
            dropout=config.mlp_dropout,
        )
        self._diff_net = ConditionedMLP(
            d_in=n_species,
            d_hidden=config.d_hidden,
            d_out=n_species * config.n_noise_channels,
            d_context=d_context,
            n_hidden_layers=config.n_hidden_layers,
            dropout=config.mlp_dropout,
        )

    @property
    def n_species(self) -> int:
        """State dimension the SDE was constructed for."""
        return self._n_species

    def drift(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drift coefficient f(X, t; CRN embedding [; protocol embedding]).

        Args:
            t: Scalar time tensor.
            state: (n_species,) or (B, n_species) current state.
            crn_context: CRN encoder output.
            protocol_embedding: Optional (d_protocol,) or (B, d_protocol) tensor
                from ProtocolEncoder. When provided, concatenated to the CRN
                context vector before conditioning.

        Returns:
            (n_species,) or (B, n_species) drift vector.
        """
        return self.drift_from_context(
            t, state, crn_context.context_vector, protocol_embedding
        )

    def diffusion(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion coefficient g(X, t; CRN embedding [; protocol embedding]).

        Args:
            t: Scalar time tensor.
            state: (n_species,) or (B, n_species) current state.
            crn_context: CRN encoder output.
            protocol_embedding: Optional (d_protocol,) or (B, d_protocol) tensor
                from ProtocolEncoder. When provided, concatenated to the CRN
                context vector before conditioning.

        Returns:
            (n_species, n_noise_channels) or (B, n_species, n_noise_channels),
            non-negative (softplus applied).
        """
        return self.diffusion_from_context(
            t, state, crn_context.context_vector, protocol_embedding
        )

    def drift_from_context(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drift from a pre-extracted context vector.

        Unlike drift(), which extracts context_vector from a CRNContext, this
        method accepts the context vector directly. This enables batched
        computation where each transition in a (N, n_species) batch has its
        own context vector from a (N, d_context) tensor.

        Args:
            t: (N,) or scalar time values.
            x: (n_species,) or (N, n_species) current states.
            context_vector: (d_context,) or (N, d_context) context vectors.
            protocol_embedding: (d_protocol,) or (N, d_protocol) optional
                protocol embeddings.

        Returns:
            (n_species,) or (N, n_species) drift values.
        """
        ctx = context_vector
        if protocol_embedding is not None:
            ctx = torch.cat([ctx, protocol_embedding], dim=-1)
        return self._drift_net(x, ctx)

    def diffusion_from_context(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context_vector: torch.Tensor,
        protocol_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion from a pre-extracted context vector.

        Unlike diffusion(), which extracts context_vector from a CRNContext,
        this method accepts the context vector directly. This enables batched
        computation where each transition in a (N, n_species) batch has its
        own context vector from a (N, d_context) tensor.

        Args:
            t: (N,) or scalar time values.
            x: (n_species,) or (N, n_species) current states.
            context_vector: (d_context,) or (N, d_context) context vectors.
            protocol_embedding: (d_protocol,) or (N, d_protocol) optional
                protocol embeddings.

        Returns:
            (n_species, n_noise_channels) or (N, n_species, n_noise_channels),
            non-negative (softplus applied).
        """
        ctx = context_vector
        if protocol_embedding is not None:
            ctx = torch.cat([ctx, protocol_embedding], dim=-1)
        raw = F.softplus(self._diff_net(x, ctx))
        n_noise = self._config.n_noise_channels
        if x.dim() == 1:
            return raw.view(self._n_species, n_noise)
        return raw.view(x.shape[0], self._n_species, n_noise)
