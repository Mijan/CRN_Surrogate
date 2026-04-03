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
        ctx = crn_context.context_vector
        if protocol_embedding is not None:
            ctx = torch.cat([ctx, protocol_embedding], dim=-1)
        return self._drift_net(state, ctx)

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
        ctx = crn_context.context_vector
        if protocol_embedding is not None:
            ctx = torch.cat([ctx, protocol_embedding], dim=-1)
        raw = self._diff_net(state, ctx)
        raw = F.softplus(raw)
        n_noise = self._config.n_noise_channels
        if state.dim() == 1:
            return raw.view(self._n_species, n_noise)
        return raw.view(state.shape[0], self._n_species, n_noise)
