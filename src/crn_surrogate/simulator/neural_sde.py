from __future__ import annotations
import torch
import torch.nn as nn
from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioning layer.

    Applies: output = gamma(context) * x + beta(context)
    """

    def __init__(self, d_context: int, d_features: int) -> None:
        """Args:
            d_context: Dimension of the conditioning context vector.
            d_features: Dimension of the features to modulate.
        """
        super().__init__()
        self._gamma = nn.Linear(d_context, d_features)
        self._beta = nn.Linear(d_context, d_features)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: (..., d_features) features to modulate.
            context: (d_context,) or (B, d_context) context vector.

        Returns:
            Modulated features, same shape as x.
        """
        gamma = self._gamma(context)
        beta = self._beta(context)
        return gamma * x + beta


class CRNNeuralSDE(nn.Module):
    """Neural SDE whose drift and diffusion are conditioned on CRN embeddings.

    Mirrors the Chemical Langevin Equation structure:
      dX = f(X, t; ctx) dt + g(X, t; ctx) dW
    where f and g are learned MLPs conditioned on the CRN encoder output
    via FiLM modulation.
    """

    def __init__(self, config: SDEConfig, n_species: int) -> None:
        """Args:
            config: SDE configuration.
            n_species: Number of species (state dimension).
        """
        super().__init__()
        self._config = config
        self._n_species = n_species
        d_context = 2 * config.d_model  # species + reaction pool concatenated

        # Drift network
        self._drift_mlp = nn.Sequential(
            nn.Linear(n_species, config.d_hidden),
            nn.SiLU(),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.SiLU(),
            nn.Linear(config.d_hidden, n_species),
        )
        self._drift_film = FiLMLayer(d_context, n_species)

        # Diffusion network
        self._diff_mlp = nn.Sequential(
            nn.Linear(n_species, config.d_hidden),
            nn.SiLU(),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.SiLU(),
            nn.Linear(config.d_hidden, n_species * config.n_noise_channels),
        )
        self._diff_film = FiLMLayer(d_context, n_species * config.n_noise_channels)

    def drift(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
    ) -> torch.Tensor:
        """Compute drift coefficient f(X, t; CRN embedding).

        Args:
            t: Scalar time tensor.
            state: (n_species,) or (B, n_species) current state.
            crn_context: CRN encoder output.

        Returns:
            (n_species,) or (B, n_species) drift vector.
        """
        h = self._drift_mlp(state)
        return self._drift_film(h, crn_context.context_vector)

    def diffusion(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        crn_context: CRNContext,
    ) -> torch.Tensor:
        """Compute diffusion coefficient g(X, t; CRN embedding).

        Args:
            t: Scalar time tensor.
            state: (n_species,) or (B, n_species) current state.
            crn_context: CRN encoder output.

        Returns:
            (n_species, n_noise_channels) or (B, n_species, n_noise_channels),
            non-negative (softplus applied).
        """
        raw = self._diff_mlp(state)
        raw = self._diff_film(raw, crn_context.context_vector)
        raw = nn.functional.softplus(raw)
        n_noise = self._config.n_noise_channels
        if state.dim() == 1:
            return raw.view(self._n_species, n_noise)
        return raw.view(state.shape[0], self._n_species, n_noise)
