from __future__ import annotations
import torch
import torch.nn as nn
from crn_surrogate.configs.model_config import EncoderConfig


class SpeciesEmbedding(nn.Module):
    """Initial feature embedding for species nodes.

    Input features: initial concentration (scalar) → projected to d_model.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
            config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._proj = nn.Linear(1, config.d_model)

    def forward(self, initial_concentration: torch.Tensor) -> torch.Tensor:
        """Embed species from their initial concentrations.

        Args:
            initial_concentration: (B, n_species) or (n_species,) initial counts.

        Returns:
            (B, n_species, d_model) or (n_species, d_model) embeddings.
        """
        return self._proj(initial_concentration.unsqueeze(-1))


class ReactionEmbedding(nn.Module):
    """Initial feature embedding for reaction nodes.

    Input features: propensity type embedding + propensity parameters.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
            config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._type_embed = nn.Embedding(config.n_propensity_types, config.d_model // 2)
        self._param_proj = nn.Linear(config.max_propensity_params, config.d_model // 2)
        self._out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        propensity_type_ids: torch.Tensor,
        propensity_params: torch.Tensor,
    ) -> torch.Tensor:
        """Embed reactions from propensity type and parameters.

        Args:
            propensity_type_ids: (B, n_reactions) or (n_reactions,) int type IDs.
            propensity_params: (B, n_reactions, max_params) or (n_reactions, max_params).

        Returns:
            (B, n_reactions, d_model) or (n_reactions, d_model) embeddings.
        """
        type_emb = self._type_embed(propensity_type_ids)
        param_emb = self._param_proj(propensity_params)
        combined = torch.cat([type_emb, param_emb], dim=-1)
        return self._out_proj(combined)
