from __future__ import annotations

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import EncoderConfig


class SpeciesEmbedding(nn.Module):
    """Initial feature embedding for species nodes.

    Combines a concentration projection and a learnable species identity
    embedding (analogous to positional encoding). The two contributions are
    summed so the output dimension remains d_model.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
        config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._conc_proj = nn.Linear(1, config.d_model)
        self._identity_embed = nn.Embedding(config.max_species, config.d_model)

    def forward(self, initial_concentration: torch.Tensor) -> torch.Tensor:
        """Embed species from their initial concentrations and positional identities.

        Args:
            initial_concentration: (n_species,) initial molecular counts.

        Returns:
            (n_species, d_model) embeddings.
        """
        n_species = initial_concentration.shape[-1]
        indices = torch.arange(n_species, device=initial_concentration.device)
        conc_emb = self._conc_proj(initial_concentration.unsqueeze(-1))  # (n_species, d_model)
        id_emb = self._identity_embed(indices)                            # (n_species, d_model)
        return conc_emb + id_emb


class ReactionEmbedding(nn.Module):
    """Initial feature embedding for reaction nodes.

    Combines a propensity type embedding (small dimension) and a parameter
    projection (larger dimension), then applies an output projection.
    The type/param dimension split is controlled by EncoderConfig.type_embed_dim.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
        config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        type_dim = config.type_embed_dim
        param_dim = config.d_model - type_dim
        self._type_embed = nn.Embedding(config.n_propensity_types, type_dim)
        self._param_proj = nn.Linear(config.max_propensity_params, param_dim)
        self._out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        propensity_type_ids: torch.Tensor,
        propensity_params: torch.Tensor,
    ) -> torch.Tensor:
        """Embed reactions from propensity type and parameters.

        Args:
            propensity_type_ids: (n_reactions,) int type IDs.
            propensity_params: (n_reactions, max_params) kinetic parameters.

        Returns:
            (n_reactions, d_model) embeddings.
        """
        type_emb = self._type_embed(propensity_type_ids)   # (n_reactions, type_dim)
        param_emb = self._param_proj(propensity_params)     # (n_reactions, param_dim)
        combined = torch.cat([type_emb, param_emb], dim=-1)  # (n_reactions, d_model)
        return self._out_proj(combined)
