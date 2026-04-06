from __future__ import annotations

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import EncoderConfig


class SpeciesEmbedding(nn.Module):
    """Initial feature embedding for species nodes.

    Uses a learnable species identity embedding (analogous to positional
    encoding) plus an external-species flag projection. The embedding is
    purely structural and does not depend on the initial state.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
        config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._identity_embed = nn.Embedding(config.max_species, config.d_model)
        self._external_proj = nn.Linear(1, config.d_model)

    def forward(
        self,
        n_species: int,
        is_external: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Embed species from their positional identities and external flags.

        Args:
            n_species: Number of species in this CRN.
            is_external: (n_species,) boolean tensor; True for externally
                controlled species. Adds a learned projection of the flag.
            device: Device for the index tensor. If None, uses the device
                of the embedding weight.

        Returns:
            (n_species, d_model) embeddings.
        """
        if device is None:
            device = self._identity_embed.weight.device
        indices = torch.arange(n_species, device=device)
        return self.embed_from_indices(indices, is_external)

    def embed_from_indices(
        self,
        species_indices: torch.Tensor,
        is_external: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed species from pre-built local identity indices.

        Unlike forward(), which constructs arange(n_species), this method
        accepts a pre-built index tensor. For batched encoding, pass
        cat([arange(n_s0), arange(n_s1), ...]) so each CRN's species get
        local identity indices (0, 1, 2, ...) regardless of their position
        in the concatenated graph.

        Args:
            species_indices: (total_species,) local identity indices.
            is_external: (total_species,) boolean tensor.

        Returns:
            (total_species, d_model) embeddings.
        """
        h = self._identity_embed(species_indices)
        if is_external is not None:
            ext_emb = self._external_proj(is_external.float().unsqueeze(-1))
            h = h + ext_emb
        return h


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
        type_emb = self._type_embed(propensity_type_ids)  # (n_reactions, type_dim)
        param_emb = self._param_proj(propensity_params)  # (n_reactions, param_dim)
        combined = torch.cat([type_emb, param_emb], dim=-1)  # (n_reactions, d_model)
        return self._out_proj(combined)
