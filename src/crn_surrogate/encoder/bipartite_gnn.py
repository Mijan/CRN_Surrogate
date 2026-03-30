"""Bipartite GNN encoder that maps a CRNTensorRepr to contextualized embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.encoder.embeddings import ReactionEmbedding, SpeciesEmbedding
from crn_surrogate.encoder.graph_utils import BipartiteEdges
from crn_surrogate.encoder.message_passing import (
    AttentiveMessagePassingLayer,
    SumMessagePassingLayer,
)
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr


@dataclass(frozen=True)
class CRNContext:
    """Output of the CRN encoder: contextualized node embeddings + pooled context.

    Attributes:
        species_embeddings: (n_species, d_model)
        reaction_embeddings: (n_reactions, d_model)
        context_vector: (2 * d_model,) — mean-pooled species + mean-pooled reactions
    """

    species_embeddings: torch.Tensor
    reaction_embeddings: torch.Tensor
    context_vector: torch.Tensor


class BipartiteGNNEncoder(nn.Module):
    """Bipartite GNN encoder for Chemical Reaction Networks.

    Consumes a CRNTensorRepr (flat tensor representation) rather than the
    symbolic CRN object. The conversion from CRN to CRNTensorRepr is handled
    by crn_surrogate.encoder.tensor_repr.crn_to_tensor_repr.

    Produces contextualized species and reaction embeddings via L rounds of
    alternating message passing over the species-reaction bipartite graph.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
        config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._species_embed = SpeciesEmbedding(config)
        self._reaction_embed = ReactionEmbedding(config)
        layer_cls = (
            AttentiveMessagePassingLayer
            if config.use_attention
            else SumMessagePassingLayer
        )
        self._layers = nn.ModuleList(
            [layer_cls(config.d_model) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        crn_repr: CRNTensorRepr,
        initial_state: torch.Tensor,
    ) -> CRNContext:
        """Encode a CRN tensor representation and return contextualized embeddings.

        Args:
            crn_repr: Flat tensor representation of the CRN.
            initial_state: (n_species,) initial molecular counts.

        Returns:
            CRNContext with species embeddings, reaction embeddings, and context vector.
        """
        edges: BipartiteEdges = crn_repr.bipartite_edges

        h_species = self._species_embed(initial_state)
        h_reactions = self._reaction_embed(
            crn_repr.propensity_type_ids,
            crn_repr.propensity_params,
        )

        for layer in self._layers:
            h_species, h_reactions = layer(h_species, h_reactions, edges)

        context = self._pool_context(h_species, h_reactions)
        return CRNContext(
            species_embeddings=h_species,
            reaction_embeddings=h_reactions,
            context_vector=context,
        )

    def _pool_context(
        self, h_species: torch.Tensor, h_reactions: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool species and reaction embeddings into a fixed-size context vector."""
        pooled_species = h_species.mean(dim=0)
        pooled_reactions = h_reactions.mean(dim=0)
        return torch.cat([pooled_species, pooled_reactions], dim=-1)
