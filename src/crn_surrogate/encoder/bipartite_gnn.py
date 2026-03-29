from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.data.crn import CRNDefinition, build_bipartite_edges, BipartiteEdges
from crn_surrogate.encoder.embeddings import SpeciesEmbedding, ReactionEmbedding
from crn_surrogate.encoder.message_passing import MessagePassingLayer


@dataclass
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

    Produces contextualized species and reaction embeddings via
    L rounds of alternating message passing over the species-reaction graph.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Args:
            config: Encoder configuration.
        """
        super().__init__()
        self._config = config
        self._species_embed = SpeciesEmbedding(config)
        self._reaction_embed = ReactionEmbedding(config)
        self._layers = nn.ModuleList([
            MessagePassingLayer(config.d_model) for _ in range(config.n_layers)
        ])

    def forward(
        self,
        crn: CRNDefinition,
        initial_state: torch.Tensor,
    ) -> CRNContext:
        """Encode a CRN and return contextualized embeddings.

        Args:
            crn: The CRN definition.
            initial_state: (n_species,) initial molecular counts.

        Returns:
            CRNContext with species embeddings, reaction embeddings, and context vector.
        """
        edges = build_bipartite_edges(crn.stoichiometry, crn.reactant_matrix)

        h_species = self._species_embed(initial_state)
        h_reactions = self._reaction_embed(
            torch.tensor([pt.value for pt in crn.propensity_types]),
            crn.propensity_params,
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
