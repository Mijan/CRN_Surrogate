"""Bipartite GNN encoder that maps a CRNTensorRepr to contextualized embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.encoder.embeddings import ReactionEmbedding, SpeciesEmbedding
from crn_surrogate.encoder.graph_utils import BipartiteEdges, merge_bipartite_edges
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
        self._context_dropout = nn.Dropout(config.context_dropout)

    def forward(self, crn_repr: CRNTensorRepr) -> CRNContext:
        """Encode a CRN tensor representation and return contextualized embeddings.

        The encoding depends only on the CRN's topology and kinetics, not on
        any particular initial state. This ensures the same CRN always produces
        the same context vector regardless of experimental conditions.

        Args:
            crn_repr: Flat tensor representation of the CRN.

        Returns:
            CRNContext with species embeddings, reaction embeddings, and
            context vector.
        """
        edges: BipartiteEdges = crn_repr.bipartite_edges

        h_species = self._species_embed(
            n_species=crn_repr.n_species,
            is_external=crn_repr.is_external,
            device=crn_repr.stoichiometry.device,
        )
        h_reactions = self._reaction_embed(
            crn_repr.propensity_type_ids,
            crn_repr.propensity_params,
        )

        for layer in self._layers:
            h_species, h_reactions = layer(h_species, h_reactions, edges)

        context = self._context_dropout(self._pool_context(h_species, h_reactions))
        return CRNContext(
            species_embeddings=h_species,
            reaction_embeddings=h_reactions,
            context_vector=context,
        )

    def forward_batch(self, crn_reprs: list[CRNTensorRepr]) -> list[CRNContext]:
        """Encode multiple CRNs in a single batched forward pass.

        Concatenates all CRN graphs into one disconnected graph with offset
        edge indices. Message passing processes the combined graph in one pass.
        Per-CRN context vectors are produced by segment-mean pooling.

        This is mathematically equivalent to calling forward() on each CRN
        individually, but executes as a single GPU operation instead of
        B sequential ones.

        Args:
            crn_reprs: List of B CRNTensorRepr objects. Bipartite edges should
                be pre-built (e.g. by accessing .bipartite_edges during
                collation on CPU).

        Returns:
            List of B CRNContext objects, one per input CRN.
        """
        B = len(crn_reprs)
        device = crn_reprs[0].stoichiometry.device
        n_species_list = [r.n_species for r in crn_reprs]
        n_reactions_list = [r.n_reactions for r in crn_reprs]

        # Merge all bipartite edges into one disconnected graph
        merged_edges = merge_bipartite_edges(
            [r.bipartite_edges for r in crn_reprs],
            n_species_list,
            n_reactions_list,
        )

        # Concatenated species embeddings: local indices per CRN (not globally offset)
        species_indices = torch.cat(
            [torch.arange(ns, device=device) for ns in n_species_list]
        )
        is_external = torch.cat([r.is_external for r in crn_reprs])
        h_species = self._species_embed.embed_from_indices(species_indices, is_external)

        # Concatenated reaction embeddings
        all_type_ids = torch.cat([r.propensity_type_ids for r in crn_reprs])
        all_params = torch.cat([r.propensity_params for r in crn_reprs])
        h_reactions = self._reaction_embed(all_type_ids, all_params)

        # Message passing on the merged disconnected graph
        for layer in self._layers:
            h_species, h_reactions = layer(h_species, h_reactions, merged_edges)

        # Segment-mean pooling per CRN, then dropout
        context_vectors = self._pool_context_batched(
            h_species,
            h_reactions,
            n_species_list,
            n_reactions_list,
            device,
        )
        context_vectors = self._context_dropout(context_vectors)  # (B, 2*d_model)

        # Split per-node embeddings back to individual CRNs
        species_splits = torch.split(h_species, n_species_list)
        reaction_splits = torch.split(h_reactions, n_reactions_list)

        return [
            CRNContext(
                species_embeddings=species_splits[i],
                reaction_embeddings=reaction_splits[i],
                context_vector=context_vectors[i],
            )
            for i in range(B)
        ]

    def _pool_context_batched(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        n_species_list: list[int],
        n_reactions_list: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Segment-mean pool species and reactions per CRN.

        Args:
            h_species: (total_species, d_model) concatenated species embeddings.
            h_reactions: (total_reactions, d_model) concatenated reaction embeddings.
            n_species_list: Number of species per CRN.
            n_reactions_list: Number of reactions per CRN.
            device: Target device.

        Returns:
            (B, 2 * d_model) context vectors, one per CRN.
        """
        B = len(n_species_list)
        d = self._config.d_model

        species_batch = torch.cat(
            [
                torch.full((ns,), i, dtype=torch.long, device=device)
                for i, ns in enumerate(n_species_list)
            ]
        )
        reaction_batch = torch.cat(
            [
                torch.full((nr,), i, dtype=torch.long, device=device)
                for i, nr in enumerate(n_reactions_list)
            ]
        )

        pooled_species = torch.zeros(B, d, device=device)
        pooled_species.index_add_(0, species_batch, h_species)
        pooled_species = pooled_species / torch.tensor(
            n_species_list, dtype=torch.float, device=device
        ).unsqueeze(1)

        pooled_reactions = torch.zeros(B, d, device=device)
        pooled_reactions.index_add_(0, reaction_batch, h_reactions)
        pooled_reactions = pooled_reactions / torch.tensor(
            n_reactions_list, dtype=torch.float, device=device
        ).unsqueeze(1)

        return torch.cat([pooled_species, pooled_reactions], dim=-1)

    def _pool_context(
        self, h_species: torch.Tensor, h_reactions: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool species and reaction embeddings into a fixed-size context vector."""
        pooled_species = h_species.mean(dim=0)
        pooled_reactions = h_reactions.mean(dim=0)
        return torch.cat([pooled_species, pooled_reactions], dim=-1)
