"""Bipartite message-passing layers for species-reaction graphs.

Provides two implementations sharing the same forward signature:
- SumMessagePassingLayer: plain MLP + sum aggregation (default).
- AttentiveMessagePassingLayer: MLP + attention-weighted aggregation.

BipartiteGNNEncoder selects which class to instantiate via EncoderConfig.use_attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crn_surrogate.encoder.graph_utils import (
    EDGE_FEAT_DIM,
    BipartiteEdges,
    _scatter_max,
)


class SumMessagePassingLayer(nn.Module):
    """One round of bipartite message passing with sum aggregation.

    Performs rxn→species then species→rxn using MLP messages and
    LayerNorm residual updates.
    """

    def __init__(self, d_model: int) -> None:
        """Args:
        d_model: Hidden dimension for species and reaction nodes.
        """
        super().__init__()
        self._d_model = d_model

        # Reaction → Species
        self._mlp_rs = nn.Sequential(
            nn.Linear(d_model + EDGE_FEAT_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._norm_s = nn.LayerNorm(d_model)

        # Species → Reaction
        self._mlp_sr = nn.Sequential(
            nn.Linear(d_model + EDGE_FEAT_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._norm_r = nn.LayerNorm(d_model)

    def forward(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One message-passing round.

        Args:
            h_species: (n_species, d_model) species node features.
            h_reactions: (n_reactions, d_model) reaction node features.
            edges: Bipartite edge indices and features.

        Returns:
            Updated (h_species, h_reactions).
        """
        h_species = self._rxn_to_species(h_species, h_reactions, edges)
        h_reactions = self._species_to_rxn(h_species, h_reactions, edges)
        return h_species, h_reactions

    def _rxn_to_species(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> torch.Tensor:
        """Aggregate reaction→species messages and update species."""
        rxn_idx = edges.rxn_to_species_index[0]
        spe_idx = edges.rxn_to_species_index[1]
        edge_feat = edges.rxn_to_species_feat

        msg_input = torch.cat([h_reactions[rxn_idx], edge_feat], dim=-1)
        msgs = self._mlp_rs(msg_input)

        agg = torch.zeros_like(h_species)
        agg.index_add_(0, spe_idx, msgs)

        return self._norm_s(h_species + agg)

    def _species_to_rxn(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> torch.Tensor:
        """Aggregate species→reaction messages and update reactions."""
        spe_idx = edges.species_to_rxn_index[0]
        rxn_idx = edges.species_to_rxn_index[1]
        edge_feat = edges.species_to_rxn_feat

        msg_input = torch.cat([h_species[spe_idx], edge_feat], dim=-1)
        msgs = self._mlp_sr(msg_input)

        agg = torch.zeros_like(h_reactions)
        agg.index_add_(0, rxn_idx, msgs)

        return self._norm_r(h_reactions + agg)


class AttentiveMessagePassingLayer(nn.Module):
    """One round of bipartite message passing with attention-weighted aggregation.

    Same MLP for computing messages as SumMessagePassingLayer, but adds
    query/key projections so the network can learn to suppress irrelevant
    incoming messages based on current node state.
    """

    def __init__(self, d_model: int) -> None:
        """Args:
        d_model: Hidden dimension for species and reaction nodes.
        """
        super().__init__()
        self._d_model = d_model
        d_attn = d_model // 4

        # Reaction → Species
        self._mlp_rs = nn.Sequential(
            nn.Linear(d_model + EDGE_FEAT_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._query_rs = nn.Linear(d_model, d_attn)
        self._key_rs = nn.Linear(d_model, d_attn)
        self._norm_s = nn.LayerNorm(d_model)

        # Species → Reaction
        self._mlp_sr = nn.Sequential(
            nn.Linear(d_model + EDGE_FEAT_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._query_sr = nn.Linear(d_model, d_attn)
        self._key_sr = nn.Linear(d_model, d_attn)
        self._norm_r = nn.LayerNorm(d_model)

        self._scale: float = d_attn**0.5

    def forward(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One attention-weighted message-passing round.

        Args:
            h_species: (n_species, d_model) species node features.
            h_reactions: (n_reactions, d_model) reaction node features.
            edges: Bipartite edge indices and features.

        Returns:
            Updated (h_species, h_reactions).
        """
        h_species = self._rxn_to_species(h_species, h_reactions, edges)
        h_reactions = self._species_to_rxn(h_species, h_reactions, edges)
        return h_species, h_reactions

    def _rxn_to_species(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> torch.Tensor:
        """Compute attention-weighted reaction→species messages."""
        rxn_idx = edges.rxn_to_species_index[0]
        spe_idx = edges.rxn_to_species_index[1]
        edge_feat = edges.rxn_to_species_feat

        msg_input = torch.cat([h_reactions[rxn_idx], edge_feat], dim=-1)
        msgs = self._mlp_rs(msg_input)  # (E, d_model)

        queries = self._query_rs(h_species[spe_idx])  # (E, d_attn)
        keys = self._key_rs(msgs)  # (E, d_attn)
        raw_scores = (queries * keys).sum(dim=-1) / self._scale  # (E,)

        weights = self._scatter_softmax(raw_scores, spe_idx, h_species.shape[0])

        agg = torch.zeros_like(h_species)
        agg.index_add_(0, spe_idx, msgs * weights.unsqueeze(-1))

        return self._norm_s(h_species + agg)

    def _species_to_rxn(
        self,
        h_species: torch.Tensor,
        h_reactions: torch.Tensor,
        edges: BipartiteEdges,
    ) -> torch.Tensor:
        """Compute attention-weighted species→reaction messages."""
        spe_idx = edges.species_to_rxn_index[0]
        rxn_idx = edges.species_to_rxn_index[1]
        edge_feat = edges.species_to_rxn_feat

        msg_input = torch.cat([h_species[spe_idx], edge_feat], dim=-1)
        msgs = self._mlp_sr(msg_input)  # (E, d_model)

        queries = self._query_sr(h_reactions[rxn_idx])  # (E, d_attn)
        keys = self._key_sr(msgs)  # (E, d_attn)
        raw_scores = (queries * keys).sum(dim=-1) / self._scale  # (E,)

        weights = self._scatter_softmax(raw_scores, rxn_idx, h_reactions.shape[0])

        agg = torch.zeros_like(h_reactions)
        agg.index_add_(0, rxn_idx, msgs * weights.unsqueeze(-1))

        return self._norm_r(h_reactions + agg)

    @staticmethod
    def _scatter_softmax(
        raw_scores: torch.Tensor,
        index: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """Stable scatter softmax: normalize scores within each receiving node's group.

        Args:
            raw_scores: (E,) unnormalized attention scores.
            index: (E,) receiving-node group assignments.
            num_groups: Total number of receiving nodes.

        Returns:
            (E,) attention weights summing to 1 per group.
        """
        group_max = _scatter_max(raw_scores, index, num_groups)
        scores_shifted = raw_scores - group_max[index]
        weights = scores_shifted.exp()
        weight_sums = torch.zeros(
            num_groups, device=weights.device, dtype=weights.dtype
        )
        weight_sums.index_add_(0, index, weights)
        return weights / (weight_sums[index] + 1e-8)
