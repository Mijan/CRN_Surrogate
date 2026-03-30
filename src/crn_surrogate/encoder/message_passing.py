"""Bipartite message-passing layer for species-reaction graphs."""
from __future__ import annotations

import torch
import torch.nn as nn

from crn_surrogate.encoder.tensor_repr import BipartiteEdges


class MessagePassingLayer(nn.Module):
    """One round of bipartite message passing: rxn→species then species→rxn.

    Uses sum aggregation and LayerNorm residual updates.
    Edge features have dimension 2 (reactant_count, net_change).
    """

    _EDGE_FEAT_DIM: int = 2

    def __init__(self, d_model: int) -> None:
        """Args:
        d_model: Hidden dimension for species and reaction nodes.
        """
        super().__init__()
        self._d_model = d_model

        # Reaction → Species
        self._mlp_rs = nn.Sequential(
            nn.Linear(d_model + self._EDGE_FEAT_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self._norm_s = nn.LayerNorm(d_model)

        # Species → Reaction
        self._mlp_sr = nn.Sequential(
            nn.Linear(d_model + self._EDGE_FEAT_DIM, d_model),
            nn.ReLU(),
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
