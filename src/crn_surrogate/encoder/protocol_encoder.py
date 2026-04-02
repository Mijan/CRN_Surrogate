"""DeepSets protocol encoder: maps a batch of InputProtocol objects to embeddings.

Architecture: per-event feature MLP → masked sum-pool → output projection.
Permutation invariance over events is guaranteed by the sum-pool aggregation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import ProtocolEncoderConfig

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import InputProtocol

# Scalar feature channels appended after the species embedding:
#   t_start, t_end, amplitude, log(amplitude), duration, midpoint
_N_SCALAR_FEATURES: int = 6


class ProtocolEncoder(nn.Module):
    """DeepSets encoder that maps a batch of InputProtocol objects to fixed-size vectors.

    Per-event features are computed for every PulseEvent in the protocol (across
    all input species). A learned species embedding distinguishes which input
    species each event belongs to (using a local 0-indexed rank rather than
    global species indices, since global connectivity is already captured by the
    bipartite GNN). Events are aggregated by masked sum-pooling, making the
    encoder invariant to event order.

    An empty protocol (no events) produces a zero vector.
    """

    def __init__(self, config: ProtocolEncoderConfig) -> None:
        """Args:
        config: Protocol encoder configuration.
        """
        super().__init__()
        self._config = config
        raw_dim = config.species_embed_dim + _N_SCALAR_FEATURES

        self._species_embed = nn.Embedding(
            config.max_input_species, config.species_embed_dim
        )

        layers: list[nn.Module] = [nn.Linear(raw_dim, config.d_event), nn.SiLU()]
        for _ in range(config.n_layers - 1):
            layers += [nn.Linear(config.d_event, config.d_event), nn.SiLU()]
        self._event_mlp = nn.Sequential(*layers)

        self._out_proj = nn.Linear(config.d_event, config.d_protocol)

    def forward(self, protocols: list[InputProtocol]) -> torch.Tensor:
        """Encode a batch of input protocols.

        Args:
            protocols: One InputProtocol per batch item. May include EMPTY_PROTOCOL.

        Returns:
            (batch, d_protocol) protocol embedding tensor.
            Returns zeros for items with no events (empty protocols).
        """
        device = next(self.parameters()).device
        features, mask = self._build_event_features(protocols, device)
        # features: (B, max_events, raw_dim), mask: (B, max_events)

        if mask.sum() == 0:
            # All protocols are empty; return zeros directly without a forward pass.
            return torch.zeros(len(protocols), self._config.d_protocol, device=device)

        B, max_events, _ = features.shape
        # Run all events through the per-event MLP in a single batched pass.
        flat = features.view(B * max_events, -1)  # (B*max_events, raw_dim)
        flat_out = self._event_mlp(flat)  # (B*max_events, d_event)
        event_out = flat_out.view(B, max_events, -1)  # (B, max_events, d_event)

        # Zero out padded positions before sum-pooling.
        event_out = event_out * mask.unsqueeze(-1).float()  # (B, max_events, d_event)

        # Sum-pool over events → (B, d_event)
        pooled = event_out.sum(dim=1)

        out = self._out_proj(pooled)  # (B, d_protocol)

        # Items with no events must produce exactly zero (not the projection bias).
        has_events = mask.any(dim=1)  # (B,)
        out = out * has_events.unsqueeze(-1).float()

        return out

    def _build_event_features(
        self,
        protocols: list[InputProtocol],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of protocols to a padded feature tensor and boolean mask.

        For each protocol, events are collected across all schedules in sorted
        species-index order. Within each schedule, events follow the schedule's
        sorted order. The local species index k (0-indexed by sorted position
        among the species present in that protocol) is used for the species
        embedding.

        Args:
            protocols: List of InputProtocol, one per batch item.
            device: Target device for the tensors.

        Returns:
            features: (B, max_events, raw_dim) padded feature tensor.
            mask: (B, max_events) boolean tensor; True for real events.
        """
        B = len(protocols)
        raw_dim = self._config.species_embed_dim + _N_SCALAR_FEATURES

        # Collect per-item event lists first to find max_events.
        all_items: list[list[tuple[int, float, float, float]]] = []
        for protocol in protocols:
            item_events: list[tuple[int, float, float, float]] = []
            sorted_species = sorted(protocol.schedules.keys())
            for local_k, species_idx in enumerate(sorted_species):
                schedule = protocol.schedules[species_idx]
                for event in schedule.events:
                    item_events.append(
                        (local_k, event.t_start, event.t_end, event.amplitude)
                    )
            all_items.append(item_events)

        max_events = max((len(evts) for evts in all_items), default=0)
        if max_events == 0:
            empty = torch.zeros(B, 0, raw_dim, device=device)
            empty_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return empty, empty_mask

        # Collect all events across the entire batch into flat lists.
        all_local_k: list[int] = []
        all_scalars: list[list[float]] = []
        event_batch_idx: list[int] = []
        event_position: list[int] = []

        for i, item_events in enumerate(all_items):
            for j, (local_k, t_start, t_end, amplitude) in enumerate(item_events):
                duration = t_end - t_start
                midpoint = (t_start + t_end) / 2.0
                all_local_k.append(local_k)
                all_scalars.append(
                    [t_start, t_end, amplitude, math.log(amplitude), duration, midpoint]
                )
                event_batch_idx.append(i)
                event_position.append(j)

        features = torch.zeros(B, max_events, raw_dim, device=device)
        mask = torch.zeros(B, max_events, dtype=torch.bool, device=device)

        if all_local_k:
            # Single batched embedding lookup for all events.
            species_embs = self._species_embed(
                torch.tensor(all_local_k, dtype=torch.long, device=device)
            )  # (total_events, species_embed_dim)
            scalar_tensor = torch.tensor(
                all_scalars, dtype=torch.float32, device=device
            )  # (total_events, 6)
            all_features = torch.cat([species_embs, scalar_tensor], dim=-1)

            # Scatter into the padded tensor.
            for flat_idx in range(len(all_local_k)):
                bi = event_batch_idx[flat_idx]
                pos = event_position[flat_idx]
                features[bi, pos] = all_features[flat_idx]
                mask[bi, pos] = True

        return features, mask
