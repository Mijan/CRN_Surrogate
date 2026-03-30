"""Feature-wise Linear Modulation (FiLM) conditioning layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioning layer.

    Applies: output = gamma(context) * x + beta(context)

    Args:
        d_context: Dimension of the conditioning context vector.
        d_features: Dimension of the features to modulate.
    """

    def __init__(self, d_context: int, d_features: int) -> None:
        super().__init__()
        self._gamma = nn.Linear(d_context, d_features)
        self._beta = nn.Linear(d_context, d_features)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: (..., d_features) features to modulate.
            context: (d_context,) or (..., d_context) context vector.

        Returns:
            Modulated features, same shape as x.
        """
        gamma = self._gamma(context)
        beta = self._beta(context)
        return gamma * x + beta
