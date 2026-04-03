"""MLP with per-layer FiLM conditioning for context-conditioned computation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crn_surrogate.simulator.film import FiLMLayer


class ConditionedMLP(nn.Module):
    """MLP with FiLM conditioning at every hidden layer.

    Architecture:
        x → Linear(d_in, d_hidden) → SiLU
          → [Linear(d_hidden, d_hidden) → FiLM(context) → SiLU] × n_hidden_layers
          → Linear(d_hidden, d_out)

    The context vector modulates intermediate representations at every hidden
    layer via FiLM (gamma * h + beta). This is strictly more expressive than
    applying FiLM only at the output, because the network can compute
    context-dependent intermediate features.

    The input projection does not receive FiLM conditioning (pure feature
    extraction). The output projection receives no FiLM and no activation;
    output-space constraints (e.g. softplus for diffusion) are the caller's
    responsibility.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        d_out: int,
        d_context: int,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Args:
            d_in: Input dimension.
            d_hidden: Hidden layer dimension.
            d_out: Output dimension.
            d_context: Dimension of the conditioning context vector.
            n_hidden_layers: Number of FiLM-conditioned hidden layers. Must be >= 1.
            dropout: Dropout probability applied after each hidden-layer activation.

        Raises:
            ValueError: If n_hidden_layers < 1.
        """
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        self._d_in = d_in
        self._d_hidden = d_hidden
        self._d_out = d_out
        self._d_context = d_context
        self._n_hidden_layers = n_hidden_layers

        self._input_proj = nn.Linear(d_in, d_hidden)
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_hidden_layers)]
        )
        self._film_layers = nn.ModuleList(
            [FiLMLayer(d_context, d_hidden) for _ in range(n_hidden_layers)]
        )
        self._output_proj = nn.Linear(d_hidden, d_out)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-layer FiLM conditioning.

        Args:
            x: (..., d_in) input features.
            context: (..., d_context) or (d_context,) conditioning vector.
                Broadcast to match x's leading dimensions.

        Returns:
            (..., d_out) output features.
        """
        h = F.silu(self._input_proj(x))
        for linear, film in zip(self._hidden_layers, self._film_layers):
            h = self._dropout(F.silu(film(linear(h), context)))
        return self._output_proj(h)

    def __repr__(self) -> str:
        return (
            f"ConditionedMLP(d_in={self._d_in}, d_hidden={self._d_hidden}, "
            f"d_out={self._d_out}, d_context={self._d_context}, "
            f"n_hidden_layers={self._n_hidden_layers})"
        )
