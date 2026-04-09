"""Tests for FiLMLayer."""

from __future__ import annotations

import torch

from crn_surrogate.simulator.film import FiLMLayer


def test_output_shape() -> None:
    layer = FiLMLayer(d_context=8, d_features=16)
    x = torch.randn(5, 16)
    context = torch.randn(8)
    out = layer(x, context)
    assert out.shape == (5, 16)


def test_context_changes_output() -> None:
    layer = FiLMLayer(d_context=8, d_features=16)
    x = torch.randn(5, 16)
    ctx1 = torch.randn(8)
    ctx2 = torch.randn(8)
    out1 = layer(x, ctx1)
    out2 = layer(x, ctx2)
    assert not torch.allclose(out1, out2)


def test_batched_context() -> None:
    layer = FiLMLayer(d_context=8, d_features=16)
    x = torch.randn(3, 16)
    context = torch.randn(3, 8)
    out = layer(x, context)
    assert out.shape == (3, 16)


def test_identity_possible() -> None:
    """Verify FiLM is not a no-op: different contexts produce different modulations."""
    layer = FiLMLayer(d_context=8, d_features=16)
    x = torch.randn(5, 16)
    ctx1 = torch.zeros(8)
    ctx2 = torch.ones(8)
    out1 = layer(x, ctx1)
    out2 = layer(x, ctx2)
    assert not torch.allclose(out1, out2)
