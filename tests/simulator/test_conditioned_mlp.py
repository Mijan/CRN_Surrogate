"""Tests for ConditionedMLP."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.simulator.conditioned_mlp import ConditionedMLP


def _make_mlp(**kwargs) -> ConditionedMLP:
    defaults = dict(d_in=3, d_hidden=16, d_out=5, d_context=8, n_hidden_layers=2)
    defaults.update(kwargs)
    return ConditionedMLP(**defaults)


def test_output_shape() -> None:
    mlp = _make_mlp()
    x = torch.randn(3)
    ctx = torch.randn(8)
    out = mlp(x, ctx)
    assert out.shape == (5,)


def test_batched_input() -> None:
    mlp = _make_mlp()
    x = torch.randn(10, 3)
    ctx = torch.randn(8)
    out = mlp(x, ctx)
    assert out.shape == (10, 5)


def test_batched_context() -> None:
    mlp = _make_mlp()
    x = torch.randn(10, 3)
    ctx = torch.randn(10, 8)
    out = mlp(x, ctx)
    assert out.shape == (10, 5)


def test_context_affects_output() -> None:
    mlp = _make_mlp()
    x = torch.randn(3)
    ctx1 = torch.randn(8)
    ctx2 = torch.randn(8)
    out1 = mlp(x, ctx1)
    out2 = mlp(x, ctx2)
    assert not torch.allclose(out1, out2)


def test_rejects_zero_hidden_layers() -> None:
    with pytest.raises(ValueError):
        _make_mlp(n_hidden_layers=0)


def test_gradients_flow_through_context() -> None:
    mlp = _make_mlp()
    x = torch.randn(3)
    ctx = torch.randn(8, requires_grad=True)
    out = mlp(x, ctx)
    out.sum().backward()
    assert ctx.grad is not None
    assert ctx.grad.abs().sum().item() > 0


def test_dropout_changes_output_in_train_mode() -> None:
    mlp = _make_mlp(dropout=0.5)
    x = torch.randn(10, 3)
    ctx = torch.randn(8)

    mlp.train()
    out1 = mlp(x, ctx)
    out2 = mlp(x, ctx)
    assert not torch.allclose(out1, out2)

    mlp.eval()
    with torch.no_grad():
        out3 = mlp(x, ctx)
        out4 = mlp(x, ctx)
    assert torch.allclose(out3, out4)
