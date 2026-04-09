"""Tests for EncoderConfig, SDEConfig, and ModelConfig."""

from __future__ import annotations

import types

import pytest

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig

# ── EncoderConfig ─────────────────────────────────────────────────────────────


def test_type_embed_dim_auto():
    cfg = EncoderConfig(d_model=64)
    assert cfg.type_embed_dim == 16  # d_model // 4


def test_type_embed_dim_explicit():
    cfg = EncoderConfig(d_model=64, type_embed_dim=8)
    assert cfg.type_embed_dim == 8


def test_type_embed_dim_rejects_too_large():
    with pytest.raises(ValueError, match="type_embed_dim"):
        EncoderConfig(d_model=64, type_embed_dim=64)


def test_encoder_config_frozen():
    cfg = EncoderConfig(d_model=64)
    with pytest.raises((AttributeError, TypeError)):
        cfg.d_model = 128  # type: ignore[misc]


# ── SDEConfig ─────────────────────────────────────────────────────────────────


def test_rejects_zero_hidden_layers():
    with pytest.raises(ValueError, match="n_hidden_layers"):
        SDEConfig(n_hidden_layers=0)


def test_from_crn():
    stub = types.SimpleNamespace(n_reactions=5)
    cfg = SDEConfig.from_crn(stub)
    assert cfg.n_noise_channels == 5


def test_sde_config_frozen():
    cfg = SDEConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.d_hidden = 256  # type: ignore[misc]


# ── ModelConfig ───────────────────────────────────────────────────────────────


def test_default_construction():
    cfg = ModelConfig()
    assert cfg.encoder.d_model == 64
    assert cfg.sde.d_hidden == 128
