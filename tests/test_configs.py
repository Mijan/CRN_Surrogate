"""Tests for EncoderConfig and SDEConfig configuration dataclasses.

Covers:
- EncoderConfig.__post_init__ auto-computes type_embed_dim = d_model // 4.
- An explicit type_embed_dim is preserved unchanged.
- EncoderConfig raises ValueError when type_embed_dim >= d_model.
- SDEConfig.from_crn() sets n_noise_channels = crn.n_reactions.
"""

import pytest

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.crn.examples import birth_death, lotka_volterra, simple_mapk_cascade

# ── EncoderConfig ─────────────────────────────────────────────────────────────


def test_encoder_config_type_embed_dim_defaults_to_d_model_over_4():
    """When type_embed_dim is not specified, __post_init__ sets it to d_model // 4."""
    config = EncoderConfig(d_model=64)
    assert config.type_embed_dim == 16


def test_encoder_config_type_embed_dim_default_varies_with_d_model():
    """The auto-computed type_embed_dim tracks d_model // 4 for any d_model."""
    for d_model in (8, 32, 128):
        config = EncoderConfig(d_model=d_model)
        assert config.type_embed_dim == d_model // 4, (
            f"d_model={d_model}: expected type_embed_dim={d_model // 4}, "
            f"got {config.type_embed_dim}"
        )


def test_encoder_config_explicit_type_embed_dim_is_preserved():
    """A non-zero explicit type_embed_dim is kept as provided."""
    config = EncoderConfig(d_model=64, type_embed_dim=8)
    assert config.type_embed_dim == 8


def test_encoder_config_raises_when_type_embed_dim_equals_d_model():
    """type_embed_dim == d_model must raise ValueError."""
    with pytest.raises(ValueError, match="type_embed_dim"):
        EncoderConfig(d_model=16, type_embed_dim=16)


def test_encoder_config_raises_when_type_embed_dim_exceeds_d_model():
    """type_embed_dim > d_model must raise ValueError."""
    with pytest.raises(ValueError, match="type_embed_dim"):
        EncoderConfig(d_model=16, type_embed_dim=32)


# ── SDEConfig.from_crn ────────────────────────────────────────────────────────


def test_sde_config_from_crn_birth_death_sets_two_noise_channels():
    """from_crn sets n_noise_channels = 2 for the 2-reaction birth-death CRN."""
    config = SDEConfig.from_crn(birth_death())
    assert config.n_noise_channels == 2


def test_sde_config_from_crn_lotka_volterra_sets_three_noise_channels():
    """from_crn sets n_noise_channels = 3 for the 3-reaction Lotka-Volterra CRN."""
    config = SDEConfig.from_crn(lotka_volterra())
    assert config.n_noise_channels == 3


def test_sde_config_from_crn_matches_crn_n_reactions():
    """from_crn always sets n_noise_channels == crn.n_reactions."""
    for crn in (birth_death(), lotka_volterra(), simple_mapk_cascade()):
        config = SDEConfig.from_crn(crn)
        assert config.n_noise_channels == crn.n_reactions, (
            f"Expected n_noise_channels={crn.n_reactions}, got {config.n_noise_channels}"
        )


def test_sde_config_from_crn_forwards_other_kwargs():
    """from_crn passes d_model, d_hidden, and clip_state through correctly."""
    config = SDEConfig.from_crn(
        birth_death(), d_model=32, d_hidden=64, clip_state=False
    )
    assert config.d_model == 32
    assert config.d_hidden == 64
    assert config.clip_state is False
