"""Tests for experiment configuration classes and registry.

Covers:
- MassAction3sV3Config builds valid encoder and SDE configs with dropout.
- Dropout values propagate correctly through build_encoder_config / build_sde_config.
- Dataset size fields match v3 spec (5000 train / 500 val).
- Registry resolves "mass_action_3s_v3" to MassAction3sV3Config.
- BaseExperimentConfig default dropout values are 0.0.
"""

from experiments.configs.base import BaseExperimentConfig
from experiments.configs.mass_action_3s import MassAction3sConfig, MassAction3sV3Config
from experiments.configs.registry import get_config

# ── MassAction3sV3Config ──────────────────────────────────────────────────────


def test_mass_action_3s_v3_experiment_name():
    cfg = MassAction3sV3Config()
    assert cfg.experiment_name == "mass_action_3s_v3"


def test_mass_action_3s_v3_encoder_config_has_context_dropout():
    cfg = MassAction3sV3Config()
    encoder_cfg = cfg.build_encoder_config()
    assert encoder_cfg.context_dropout == 0.1


def test_mass_action_3s_v3_sde_config_has_mlp_dropout():
    cfg = MassAction3sV3Config()
    sde_cfg = cfg.build_sde_config()
    assert sde_cfg.mlp_dropout == 0.1


def test_mass_action_3s_v3_dataset_n_train():
    cfg = MassAction3sV3Config()
    assert cfg.dataset.n_train == 50000


def test_mass_action_3s_v3_dataset_n_val():
    cfg = MassAction3sV3Config()
    assert cfg.dataset.n_val == 5000


def test_mass_action_3s_v3_val_every():
    cfg = MassAction3sV3Config()
    assert cfg.val_every == 5


# ── BaseExperimentConfig dropout defaults ──────────────────────────────────────


def test_base_config_default_context_dropout_is_zero():
    """Existing experiments are unaffected: default context_dropout is 0.0."""

    class _MinimalConfig(BaseExperimentConfig):
        experiment_name: str = "test"

    cfg = _MinimalConfig()
    assert cfg.context_dropout == 0.0
    assert cfg.build_encoder_config().context_dropout == 0.0


def test_base_config_default_mlp_dropout_is_zero():
    """Existing experiments are unaffected: default mlp_dropout is 0.0."""

    class _MinimalConfig(BaseExperimentConfig):
        experiment_name: str = "test"

    cfg = _MinimalConfig()
    assert cfg.mlp_dropout == 0.0
    assert cfg.build_sde_config().mlp_dropout == 0.0


def test_mass_action_3s_v2_config_zero_dropout():
    """v2 config must still produce zero dropout (not affected by v3 changes)."""
    cfg = MassAction3sConfig()
    assert cfg.build_encoder_config().context_dropout == 0.0
    assert cfg.build_sde_config().mlp_dropout == 0.0


# ── Registry ──────────────────────────────────────────────────────────────────


def test_registry_resolves_v3_config():
    cfg = get_config("mass_action_3s_v3")
    assert isinstance(cfg, MassAction3sV3Config)


def test_registry_still_resolves_v2_config():
    cfg = get_config("mass_action_3s")
    assert isinstance(cfg, MassAction3sConfig)
