"""Tests for Hydra-based experiment configuration and builder functions.

Covers:
- build_encoder_config produces correct dropout and d_model values.
- build_sde_config produces correct dropout and noise channel values.
- build_training_config maps scheduler_type strings correctly.
- build_model_config composes encoder and SDE configs correctly.
- YAML config files are valid and parseable by OmegaConf.
- Experiment preset YAMLs override the expected fields.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from crn_surrogate.configs.training_config import SchedulerType

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs"


def _load_preset(name: str | None):
    """Load a full config by composing base + experiment preset via OmegaConf merge.

    Simulates Hydra's defaults resolution: if the preset overrides a group
    (e.g. /model: large), the corresponding YAML is loaded first, then the
    preset's own field overrides are merged on top.
    """
    base = OmegaConf.load(CONFIGS_DIR / "config.yaml")
    base = OmegaConf.masked_copy(base, [k for k in base if k != "defaults"])

    # Start with default groups
    groups: dict[str, str] = {
        "model": "default",
        "training": "default",
        "dataset": "mass_action_3s",
        "solver": "stochastic",
        "measurement": "default",
    }

    if name is not None:
        preset_path = CONFIGS_DIR / "experiment" / f"{name}.yaml"
        preset_raw = OmegaConf.load(preset_path)
        # Parse defaults list to pick up group overrides (e.g. /model: large)
        if "defaults" in preset_raw:
            for entry in preset_raw.defaults:
                if isinstance(entry, str):
                    continue
                for group, variant in entry.items():
                    groups[group.lstrip("/")] = variant

    cfg = OmegaConf.merge(
        base,
        {"model": OmegaConf.load(CONFIGS_DIR / "model" / f"{groups['model']}.yaml")},
        {"training": OmegaConf.load(CONFIGS_DIR / "training" / f"{groups['training']}.yaml")},
        {"dataset": OmegaConf.load(CONFIGS_DIR / "dataset" / f"{groups['dataset']}.yaml")},
        {"solver": OmegaConf.load(CONFIGS_DIR / "solver" / f"{groups['solver']}.yaml")},
        {"measurement": OmegaConf.load(CONFIGS_DIR / "measurement" / f"{groups['measurement']}.yaml")},
    )

    if name is not None:
        preset = OmegaConf.masked_copy(preset_raw, [k for k in preset_raw if k != "defaults"])
        cfg = OmegaConf.merge(cfg, preset)

    return cfg


def _default_cfg():
    """Build a minimal DictConfig matching the default YAML structure."""
    return _load_preset(None)


# ── build_encoder_config ──────────────────────────────────────────────────────


def test_build_encoder_config_d_model():
    from experiments.builders import build_encoder_config

    cfg = _default_cfg()
    enc = build_encoder_config(cfg)
    assert enc.d_model == cfg.model.d_model


def test_build_encoder_config_default_zero_dropout():
    from experiments.builders import build_encoder_config

    cfg = _default_cfg()
    enc = build_encoder_config(cfg)
    assert enc.context_dropout == 0.0


def test_build_encoder_config_nonzero_dropout_propagates():
    from experiments.builders import build_encoder_config

    cfg = OmegaConf.merge(_default_cfg(), {"model": {"context_dropout": 0.1}})
    enc = build_encoder_config(cfg)
    assert enc.context_dropout == pytest.approx(0.1)


# ── build_sde_config ──────────────────────────────────────────────────────────


def test_build_sde_config_d_model():
    from experiments.builders import build_sde_config

    cfg = _default_cfg()
    sde = build_sde_config(cfg)
    assert sde.d_model == cfg.model.d_model


def test_build_sde_config_default_zero_mlp_dropout():
    from experiments.builders import build_sde_config

    cfg = _default_cfg()
    sde = build_sde_config(cfg)
    assert sde.mlp_dropout == 0.0


def test_build_sde_config_mlp_dropout_propagates():
    from experiments.builders import build_sde_config

    cfg = OmegaConf.merge(_default_cfg(), {"model": {"mlp_dropout": 0.2}})
    sde = build_sde_config(cfg)
    assert sde.mlp_dropout == pytest.approx(0.2)


def test_build_sde_config_n_noise_channels_from_max_n_reactions():
    from experiments.builders import build_sde_config

    cfg = _default_cfg()
    sde = build_sde_config(cfg)
    assert sde.n_noise_channels == cfg.model.max_n_reactions


# ── build_training_config ─────────────────────────────────────────────────────


def test_build_training_config_cosine_scheduler():
    from experiments.builders import build_training_config

    cfg = OmegaConf.merge(_default_cfg(), {"training": {"scheduler_type": "cosine"}})
    train = build_training_config(cfg, use_wandb=False)
    assert train.scheduler_type == SchedulerType.COSINE


def test_build_training_config_reduce_on_plateau_scheduler():
    from experiments.builders import build_training_config

    cfg = OmegaConf.merge(
        _default_cfg(), {"training": {"scheduler_type": "reduce_on_plateau"}}
    )
    train = build_training_config(cfg, use_wandb=False)
    assert train.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU


def test_build_training_config_lr_propagates():
    from experiments.builders import build_training_config

    cfg = OmegaConf.merge(_default_cfg(), {"training": {"lr": 5e-4}})
    train = build_training_config(cfg, use_wandb=False)
    assert train.lr == pytest.approx(5e-4)


# ── YAML validity ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rel_path",
    [
        "model/default.yaml",
        "model/large.yaml",
        "training/default.yaml",
        "training/long.yaml",
        "dataset/mass_action_3s.yaml",
        "dataset/mass_action_3s_large.yaml",
        "dataset/mass_action_3s_det.yaml",
        "solver/stochastic.yaml",
        "solver/deterministic.yaml",
        "measurement/default.yaml",
        "experiment/mass_action_3s_v5.yaml",
        "experiment/mass_action_3s_v7.yaml",
        "experiment/mass_action_3s_v8.yaml",
        "experiment/mass_action_3s_det.yaml",
    ],
)
def test_yaml_file_is_parseable(rel_path: str):
    path = CONFIGS_DIR / rel_path
    cfg = OmegaConf.load(path)
    assert cfg is not None


# ── Experiment preset overrides ───────────────────────────────────────────────


def test_mass_action_3s_v5_experiment_name():
    cfg = _load_preset("mass_action_3s_v5")
    assert cfg.experiment_name == "mass_action_3s_v5"


def test_mass_action_3s_v5_uses_large_model():
    cfg = _load_preset("mass_action_3s_v5")
    assert cfg.model.d_model == 128


def test_mass_action_3s_v5_dropout_overrides():
    cfg = _load_preset("mass_action_3s_v5")
    assert cfg.model.context_dropout == pytest.approx(0.1)
    assert cfg.model.mlp_dropout == pytest.approx(0.1)


def test_mass_action_3s_det_uses_deterministic_solver():
    cfg = _load_preset("mass_action_3s_det")
    assert cfg.solver.deterministic is True


def test_mass_action_3s_v8_uses_log1p():
    cfg = _load_preset("mass_action_3s_v8")
    assert cfg.solver.use_log1p is True


def test_mass_action_3s_v7_dataset_n_train():
    cfg = _load_preset("mass_action_3s_v7")
    assert cfg.dataset.n_train == 30000
