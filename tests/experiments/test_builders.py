"""Tests for experiments/builders.py — Hydra config → library config translation."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crn_surrogate.measurement.config import NoiseMode, NoiseSharing
from crn_surrogate.simulation.data_simulator import ODESimulator, SSASimulator
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.losses import NLLStepLoss, RelativeMSEStepLoss
from experiments.builders import (
    build_data_simulator,
    build_dataset_generator_config,
    build_encoder_config,
    build_model,
    build_model_config,
    build_sde_config,
    build_simulator,
    build_step_loss,
    build_training_config,
    select_device,
)


def make_test_cfg(**overrides):
    base = {
        "model": {
            "d_model": 32,
            "n_encoder_layers": 2,
            "d_hidden": 64,
            "n_sde_hidden_layers": 1,
            "d_protocol": 0,
            "context_dropout": 0.0,
            "mlp_dropout": 0.0,
            "max_n_species": 3,
            "max_n_reactions": 4,
        },
        "training": {
            "lr": 1e-3,
            "max_epochs": 10,
            "batch_size": 4,
            "n_trajectory_samples": 2,
            "dt": 0.1,
            "grad_clip_norm": 1.0,
            "scheduler_type": "cosine",
            "training_mode": "teacher_forcing",
            "scheduled_sampling_start_epoch": 50,
            "scheduled_sampling_end_epoch": 200,
            "val_every": 5,
            "checkpoint_every": 0,
            "num_workers": 0,
            "shuffle_train": True,
            "gpu_memory_fraction": 0.5,
        },
        "solver": {
            "deterministic": False,
            "use_log1p": False,
            "clip_state": True,
        },
        "measurement": {
            "noise_mode": "learned",
            "noise_sharing": "shared",
            "noise_init_eps": 0.02,
            "min_variance": 0.01,
        },
        "dataset": {
            "n_train": 10,
            "n_val": 5,
            "n_trajectories": 2,
            "t_max": 10.0,
            "n_time_points": 20,
            "initial_state_mean": 10.0,
            "initial_state_spread": 3.0,
            "topology": {
                "n_species_range": [1, 3],
                "n_reactions_range": [2, 4],
                "max_reactant_order": 2,
                "max_product_count": 2,
            },
            "rate_constant_range": [0.01, 10.0],
        },
        "generation": {
            "output_dir": "/tmp/test_gen",
            "checkpoint_every": 0,
            "sim_timeout": 5,
            "n_init_conditions": 1,
            "use_fast_ssa": False,
            "use_ode_prescreen": False,
        },
        "experiment_name": "test",
        "wandb_project": "test",
        "wandb_group": "",
        "seed": 42,
        "device": "cpu",
        "no_wandb": True,
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ── Encoder config ────────────────────────────────────────────────────────────


def test_build_encoder_config() -> None:
    cfg = make_test_cfg()
    enc = build_encoder_config(cfg)
    assert enc.d_model == 32
    assert enc.n_layers == 2
    assert enc.use_attention is True


# ── SDE config ────────────────────────────────────────────────────────────────


def test_build_sde_config() -> None:
    cfg = make_test_cfg()
    sde = build_sde_config(cfg)
    assert sde.d_model == 32
    assert sde.d_hidden == 64
    assert sde.n_noise_channels == cfg.model.max_n_reactions


# ── Training config ───────────────────────────────────────────────────────────


def test_build_training_config() -> None:
    cfg = make_test_cfg()
    tc = build_training_config(cfg, use_wandb=False)
    assert tc.lr == pytest.approx(1e-3)
    assert tc.batch_size == 4
    from crn_surrogate.configs.training_config import SchedulerType

    assert tc.scheduler_type == SchedulerType.COSINE


# ── Measurement config ────────────────────────────────────────────────────────


def test_build_measurement_config() -> None:
    cfg = make_test_cfg()
    mc = build_model_config(cfg).measurement
    assert mc.noise.mode == NoiseMode.LEARNED


# ── Model ─────────────────────────────────────────────────────────────────────


def test_build_model_stochastic() -> None:
    cfg = make_test_cfg()
    model = build_model(cfg, torch.device("cpu"))
    assert isinstance(model, NeuralSDE)


def test_build_model_deterministic() -> None:
    cfg = make_test_cfg(
        solver={"deterministic": True, "use_log1p": False, "clip_state": True}
    )
    model = build_model(cfg, torch.device("cpu"))
    assert isinstance(model, NeuralDrift)
    assert not isinstance(model, NeuralSDE)


# ── Simulator ─────────────────────────────────────────────────────────────────


def test_build_simulator_stochastic() -> None:
    cfg = make_test_cfg()
    sim = build_simulator(cfg)
    assert isinstance(sim, EulerMaruyamaSolver)


def test_build_simulator_deterministic() -> None:
    cfg = make_test_cfg(
        solver={"deterministic": True, "use_log1p": False, "clip_state": True}
    )
    sim = build_simulator(cfg)
    assert isinstance(sim, EulerODESolver)


def test_build_simulator_log1p() -> None:
    cfg = make_test_cfg(
        solver={"deterministic": False, "use_log1p": True, "clip_state": True}
    )
    sim = build_simulator(cfg)
    # Log1pTransform: forward of 0 should be 0, forward of 1 should be log(2)
    x = torch.tensor([0.0, 1.0])
    transformed = sim._state_transform.forward(x)
    assert transformed[0].item() == pytest.approx(0.0, abs=1e-6)
    assert transformed[1].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item(), abs=1e-6
    )


# ── Data simulator ────────────────────────────────────────────────────────────


def test_build_data_simulator_deterministic() -> None:
    cfg = make_test_cfg(
        solver={"deterministic": True, "use_log1p": False, "clip_state": True}
    )
    sim = build_data_simulator(cfg)
    assert isinstance(sim, ODESimulator)


def test_build_data_simulator_stochastic() -> None:
    cfg = make_test_cfg()
    sim = build_data_simulator(cfg)
    assert isinstance(sim, SSASimulator)


# ── Dataset generator config ──────────────────────────────────────────────────


def test_build_dataset_generator_config() -> None:
    cfg = make_test_cfg()
    gen_cfg = build_dataset_generator_config(cfg)
    assert gen_cfg.topology.n_species_range == (1, 3)
    assert gen_cfg.topology.n_reactions_range == (2, 4)
    assert gen_cfg.rate_constant_range == (0.01, 10.0)


# ── Step loss ─────────────────────────────────────────────────────────────────


def test_build_step_loss_deterministic() -> None:
    cfg = make_test_cfg(
        solver={"deterministic": True, "use_log1p": False, "clip_state": True}
    )
    loss = build_step_loss(cfg, torch.device("cpu"))
    assert isinstance(loss, RelativeMSEStepLoss)


def test_build_step_loss_stochastic() -> None:
    cfg = make_test_cfg()
    loss = build_step_loss(cfg, torch.device("cpu"))
    assert isinstance(loss, NLLStepLoss)


def test_build_step_loss_stochastic_has_params() -> None:
    cfg = make_test_cfg()
    loss = build_step_loss(cfg, torch.device("cpu"))
    assert isinstance(loss, NLLStepLoss)
    assert len(loss.parameters()) > 0


# ── Device selection ──────────────────────────────────────────────────────────


def test_select_device_cpu() -> None:
    device = select_device("cpu")
    assert device == torch.device("cpu")


# ── LabeledEnum passthrough in build_training_config ─────────────────────────


def test_build_training_config_training_mode() -> None:
    cfg = OmegaConf.merge(
        make_test_cfg(), {"training": {"training_mode": "full_rollout"}}
    )
    tc = build_training_config(cfg, use_wandb=False)
    from crn_surrogate.configs.training_config import TrainingMode

    assert tc.training_mode == TrainingMode.FULL_ROLLOUT


def test_build_training_config_scheduled_sampling_fields() -> None:
    cfg = OmegaConf.merge(
        make_test_cfg(),
        {
            "training": {
                "training_mode": "scheduled_sampling",
                "scheduled_sampling_start_epoch": 10,
                "scheduled_sampling_end_epoch": 80,
            }
        },
    )
    tc = build_training_config(cfg, use_wandb=False)
    assert tc.scheduled_sampling_start_epoch == 10
    assert tc.scheduled_sampling_end_epoch == 80


def test_build_training_config_invalid_mode_raises() -> None:
    cfg = OmegaConf.merge(make_test_cfg(), {"training": {"training_mode": "bogus"}})
    with pytest.raises(ValueError, match="TrainingMode"):
        build_training_config(cfg, use_wandb=False)


# ── NoiseMode / NoiseSharing passthrough ─────────────────────────────────────


def test_build_measurement_config_noise_mode_fixed() -> None:
    cfg = OmegaConf.merge(make_test_cfg(), {"measurement": {"noise_mode": "fixed"}})
    mc = build_model_config(cfg).measurement
    assert mc.noise.mode == NoiseMode.FIXED


def test_build_measurement_config_noise_sharing_per_species() -> None:
    cfg = OmegaConf.merge(
        make_test_cfg(), {"measurement": {"noise_sharing": "per_species"}}
    )
    mc = build_model_config(cfg).measurement
    assert mc.noise.sharing == NoiseSharing.PER_SPECIES
