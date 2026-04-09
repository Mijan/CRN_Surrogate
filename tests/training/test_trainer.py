"""Smoke tests for Trainer construction and single-batch mechanics."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.measurement.config import MeasurementConfig
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.simulator.state_transform import get_state_transform
from crn_surrogate.training.trainer import Trainer


def _small_model_config(n_noise_channels: int = 2) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=1, use_attention=False),
        sde=SDEConfig(
            d_model=16,
            d_hidden=32,
            n_noise_channels=n_noise_channels,
            n_hidden_layers=1,
        ),
        measurement=MeasurementConfig(),
    )


def _small_train_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        lr=1e-3,
        max_epochs=1,
        batch_size=2,
        n_ssa_samples=2,
        dt=0.1,
        val_every=1,
        scheduler_type=SchedulerType.COSINE,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )


def _build_stochastic(tmp_path):
    model_config = _small_model_config(n_noise_channels=2)
    encoder = BipartiteGNNEncoder(model_config.encoder)
    model = NeuralSDE(model_config.sde, n_species=2)
    train_config = _small_train_config(tmp_path)
    solver = EulerMaruyamaSolver(
        SolverConfig(), state_transform=get_state_transform(False)
    )
    return encoder, model, model_config, train_config, solver


def _build_deterministic(tmp_path):
    model_config = _small_model_config(n_noise_channels=2)
    encoder = BipartiteGNNEncoder(model_config.encoder)
    model = NeuralDrift(model_config.sde, n_species=2)
    train_config = _small_train_config(tmp_path)
    solver = EulerODESolver(SolverConfig(), state_transform=get_state_transform(False))
    return encoder, model, model_config, train_config, solver


# ── Construction ──────────────────────────────────────────────────────────────


def test_trainer_constructs(tmp_path) -> None:
    encoder, model, model_config, train_config, solver = _build_stochastic(tmp_path)
    Trainer(encoder, model, model_config, train_config, solver)


def test_trainer_constructs_deterministic(tmp_path) -> None:
    encoder, model, model_config, train_config, solver = _build_deterministic(tmp_path)
    Trainer(encoder, model, model_config, train_config, solver)


# ── Single-batch loss ─────────────────────────────────────────────────────────


def _make_two_species_dataset() -> CRNTrajectoryDataset:
    crn = CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1.0, -1.0]),
                propensity=mass_action(0.5, torch.tensor([0.0, 1.0])),
                name="b_to_a",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 1.0]),
                propensity=mass_action(0.3, torch.tensor([1.0, 0.0])),
                name="a_to_b",
            ),
        ]
    )
    crn_repr = crn_to_tensor_repr(crn)
    items = [
        TrajectoryItem(
            crn_repr=crn_repr,
            initial_state=torch.ones(2) * 5.0,
            trajectories=torch.rand(2, 10, 2) * 10 + 1,
            times=torch.linspace(0.0, 1.0, 10),
        )
        for _ in range(2)
    ]
    return CRNTrajectoryDataset(items)


def test_single_batch_loss_finite(tmp_path) -> None:
    encoder, model, model_config, train_config, solver = _build_stochastic(tmp_path)
    trainer = Trainer(encoder, model, model_config, train_config, solver)
    dataset = _make_two_species_dataset()
    result = trainer.train(dataset)
    assert len(result.train_losses) == 1
    assert torch.isfinite(torch.tensor(result.train_losses[0]))
