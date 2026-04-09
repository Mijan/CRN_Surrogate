"""Tests for periodic checkpointing in the Trainer.

Covers:
- Periodic checkpoints are saved at the configured interval.
- Periodic checkpoints are independent of validation performance.
- Only the last 3 periodic checkpoints are kept on disk.
- Periodic checkpoints contain full training state.
- Training can resume from a periodic checkpoint.
"""

import os

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.simulator.neural_sde import NeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.trainer import Trainer

# ── Shared setup ─────────────────────────────────────────────────────────────


def _small_model():
    crn = birth_death(k_birth=2.0, k_death=0.5)
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=8, n_layers=1),
        sde=SDEConfig.from_crn(crn, d_model=8, d_hidden=16),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = NeuralSDE(model_config.sde, n_species=1)
    solver = EulerMaruyamaSolver(SolverConfig())
    return encoder, sde, solver, model_config, crn


def _make_dataset(
    crn, n_items: int = 4, M: int = 4, T: int = 8
) -> CRNTrajectoryDataset:
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, T)
    init = torch.tensor([0.0])
    crn_repr = crn_to_tensor_repr(crn)
    items = []
    for _ in range(n_items):
        trajs = Trajectory.stack_on_grid(
            ssa.simulate_batch(
                stoichiometry=crn.stoichiometry_matrix,
                propensity_fn=crn.evaluate_propensities,
                initial_state=init.clone(),
                t_max=5.0,
                n_trajectories=M,
            ),
            time_grid,
        )
        items.append(
            TrajectoryItem(
                crn_repr=crn_repr,
                initial_state=init.clone(),
                trajectories=trajs,
                times=time_grid,
            )
        )
    return CRNTrajectoryDataset(items)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_periodic_checkpoint_saved_at_interval(tmp_path):
    """Periodic checkpoints are saved at the configured interval."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=9,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=3,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    trainer.train(_make_dataset(crn))

    periodic_files = {f for f in os.listdir(ckpt_dir) if f.startswith("periodic_epoch")}
    # Epochs 3, 6, 9 should have been saved (last 3 kept)
    assert (
        "periodic_epoch3.pt" in periodic_files or "periodic_epoch6.pt" in periodic_files
    )
    assert "periodic_epoch9.pt" in periodic_files


def test_periodic_checkpoint_independent_of_validation(tmp_path):
    """Periodic checkpoints are saved even without a validation dataset."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=6,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    # No val_dataset — periodic checkpoints must still be created
    trainer.train(_make_dataset(crn))

    periodic_files = [f for f in os.listdir(ckpt_dir) if f.startswith("periodic_epoch")]
    assert len(periodic_files) >= 1


def test_periodic_checkpoint_cleanup_keeps_last_three(tmp_path):
    """Only the last 3 periodic checkpoints are kept on disk."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=10,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    trainer.train(_make_dataset(crn))

    # Epochs 2, 4, 6, 8, 10 → 5 checkpoints generated → only last 3 kept
    periodic_files = [f for f in os.listdir(ckpt_dir) if f.startswith("periodic_epoch")]
    assert len(periodic_files) <= 3
    # The most recent one must be present
    assert "periodic_epoch10.pt" in periodic_files


def test_periodic_checkpoint_contains_full_state(tmp_path):
    """Periodic checkpoints contain optimizer, scheduler, and best_val_loss."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = tmp_path / "ckpt"
    config = TrainingConfig(
        max_epochs=4,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=4,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(ckpt_dir),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    trainer.train(_make_dataset(crn))

    ckpt_path = ckpt_dir / "periodic_epoch4.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, weights_only=False)
    for key in (
        "epoch",
        "encoder_state",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_val_loss",
        "train_loss",
    ):
        assert key in ckpt, f"Missing key: {key}"
    assert ckpt["epoch"] == 4


def test_periodic_checkpoint_disabled_by_default(tmp_path):
    """checkpoint_every=0 produces no periodic checkpoints."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=4,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=0,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    trainer.train(_make_dataset(crn))

    if os.path.exists(ckpt_dir):
        periodic_files = [
            f for f in os.listdir(ckpt_dir) if f.startswith("periodic_epoch")
        ]
        assert len(periodic_files) == 0


def test_resume_from_periodic_checkpoint(tmp_path):
    """Training can resume from a periodic checkpoint."""
    encoder, sde, solver, model_config, crn = _small_model()
    ckpt_dir = tmp_path / "ckpt"
    config = TrainingConfig(
        max_epochs=4,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(ckpt_dir),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    trainer.train(_make_dataset(crn))

    ckpt_path = ckpt_dir / "periodic_epoch2.pt"
    assert ckpt_path.exists()

    # Resume from epoch 2 and train 2 more epochs
    encoder2, sde2, solver2, model_config2, _ = _small_model()
    config2 = TrainingConfig(
        max_epochs=6,
        batch_size=4,
        n_sde_samples=2,
        checkpoint_every=0,
        log_dir=str(tmp_path / "logs2"),
        checkpoint_dir=str(tmp_path / "ckpt2"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer2 = Trainer(encoder2, sde2, model_config2, config2, simulator=solver2)
    ckpt = torch.load(ckpt_path, weights_only=False)
    start_epoch = trainer2.load_checkpoint(ckpt)
    assert start_epoch == 3  # epoch 2 + 1

    result = trainer2.train(_make_dataset(crn), start_epoch=start_epoch)
    assert len(result.train_losses) == 4  # epochs 3, 4, 5, 6
