"""Tests for Trainer checkpoint save/load and training resume.

Covers:
- Checkpoint contains optimizer_state, scheduler_state, best_val_loss, epoch.
- load_checkpoint restores optimizer state to match the saved state.
- Training resumes from checkpoint epoch + 1.
- start_epoch skips earlier epochs (result length matches epochs actually run).
"""

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

# ── Shared setup ──────────────────────────────────────────────────────────────


def _small_model():
    """Tiny model (d_model=8) for fast unit tests."""
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
    """Build a tiny dataset with n_items CRN instances."""
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, T)
    init = torch.tensor([5.0])
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


def _make_trainer(crn, model_config, tmp_path, max_epochs: int = 2, val_every: int = 1):
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = NeuralSDE(model_config.sde, n_species=1)
    solver = EulerMaruyamaSolver(SolverConfig())
    config = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=2,
        n_sde_samples=2,
        val_every=val_every,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    return Trainer(encoder, sde, model_config, config, simulator=solver), encoder, sde


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_checkpoint_contains_optimizer_state(tmp_path):
    """Checkpoint includes optimizer_state, scheduler_state, best_val_loss, epoch."""
    encoder, sde, solver, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    dataset = _make_dataset(crn)
    trainer.train(dataset, val_dataset=_make_dataset(crn, n_items=2))

    ckpt_files = sorted((tmp_path / "ckpt").glob("best_epoch*.pt"))
    assert ckpt_files, "Expected at least one checkpoint file"

    ckpt = torch.load(ckpt_files[-1], weights_only=False)
    assert "optimizer_state" in ckpt
    assert "scheduler_state" in ckpt
    assert "best_val_loss" in ckpt
    assert "epoch" in ckpt
    assert "encoder_state" in ckpt
    assert "model_state" in ckpt


def test_load_checkpoint_restores_optimizer(tmp_path):
    """After load_checkpoint, optimizer state_dict matches the saved state."""
    encoder, sde, solver, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    dataset = _make_dataset(crn)
    trainer.train(dataset, val_dataset=_make_dataset(crn, n_items=2))

    ckpt_files = sorted((tmp_path / "ckpt").glob("best_epoch*.pt"))
    assert ckpt_files
    ckpt = torch.load(ckpt_files[-1], weights_only=False)
    saved_opt_state = ckpt["optimizer_state"]

    # Create a fresh trainer and load the checkpoint
    tmp2 = tmp_path / "ckpt2"
    config2 = TrainingConfig(
        max_epochs=5,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs2"),
        checkpoint_dir=str(tmp2),
        scheduler_type=SchedulerType.COSINE,
    )
    encoder2 = BipartiteGNNEncoder(model_config.encoder)
    sde2 = NeuralSDE(model_config.sde, n_species=1)
    solver2 = EulerMaruyamaSolver(SolverConfig())
    trainer2 = Trainer(encoder2, sde2, model_config, config2, simulator=solver2)
    trainer2.load_checkpoint(ckpt)

    # Verify optimizer state matches
    restored = trainer2._optimizer.state_dict()
    # Compare param_groups lr
    assert restored["param_groups"][0]["lr"] == saved_opt_state["param_groups"][0]["lr"]


def test_resume_continues_from_correct_epoch(tmp_path):
    """Training resumes from checkpoint epoch + 1 and result has correct length."""
    encoder, sde, solver, model_config, crn = _small_model()
    # Train 2 epochs to create a checkpoint
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    dataset = _make_dataset(crn)
    trainer.train(dataset, val_dataset=_make_dataset(crn, n_items=2))

    ckpt_files = sorted((tmp_path / "ckpt").glob("best_epoch*.pt"))
    assert ckpt_files
    ckpt = torch.load(ckpt_files[-1], weights_only=False)

    # Resume from checkpoint in a new trainer with max_epochs=4
    config2 = TrainingConfig(
        max_epochs=4,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs2"),
        checkpoint_dir=str(tmp_path / "ckpt2"),
        scheduler_type=SchedulerType.COSINE,
    )
    encoder2 = BipartiteGNNEncoder(model_config.encoder)
    sde2 = NeuralSDE(model_config.sde, n_species=1)
    solver2 = EulerMaruyamaSolver(SolverConfig())
    trainer2 = Trainer(encoder2, sde2, model_config, config2, simulator=solver2)
    start_epoch = trainer2.load_checkpoint(ckpt)

    result = trainer2.train(
        dataset, val_dataset=_make_dataset(crn, n_items=2), start_epoch=start_epoch
    )

    # Resumed from epoch N, so trained epochs N+1 through max_epochs=4
    # ckpt was at epoch <= 2; resumed from epoch ckpt["epoch"] + 1
    expected_n_train = config2.max_epochs - ckpt["epoch"]
    assert len(result.train_losses) == expected_n_train


def test_train_start_epoch_skips_earlier_epochs(tmp_path):
    """start_epoch=3 with max_epochs=5 runs exactly 3 epochs (3, 4, 5)."""
    encoder, sde, solver, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=5,
        batch_size=2,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config, simulator=solver)
    result = trainer.train(_make_dataset(crn), start_epoch=3)
    assert len(result.train_losses) == 3
