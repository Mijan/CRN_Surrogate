"""Tests for the Trainer training loop.

Uses a tiny birth-death model (d_model=8, 2 epochs, 4 training items) so the
tests run fast while still exercising the real code paths.

Covers:
- train() returns a TrainingResult with the correct number of epoch entries.
- Validation losses are recorded at the correct epoch intervals.
- A checkpoint file is saved when validation loss improves.
- Profiler CSV files are written to the configured log directory after training.
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
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.training.trainer import Trainer

# ── Shared setup ─────────────────────────────────────────────────────────────


def _small_model():
    """Tiny model (d_model=8) for fast unit tests."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=8, n_layers=1),
        sde=SDEConfig.from_crn(crn, d_model=8, d_hidden=16),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = CRNNeuralSDE(model_config.sde, n_species=1)
    return encoder, sde, model_config, crn


def _make_dataset(
    crn, n_items: int = 4, M: int = 4, T: int = 8
) -> CRNTrajectoryDataset:
    """Build a tiny dataset: n_items CRN instances, each with M SSA trajectories."""
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


# ── TrainingResult structure ──────────────────────────────────────────────────


def test_trainer_train_losses_has_one_entry_per_epoch(tmp_path):
    """train_losses must contain exactly max_epochs entries after training."""
    encoder, sde, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=3,
        batch_size=2,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    result = trainer.train(_make_dataset(crn))
    assert len(result.train_losses) == 3


def test_trainer_val_losses_recorded_at_correct_epoch_intervals(tmp_path):
    """val_losses and val_epochs must be recorded exactly at multiples of val_every."""
    encoder, sde, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=4,
        batch_size=2,
        n_sde_samples=2,
        val_every=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    result = trainer.train(
        _make_dataset(crn), val_dataset=_make_dataset(crn, n_items=2)
    )
    assert result.val_epochs == [2, 4]
    assert len(result.val_losses) == 2


def test_trainer_no_val_losses_when_no_val_dataset(tmp_path):
    """When no validation dataset is provided, val_losses and val_epochs stay empty."""
    encoder, sde, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    result = trainer.train(_make_dataset(crn))
    assert result.val_losses == []
    assert result.val_epochs == []


def test_trainer_all_train_losses_are_finite(tmp_path):
    """Every recorded train loss must be a finite number (no NaN/Inf)."""
    encoder, sde, model_config, crn = _small_model()
    config = TrainingConfig(
        max_epochs=3,
        batch_size=2,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    result = trainer.train(_make_dataset(crn))
    for epoch, loss in enumerate(result.train_losses, start=1):
        assert loss == loss and loss < float("inf"), (
            f"Epoch {epoch} train loss is not finite: {loss}"
        )


# ── Checkpointing ─────────────────────────────────────────────────────────────


def test_trainer_saves_checkpoint_when_val_loss_improves(tmp_path):
    """A .pt checkpoint file must appear under checkpoint_dir after validation."""
    encoder, sde, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    trainer.train(_make_dataset(crn), val_dataset=_make_dataset(crn, n_items=2))
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    assert len(checkpoints) >= 1


def test_trainer_checkpoint_contains_expected_keys(tmp_path):
    """The saved checkpoint dict must contain encoder_state, sde_state, val_loss, epoch."""
    encoder, sde, model_config, crn = _small_model()
    ckpt_dir = str(tmp_path / "ckpt")
    config = TrainingConfig(
        max_epochs=1,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=ckpt_dir,
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    trainer.train(_make_dataset(crn), val_dataset=_make_dataset(crn, n_items=2))
    pt_file = next(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    ckpt = torch.load(os.path.join(ckpt_dir, pt_file), weights_only=False)
    assert {"epoch", "encoder_state", "sde_state", "val_loss"} == set(ckpt.keys())


# ── Profiler CSV integration ──────────────────────────────────────────────────


def test_trainer_writes_profiler_epoch_csv(tmp_path):
    """Training must produce profiler_epochs.csv in the configured log directory."""
    encoder, sde, model_config, crn = _small_model()
    log_dir = str(tmp_path / "logs")
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        log_dir=log_dir,
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    trainer.train(_make_dataset(crn))
    assert os.path.exists(os.path.join(log_dir, "profiler_epochs.csv"))


def test_trainer_writes_profiler_batch_csv(tmp_path):
    """Training must produce profiler_batches.csv in the configured log directory."""
    encoder, sde, model_config, crn = _small_model()
    log_dir = str(tmp_path / "logs")
    config = TrainingConfig(
        max_epochs=2,
        batch_size=2,
        n_sde_samples=2,
        log_dir=log_dir,
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    trainer.train(_make_dataset(crn))
    assert os.path.exists(os.path.join(log_dir, "profiler_batches.csv"))


def test_batch_to_device_moves_tensors_and_passes_through_non_tensors(tmp_path):
    """_batch_to_device moves all tensor values and leaves non-tensors unchanged."""
    encoder, sde, model_config, _ = _small_model()
    config = TrainingConfig(
        max_epochs=1,
        batch_size=2,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    batch = {
        "stoichiometry": torch.randn(2, 3, 1),
        "species_mask": torch.ones(2, 1, dtype=torch.bool),
        "some_string": "not_a_tensor",
    }
    moved = trainer._batch_to_device(batch)
    assert moved["stoichiometry"].device == trainer._device
    assert moved["species_mask"].device == trainer._device
    assert moved["some_string"] == "not_a_tensor"


def test_trainer_profiler_batch_csv_contains_forward_and_backward_columns(tmp_path):
    """Each row in profiler_batches.csv must have forward and backward timing columns."""
    import csv as csv_module

    encoder, sde, model_config, crn = _small_model()
    log_dir = str(tmp_path / "logs")
    config = TrainingConfig(
        max_epochs=1,
        batch_size=2,
        n_sde_samples=2,
        log_dir=log_dir,
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    trainer.train(_make_dataset(crn))

    with open(os.path.join(log_dir, "profiler_batches.csv")) as f:
        header = next(csv_module.reader(f))
    assert "forward" in header
    assert "backward" in header


# ── Variable-topology ─────────────────────────────────────────────────────────


def test_trainer_variable_topology(tmp_path):
    """Trainer handles a dataset with CRNs of different n_species without error."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import constant_rate, mass_action
    from crn_surrogate.crn.reaction import Reaction

    # 1-species CRN: birth + death
    rxns_1s = [
        Reaction(stoichiometry=torch.tensor([1.0]), propensity=constant_rate(k=1.0)),
        Reaction(
            stoichiometry=torch.tensor([-1.0]),
            propensity=mass_action(0.5, torch.tensor([1.0])),
        ),
    ]
    crn_1s = CRN(rxns_1s)

    # 2-species CRN: S0 produced, S0 degrades, S0->S1, S1 degrades
    rxns_2s = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0]), propensity=constant_rate(k=1.0)
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]),
            propensity=mass_action(0.5, torch.tensor([1.0, 0.0])),
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 1.0]),
            propensity=mass_action(0.3, torch.tensor([1.0, 0.0])),
        ),
        Reaction(
            stoichiometry=torch.tensor([0.0, -1.0]),
            propensity=mass_action(0.4, torch.tensor([0.0, 1.0])),
        ),
    ]
    crn_2s = CRN(rxns_2s)

    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, 8)

    def _make_item(crn, init):
        crn_repr = crn_to_tensor_repr(crn)
        trajs = Trajectory.stack_on_grid(
            ssa.simulate_batch(
                stoichiometry=crn.stoichiometry_matrix,
                propensity_fn=crn.evaluate_propensities,
                initial_state=init.clone(),
                t_max=5.0,
                n_trajectories=4,
            ),
            time_grid,
        )
        return TrajectoryItem(
            crn_repr=crn_repr, initial_state=init, trajectories=trajs, times=time_grid
        )

    items = [_make_item(crn_1s, torch.tensor([5.0])) for _ in range(2)]
    items += [_make_item(crn_2s, torch.tensor([5.0, 3.0])) for _ in range(2)]

    train_dataset = CRNTrajectoryDataset(items)

    # SDE configured for max 2 species
    max_n_species = 2
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=8, n_layers=1),
        sde=SDEConfig(d_model=8, d_hidden=16, n_noise_channels=4),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = CRNNeuralSDE(model_config.sde, n_species=max_n_species)

    config = TrainingConfig(
        max_epochs=1,
        batch_size=4,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)
    result = trainer.train(train_dataset)

    assert len(result.train_losses) == 1
    assert result.train_losses[0] == result.train_losses[0]  # not NaN
    assert result.train_losses[0] < float("inf")
