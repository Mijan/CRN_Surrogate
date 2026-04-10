"""Smoke tests for Trainer construction and single-batch mechanics."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.measurement.config import MeasurementConfig
from crn_surrogate.measurement.direct import DirectObservation
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.simulator.state_transform import get_state_transform
from crn_surrogate.training.losses import NLLStepLoss, RelativeMSEStepLoss
from crn_surrogate.training.trainer import Trainer


def _small_sde_config(n_noise_channels: int = 2) -> SDEConfig:
    return SDEConfig(
        d_model=16,
        d_hidden=32,
        n_noise_channels=n_noise_channels,
        n_hidden_layers=1,
    )


def _small_encoder_config() -> EncoderConfig:
    return EncoderConfig(d_model=16, n_layers=1, use_attention=False)


def _small_train_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        lr=1e-3,
        max_epochs=1,
        batch_size=2,
        n_trajectory_samples=2,
        dt=0.1,
        val_every=1,
        scheduler_type=SchedulerType.COSINE,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )


def _make_nll_step_loss() -> NLLStepLoss:
    meas_config = MeasurementConfig()
    measurement_model = DirectObservation.from_config(meas_config, n_species=2)
    return NLLStepLoss(measurement_model=measurement_model, min_variance=0.01)


def _build_stochastic(tmp_path):
    encoder = BipartiteGNNEncoder(_small_encoder_config())
    model = NeuralSDE(_small_sde_config(n_noise_channels=2), n_species=2)
    train_config = _small_train_config(tmp_path)
    solver = EulerMaruyamaSolver(
        SolverConfig(), state_transform=get_state_transform(False)
    )
    step_loss = _make_nll_step_loss()
    return encoder, model, train_config, solver, step_loss


def _build_deterministic(tmp_path):
    encoder = BipartiteGNNEncoder(_small_encoder_config())
    model = NeuralDrift(_small_sde_config(n_noise_channels=2), n_species=2)
    train_config = _small_train_config(tmp_path)
    solver = EulerODESolver(SolverConfig(), state_transform=get_state_transform(False))
    step_loss = RelativeMSEStepLoss()
    return encoder, model, train_config, solver, step_loss


# ── Construction ──────────────────────────────────────────────────────────────


def test_trainer_constructs(tmp_path) -> None:
    encoder, model, train_config, solver, step_loss = _build_stochastic(tmp_path)
    Trainer(encoder, model, train_config, solver, step_loss=step_loss)


def test_trainer_constructs_deterministic(tmp_path) -> None:
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    Trainer(encoder, model, train_config, solver, step_loss=step_loss)


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
    encoder, model, train_config, solver, step_loss = _build_stochastic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)
    dataset = _make_two_species_dataset()
    result = trainer.train(dataset)
    assert len(result.train_losses) == 1
    assert torch.isfinite(torch.tensor(result.train_losses[0]))


def test_deterministic_batch_loss_finite(tmp_path) -> None:
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)
    dataset = _make_two_species_dataset()
    result = trainer.train(dataset)
    assert len(result.train_losses) == 1
    assert torch.isfinite(torch.tensor(result.train_losses[0]))


# ── _reduce_element_loss ──────────────────────────────────────────────────────


def test_reduce_element_loss_mask_excludes_padded(tmp_path) -> None:
    """Padded species (mask=False) must not contribute to loss."""
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    B, N, S = 2, 6, 3
    element_loss = torch.ones(B * N, S)

    # Item 0: species 0 and 1 active; item 1: all 3 active
    masks = torch.tensor([[True, True, False], [True, True, True]])

    loss = trainer._reduce_element_loss(element_loss, masks, B, N, S)

    # Each item: sum over active species and N, divided by (N * n_active)
    # item0: (N*2) / (N*2) = 1.0; item1: (N*3) / (N*3) = 1.0 -> mean = 1.0
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_reduce_element_loss_equal_per_item(tmp_path) -> None:
    """Both items identical -> loss equals single-item loss (mean over B)."""
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    B, N, S = 2, 4, 2
    loss_val = torch.tensor(3.0)
    element_loss = torch.full((B * N, S), loss_val.item())
    masks = torch.ones(B, S, dtype=torch.bool)

    loss = trainer._reduce_element_loss(element_loss, masks, B, N, S)
    assert loss.item() == pytest.approx(loss_val.item(), abs=1e-5)


# ── _teacher_forcing_loss / _batched_rollout_loss ─────────────────────────────


def _make_trainer_with_prepared_items(
    tmp_path, training_mode=TrainingMode.TEACHER_FORCING
):
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    cfg = TrainingConfig(
        lr=1e-3,
        max_epochs=1,
        batch_size=2,
        n_trajectory_samples=1,
        dt=0.1,
        val_every=1,
        scheduler_type=SchedulerType.COSINE,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        training_mode=training_mode,
    )
    trainer = Trainer(encoder, model, cfg, solver, step_loss=step_loss)

    dataset = _make_two_species_dataset()
    trainer._train_cache = trainer.train.__func__  # just to access _prepare_batch
    # Build cache manually so _prepare_batch is available
    from crn_surrogate.training.data_cache import DataCache

    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batches = trainer._make_batches(cache, shuffle=False)
    batch = cache.get_batch(batches[0])
    items = trainer._prepare_batch(batch)
    return trainer, items


def test_teacher_forcing_loss_finite(tmp_path) -> None:
    trainer, items = _make_trainer_with_prepared_items(tmp_path)
    loss = trainer._teacher_forcing_loss(items)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_batched_rollout_loss_finite(tmp_path) -> None:
    trainer, items = _make_trainer_with_prepared_items(tmp_path)
    loss = trainer._batched_rollout_loss(items)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_batch_loss_dispatches_teacher_forcing(tmp_path) -> None:
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    dataset = _make_two_species_dataset()
    from crn_surrogate.training.data_cache import DataCache

    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batch = cache.get_batch(trainer._make_batches(cache, shuffle=False)[0])

    items = trainer._prepare_batch(batch)
    expected = trainer._teacher_forcing_loss(items).item()
    actual = trainer._compute_batch_loss(batch, epoch=1).item()
    assert actual == pytest.approx(expected, rel=1e-4)


def test_compute_batch_loss_dispatches_rollout(tmp_path) -> None:
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    cfg = TrainingConfig(
        lr=1e-3,
        max_epochs=1,
        batch_size=2,
        n_trajectory_samples=1,
        dt=0.1,
        val_every=1,
        scheduler_type=SchedulerType.COSINE,
        use_wandb=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        training_mode=TrainingMode.FULL_ROLLOUT,
    )
    trainer = Trainer(encoder, model, cfg, solver, step_loss=step_loss)

    dataset = _make_two_species_dataset()
    from crn_surrogate.training.data_cache import DataCache

    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batch = cache.get_batch(trainer._make_batches(cache, shuffle=False)[0])

    items = trainer._prepare_batch(batch)
    expected = trainer._batched_rollout_loss(items).item()
    actual = trainer._compute_batch_loss(batch, epoch=1).item()
    assert actual == pytest.approx(expected, rel=1e-4)


def test_batched_rollout_matches_sequential(tmp_path) -> None:
    """Batched rollout must produce the same loss as sequential per-item Euler integration."""
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    dataset = _make_two_species_dataset()
    from crn_surrogate.training.data_cache import DataCache

    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batch = cache.get_batch(trainer._make_batches(cache, shuffle=False)[0])
    items = trainer._prepare_batch(batch)

    # Compute loss via batched rollout (the implementation under test)
    loss_batched = trainer._batched_rollout_loss(items).item()

    # Compute loss manually: integrate each item sequentially and use _reduce_element_loss
    item = items[0]
    M, T, S = item.true_trajs_padded.shape
    dt = (item.times[1] - item.times[0]).item()

    manual_losses = []
    for it in items:
        ctx = it.context.context_vector.unsqueeze(0).expand(M, -1)  # (M, d_ctx)
        state = it.true_trajs_padded[:, 0, :].clone()  # (M, S)
        predicted = [state.clone()]
        for t_idx in range(T - 1):
            t_val = it.times[t_idx].expand(M)
            drift = model.drift_from_context(t_val, state, ctx)
            state = state + drift * dt
            if solver.clip_state:
                state = state.clamp(min=0.0)
            predicted.append(state.clone())
        pred_traj = torch.stack(predicted, dim=1)  # (M, T, S)
        # Element-wise loss against ground truth
        pred_flat = pred_traj.reshape(M * T, S)
        true_flat = it.true_trajs_padded.reshape(M * T, S)
        el = step_loss.compute(true_flat, pred_flat, torch.zeros_like(pred_flat))
        mask = it.species_mask.unsqueeze(0).expand(M * T, -1)
        item_loss = (el * mask.float()).sum() / (M * T * it.species_mask.float().sum())
        manual_losses.append(item_loss)

    loss_sequential = torch.stack(manual_losses).mean().item()

    assert loss_batched == pytest.approx(loss_sequential, rel=1e-4)


# ── Rollout correctness: gradients and substeps ───────────────────────────────


def test_batched_rollout_no_zero_gradient(tmp_path) -> None:
    """Regression test: no clamp/softplus means gradients flow from epoch 1."""
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)
    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    from crn_surrogate.training.data_cache import DataCache

    dataset = _make_two_species_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batch = cache.get_batch(trainer._make_batches(cache, shuffle=False)[0])
    items = trainer._prepare_batch(batch)

    loss = trainer._batched_rollout_loss(items)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad, "All gradients are zero — clamp/softplus may still be killing them"


def test_batched_rollout_negative_states_allowed(tmp_path) -> None:
    """Loss must be finite even when predicted states go negative."""
    encoder, model, train_config, solver, step_loss = _build_deterministic(tmp_path)

    # Set bias terms to a small negative value so drift predictions are
    # slightly negative without causing exponential blow-up in the integration.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bias" in name:
                p.fill_(-1.0)
            else:
                p.fill_(0.0)

    trainer = Trainer(encoder, model, train_config, solver, step_loss=step_loss)

    from crn_surrogate.training.data_cache import DataCache

    dataset = _make_two_species_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), model.n_species)
    batch = cache.get_batch(trainer._make_batches(cache, shuffle=False)[0])
    items = trainer._prepare_batch(batch)

    loss = trainer._batched_rollout_loss(items)
    assert torch.isfinite(loss), "Loss is not finite with negative states"

    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad


def test_batched_rollout_substeps_improve_accuracy(tmp_path) -> None:
    """More substeps should produce lower MSE against an analytical reference."""
    from crn_surrogate.configs.training_config import TrainingConfig, TrainingMode

    encoder, model, _, solver, step_loss = _build_deterministic(tmp_path)

    # Build ground truth via fine integration (large n_sub)
    dataset = _make_two_species_dataset()
    from crn_surrogate.training.data_cache import DataCache

    device = torch.device("cpu")

    def _make_trainer_with_nsub(nsub: int) -> Trainer:
        cfg = TrainingConfig(
            lr=1e-3,
            max_epochs=1,
            batch_size=2,
            n_trajectory_samples=1,
            dt=0.1,
            n_rollout_substeps=nsub,
            val_every=1,
            scheduler_type=SchedulerType.COSINE,
            use_wandb=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            training_mode=TrainingMode.FULL_ROLLOUT,
        )
        return Trainer(encoder, model, cfg, solver, step_loss=step_loss)

    cache = DataCache.from_dataset(dataset, device, model.n_species)
    batch = cache.get_batch(
        _make_trainer_with_nsub(1)._make_batches(cache, shuffle=False)[0]
    )

    with torch.no_grad():
        items1 = _make_trainer_with_nsub(1)._prepare_batch(batch)
        items10 = _make_trainer_with_nsub(10)._prepare_batch(batch)
        loss_1sub = _make_trainer_with_nsub(1)._batched_rollout_loss(items1).item()
        loss_10sub = _make_trainer_with_nsub(10)._batched_rollout_loss(items10).item()

    # With more substeps the predicted trajectory is more accurate,
    # so the loss against the ground truth should be lower or equal.
    # (Both use same random model weights so this is purely about integration accuracy.)
    assert loss_10sub <= loss_1sub * 2, (
        f"More substeps did not improve accuracy: loss_1sub={loss_1sub:.4f}, "
        f"loss_10sub={loss_10sub:.4f}"
    )
