"""Tests for training speed optimizations.

Covers:
- _validate with compute_rollout=False returns 0.0 rollout loss and finite NLL.
- _validate with compute_rollout=True returns nonzero rollout loss and finite NLL.
- torch.compile produces identical encoder outputs to the uncompiled model.
"""

import numpy as np
import pytest
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

# ── Shared setup ──────────────────────────────────────────────────────────────


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


def _make_trainer(crn, model_config, tmp_path) -> Trainer:
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = CRNNeuralSDE(model_config.sde, n_species=1)
    config = TrainingConfig(
        max_epochs=1,
        batch_size=2,
        n_sde_samples=2,
        val_every=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    return Trainer(encoder, sde, model_config, config)


# ── _validate compute_rollout tests ──────────────────────────────────────────


def test_validate_without_rollout_returns_zero_rollout_loss(tmp_path):
    """compute_rollout=False returns 0.0 for rollout loss and a finite NLL."""
    _, _, model_config, crn = _small_model()
    trainer = _make_trainer(crn, model_config, tmp_path)
    val_dataset = _make_dataset(crn, n_items=2)

    val_loss, val_nll = trainer._validate(val_dataset, compute_rollout=False)

    assert val_loss == 0.0
    assert np.isfinite(val_nll)
    assert val_nll != 0.0


def test_validate_with_rollout_returns_nonzero_rollout_loss(tmp_path):
    """compute_rollout=True returns a nonzero finite rollout loss."""
    _, _, model_config, crn = _small_model()
    trainer = _make_trainer(crn, model_config, tmp_path)
    val_dataset = _make_dataset(crn, n_items=2)

    val_loss, val_nll = trainer._validate(val_dataset, compute_rollout=True)

    assert val_loss != 0.0
    assert np.isfinite(val_loss)
    assert np.isfinite(val_nll)


# ── torch.compile tests ───────────────────────────────────────────────────────


@pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile requires PyTorch 2.0+",
)
def test_compiled_encoder_same_output():
    """torch.compile produces identical encoder outputs to the uncompiled model."""
    crn = birth_death(k_birth=2.0, k_death=0.5)
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=8, n_layers=1),
        sde=SDEConfig.from_crn(crn, d_model=8, d_hidden=16),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    encoder.eval()

    crn_repr = crn_to_tensor_repr(crn)
    init_state = torch.tensor([5.0])

    with torch.no_grad():
        ctx_raw = encoder(crn_repr, init_state)

    encoder_c = torch.compile(encoder)
    with torch.no_grad():
        ctx_compiled = encoder_c(crn_repr, init_state)

    torch.testing.assert_close(
        ctx_raw.context_vector,
        ctx_compiled.context_vector,
        atol=1e-5,
        rtol=1e-5,
    )
