"""End-to-end integration tests: CRN → encoder → SDE → trajectory, gradients, training.

These tests exercise complete pipelines across all subsystems:
- CRN definition → encoder → SDE forward pass → trajectory shape and properties.
- Loss computation on the solved trajectory with gradient flow back to parameters.
- Trainer.train() runs without error and produces a valid TrainingResult.
"""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.gillespie import (
    GillespieSSA,
    birth_death_crn,
    interpolate_to_grid,
    lotka_volterra_crn,
)
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.losses import MeanMatchingLoss
from crn_surrogate.training.trainer import Trainer

# ── Forward pass shapes ───────────────────────────────────────────────────────


def test_end_to_end_birth_death_trajectory_shape_and_metadata():
    """Full forward pass for birth-death: verify trajectory shape and Trajectory attributes."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=2),
        sde=SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=1)
    solver = EulerMaruyamaSolver(config.sde)
    crn = birth_death_crn()
    t_span = torch.linspace(0, 5, 10)

    ctx = encoder(crn, torch.tensor([10.0]))
    traj = solver.solve(sde, torch.tensor([10.0]), ctx, t_span, dt=0.1)

    assert traj.states.shape == (10, 1)
    assert traj.n_steps == 10
    assert traj.n_species == 1


def test_end_to_end_lotka_volterra_trajectory_shape():
    """Full forward pass for Lotka-Volterra (2 species): trajectory must be (T, 2)."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=32, n_layers=2),
        sde=SDEConfig(d_model=32, d_hidden=64, n_noise_channels=8),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=2)
    solver = EulerMaruyamaSolver(config.sde)
    t_span = torch.linspace(0, 10, 20)

    ctx = encoder(lotka_volterra_crn(), torch.tensor([50.0, 20.0]))
    traj = solver.solve(sde, torch.tensor([50.0, 20.0]), ctx, t_span, dt=0.1)

    assert traj.states.shape == (20, 2)


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_gradient_flows_from_loss_through_sde_to_encoder():
    """Gradients from the mean-matching loss must reach both SDE and encoder parameters,
    confirming that the full computation graph is connected end-to-end."""
    config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=1),
        sde=SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4),
    )
    encoder = BipartiteGNNEncoder(config.encoder)
    sde = CRNNeuralSDE(config.sde, n_species=1)
    solver = EulerMaruyamaSolver(config.sde)
    crn = birth_death_crn()
    t_span = torch.linspace(0, 2, 5)

    ctx = encoder(crn, torch.tensor([10.0]))
    traj = solver.solve(sde, torch.tensor([10.0]), ctx, t_span, dt=0.1)

    pred_states = traj.states.unsqueeze(0)  # (K=1, T, n_species)
    true_states = torch.ones(1, 5, 1) * 10.0  # (M=1, T, n_species)
    MeanMatchingLoss().compute(pred_states, true_states).backward()

    sde_grads_present = any(p.grad is not None for p in sde.parameters())
    enc_grads_present = any(p.grad is not None for p in encoder.parameters())
    assert sde_grads_present, "No gradients reached SDE parameters"
    assert enc_grads_present, "No gradients reached encoder parameters"


# ── Trainer integration ───────────────────────────────────────────────────────


def _tiny_dataset(
    crn, n_items: int = 4, M: int = 4, T: int = 8
) -> CRNTrajectoryDataset:
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, T)
    init = torch.zeros(crn.n_species)
    items = []
    for _ in range(n_items):
        trajs = torch.stack(
            [
                interpolate_to_grid(
                    ssa.simulate(crn, init.clone(), t_max=5.0).times,
                    ssa.simulate(crn, init.clone(), t_max=5.0).states,
                    time_grid,
                )
                for _ in range(M)
            ]
        )
        items.append(
            TrajectoryItem(
                crn=crn, initial_state=init.clone(), trajectories=trajs, times=time_grid
            )
        )
    return CRNTrajectoryDataset(items)


def test_trainer_completes_and_returns_valid_training_result(tmp_path):
    """Trainer.train() must complete without error and return a TrainingResult
    whose train_losses are all finite and whose val_epochs match val_every."""
    crn = birth_death_crn(k1=2.0, k2=0.5)
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=8, n_layers=1),
        sde=SDEConfig.from_crn(crn, d_model=8, d_hidden=16),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    sde = CRNNeuralSDE(model_config.sde, n_species=1)
    train_config = TrainingConfig(
        max_epochs=4,
        batch_size=2,
        n_sde_samples=2,
        val_every=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )

    result = Trainer(encoder, sde, model_config, train_config).train(
        _tiny_dataset(crn), val_dataset=_tiny_dataset(crn, n_items=2)
    )

    assert len(result.train_losses) == 4
    assert result.val_epochs == [2, 4]
    assert all(loss == loss and loss < float("inf") for loss in result.train_losses), (
        "NaN or Inf detected in train losses"
    )
