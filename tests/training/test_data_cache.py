"""Tests for DataCache: pre-transfer GPU dataset cache."""

from __future__ import annotations

import pytest
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
from crn_surrogate.simulator.neural_sde import NeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.simulator.state_transform import get_state_transform
from crn_surrogate.training.data_cache import DataCache
from crn_surrogate.training.trainer import Trainer

N_ITEMS = 5
M_TRAJ = 3
T_STEPS = 8
N_SPECIES = 2
N_SPECIES_PAD = 3


def _make_crn() -> CRN:
    return CRN(
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


def _make_dataset(n: int = N_ITEMS) -> CRNTrajectoryDataset:
    crn = _make_crn()
    crn_repr = crn_to_tensor_repr(crn)
    items = [
        TrajectoryItem(
            crn_repr=crn_repr,
            initial_state=torch.ones(N_SPECIES) * float(i + 1),
            trajectories=torch.rand(M_TRAJ, T_STEPS, N_SPECIES) * 10 + float(i),
            times=torch.linspace(0.0, 1.0, T_STEPS),
        )
        for i in range(n)
    ]
    return CRNTrajectoryDataset(items)


def test_from_dataset_shapes() -> None:
    dataset = _make_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), N_SPECIES_PAD)

    assert cache.trajectories.shape == (N_ITEMS, M_TRAJ, T_STEPS, N_SPECIES_PAD)
    assert cache.times.shape == (N_ITEMS, T_STEPS)
    assert cache.init_states.shape == (N_ITEMS, N_SPECIES_PAD)
    assert cache.species_masks.shape == (N_ITEMS, N_SPECIES_PAD)
    assert len(cache.crn_reprs) == N_ITEMS
    assert len(cache.n_species_per_item) == N_ITEMS
    assert len(cache.n_reactions_per_item) == N_ITEMS


def test_get_batch_returns_correct_items() -> None:
    dataset = _make_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), N_SPECIES_PAD)

    indices = torch.tensor([0, 2, 4])
    batch = cache.get_batch(indices)

    assert batch["trajectories"].shape == (3, M_TRAJ, T_STEPS, N_SPECIES_PAD)
    assert batch["times"].shape == (3, T_STEPS)

    # Verify trajectories match original items (padded species dim)
    for batch_pos, item_idx in enumerate(indices.tolist()):
        item = dataset[item_idx]
        expected = torch.zeros(M_TRAJ, T_STEPS, N_SPECIES_PAD)
        expected[:, :, :N_SPECIES] = item.trajectories
        torch.testing.assert_close(batch["trajectories"][batch_pos], expected)


def test_shuffled_batches_cover_all_items() -> None:
    dataset = _make_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), N_SPECIES_PAD)

    # Build a minimal Trainer just for _make_batches
    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=1, use_attention=False),
        sde=SDEConfig(
            d_model=16, d_hidden=32, n_noise_channels=N_SPECIES, n_hidden_layers=1
        ),
        measurement=MeasurementConfig(),
    )
    encoder = BipartiteGNNEncoder(model_config.encoder)
    model = NeuralSDE(model_config.sde, n_species=N_SPECIES_PAD)
    train_config = TrainingConfig(
        batch_size=2,
        max_epochs=1,
        scheduler_type=SchedulerType.COSINE,
        use_wandb=False,
        checkpoint_dir="/tmp/ckpt",
        log_dir="/tmp/logs",
    )
    solver = EulerMaruyamaSolver(
        SolverConfig(), state_transform=get_state_transform(False)
    )
    trainer = Trainer(encoder, model, model_config, train_config, solver)

    batches = trainer._make_batches(cache, shuffle=True)
    all_indices = torch.cat(batches)
    assert all_indices.sort().values.tolist() == list(range(N_ITEMS))


def test_species_mask_correct() -> None:
    dataset = _make_dataset()
    cache = DataCache.from_dataset(dataset, torch.device("cpu"), N_SPECIES_PAD)

    # Active species are the first N_SPECIES slots; padded slot should be False
    for i in range(N_ITEMS):
        assert cache.species_masks[i, :N_SPECIES].all()
        assert not cache.species_masks[i, N_SPECIES_PAD - 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_small_dataset_fits_on_gpu() -> None:
    dataset = _make_dataset()
    device = torch.device("cuda:0")
    cache = DataCache.from_dataset(
        dataset, device, N_SPECIES_PAD, gpu_memory_fraction=0.5
    )

    assert cache.trajectories_on_gpu
    assert cache.trajectories.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cache_crn_reprs_on_correct_device() -> None:
    dataset = _make_dataset()
    device = torch.device("cuda:0")
    cache = DataCache.from_dataset(
        dataset, device, N_SPECIES_PAD, gpu_memory_fraction=0.5
    )

    for repr_ in cache.crn_reprs:
        assert repr_.stoichiometry.device.type == "cuda"
