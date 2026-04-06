"""Configuration for the mass-action <=3 species experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

from crn_surrogate.data.generation.mass_action_generator import (
    MassActionGeneratorConfig,
    RandomTopologyConfig,
)
from experiments.configs.base import BaseExperimentConfig


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset generation parameters for the mass-action <=3 species experiment.

    Attributes:
        generator: Config for the random CRN generator.
        n_train: Number of training CRN instances.
        n_val: Number of validation CRN instances.
        n_ssa_trajectories: SSA trajectories per CRN for dataset generation.
        t_max: Simulation end time.
        n_time_points: Number of evenly-spaced time grid points.
        n_workers: Parallel workers for SSA simulation.
        initial_state_mean: Geometric mean of initial molecule counts.
        initial_state_spread: Geometric standard deviation for initial state sampling.
    """

    generator: MassActionGeneratorConfig = field(
        default_factory=lambda: MassActionGeneratorConfig(
            topology=RandomTopologyConfig(
                n_species_range=(1, 3),
                n_reactions_range=(2, 6),
                max_reactant_order=2,
                max_product_count=2,
            ),
            rate_constant_range=(0.01, 10.0),
        )
    )
    n_train: int = 500
    n_val: int = 100
    n_ssa_trajectories: int = 32
    t_max: float = 20.0
    n_time_points: int = 50
    n_workers: int = 8
    initial_state_mean: float = 10.0
    initial_state_spread: float = 3.0


@dataclass(frozen=True)
class MassAction3sConfig(BaseExperimentConfig):
    """Full experiment configuration for the mass-action <=3 species experiment."""

    experiment_name: str = "mass_action_3s_v2"
    wandb_group: str = "mass-action-3s"
    max_n_species: int = 3
    max_n_reactions: int = 6
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    d_model: int = 128
    d_hidden: int = 256
    n_sde_hidden_layers: int = 3

    batch_size: int = 64

@dataclass(frozen=True)
class MassAction3sV5Config(BaseExperimentConfig):
    """Mass-action 3s v4: state-independent encoder.

    Same hyperparameters as v4. The only change is architectural:
    the encoder no longer receives initial_state, producing a purely
    topological/kinetic CRN context. This ensures the same CRN always
    maps to the same context vector regardless of initial conditions.
    """

    experiment_name: str = "mass_action_3s_v5"
    wandb_group: str = "mass-action-3s"

    # Architecture (same as v3)
    max_n_species: int = 3
    max_n_reactions: int = 6
    d_model: int = 128
    n_encoder_layers: int = 3
    d_hidden: int = 256
    n_sde_hidden_layers: int = 3

    # Dropout (same as v3)
    context_dropout: float = 0.1
    mlp_dropout: float = 0.1

    # Training (same as v3 witch changed batch size)
    max_epochs: int = 50000
    batch_size: int = 512
    lr: float = 1e-3
    dt: float = 0.1
    val_every: int = 5
    n_ssa_samples: int = 16  # matches n_ssa_trajectories / n_init_conditions
    checkpoint_every: int = 10  # save every 10 epochs; keep last 3 on disk

    # Dataset (same as v3)
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            n_train=50000,
            n_val=5000,
            n_ssa_trajectories=64,
        )
    )

@dataclass(frozen=True)
class MassAction3sV4Config(BaseExperimentConfig):
    """Mass-action 3s v4: state-independent encoder.

    Same hyperparameters as v3. The only change is architectural:
    the encoder no longer receives initial_state, producing a purely
    topological/kinetic CRN context. This ensures the same CRN always
    maps to the same context vector regardless of initial conditions.
    """

    experiment_name: str = "mass_action_3s_v4"
    wandb_group: str = "mass-action-3s"

    # Architecture (same as v3)
    max_n_species: int = 3
    max_n_reactions: int = 6
    d_model: int = 128
    n_encoder_layers: int = 3
    d_hidden: int = 256
    n_sde_hidden_layers: int = 3

    # Dropout (same as v3)
    context_dropout: float = 0.1
    mlp_dropout: float = 0.1

    # Training (same as v3 witch changed batch size)
    max_epochs: int = 50000
    batch_size: int = 512
    lr: float = 1e-3
    dt: float = 0.1
    val_every: int = 5
    n_ssa_samples: int = 16  # matches n_ssa_trajectories / n_init_conditions
    checkpoint_every: int = 10  # save every 10 epochs; keep last 3 on disk

    # Dataset (same as v3)
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            n_train=50000,
            n_val=5000,
            n_ssa_trajectories=64,
        )
    )



@dataclass(frozen=True)
class MassAction3sV3Config(BaseExperimentConfig):
    """Mass-action 3s v3: larger dataset, dropout, multiple initial conditions."""

    experiment_name: str = "mass_action_3s_v3"
    wandb_group: str = "mass-action-3s"

    # Architecture (same as v2)
    max_n_species: int = 3
    max_n_reactions: int = 6
    d_model: int = 128
    n_encoder_layers: int = 3
    d_hidden: int = 256
    n_sde_hidden_layers: int = 3

    # Dropout
    context_dropout: float = 0.1
    mlp_dropout: float = 0.1

    # Training
    max_epochs: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    dt: float = 0.1
    val_every: int = 5
    n_ssa_samples: int = 64

    # Dataset (larger)
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            n_train=50000,
            n_val=5000,
            n_ssa_trajectories=64,
        )
    )
