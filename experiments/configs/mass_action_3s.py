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

    experiment_name: str = "mass_action_3s_v1"
    wandb_group: str = "mass-action-3s"
    max_n_species: int = 3
    max_n_reactions: int = 6
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# Backward-compatible alias
ExperimentConfig = MassAction3sConfig
