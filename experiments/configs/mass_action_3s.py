"""Configuration for the mass-action <=3 species experiment."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig, TrainingMode
from crn_surrogate.data.generation.mass_action_generator import (
    MassActionGeneratorConfig,
    RandomTopologyConfig,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset generation parameters.

    Attributes:
        generator: Config for the random CRN generator.
        n_train: Number of training CRN instances.
        n_val: Number of validation CRN instances.
        n_ssa_trajectories: SSA trajectories per CRN.
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
class ExperimentConfig:
    """Full experiment configuration for the mass-action <=3 species experiment.

    Attributes:
        wandb_project: W&B project name.
        wandb_group: W&B run group.
        experiment_name: Identifier used in artifact and file names.
        dataset: Dataset generation parameters.
        max_n_species: Maximum species count (SDE state dimension).
        max_n_reactions: Maximum reaction count (SDE noise channels).
        d_model: Hidden dimension for all node embeddings.
        n_encoder_layers: Number of bipartite message-passing rounds.
        d_hidden: Hidden dimension inside drift/diffusion MLPs.
        n_sde_hidden_layers: FiLM-conditioned hidden layers per network.
        max_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        dt: Euler-Maruyama step size.
        grad_clip_norm: Gradient clipping threshold.
        scheduler_type: LR scheduler ("cosine" or "reduce_on_plateau").
        val_every: Validation frequency in epochs.
    """

    # Identifiers
    wandb_project: str = "crn-surrogate"
    wandb_group: str = "mass-action-3s"
    experiment_name: str = "mass_action_3s_v1"

    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Architecture
    max_n_species: int = 3
    max_n_reactions: int = 6
    d_model: int = 64
    n_encoder_layers: int = 3
    d_hidden: int = 128
    n_sde_hidden_layers: int = 2

    # Training
    max_epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    dt: float = 0.1
    grad_clip_norm: float = 1.0
    scheduler_type: str = "cosine"
    val_every: int = 10

    def build_encoder_config(self) -> EncoderConfig:
        """Build encoder configuration from experiment settings."""
        return EncoderConfig(
            d_model=self.d_model,
            n_layers=self.n_encoder_layers,
            use_attention=True,
        )

    def build_sde_config(self) -> SDEConfig:
        """Build SDE config sized for the maximum topology in this experiment."""
        return SDEConfig(
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            n_noise_channels=self.max_n_reactions,
            n_hidden_layers=self.n_sde_hidden_layers,
            clip_state=True,
            d_protocol=0,
        )

    def build_model_config(self) -> ModelConfig:
        """Build the full model config from experiment settings."""
        return ModelConfig(
            encoder=self.build_encoder_config(),
            sde=self.build_sde_config(),
        )

    def build_training_config(self, *, use_wandb: bool = True) -> TrainingConfig:
        """Build training configuration from experiment settings.

        Args:
            use_wandb: Whether to enable W&B logging.

        Returns:
            TrainingConfig with all fields set from this experiment config.
        """
        sched = (
            SchedulerType.COSINE
            if self.scheduler_type == "cosine"
            else SchedulerType.REDUCE_ON_PLATEAU
        )
        return TrainingConfig(
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            n_ssa_samples=self.dataset.n_ssa_trajectories,
            dt=self.dt,
            val_every=self.val_every,
            grad_clip_norm=self.grad_clip_norm,
            scheduler_type=sched,
            training_mode=TrainingMode.TEACHER_FORCING,
            use_wandb=use_wandb,
            wandb_project=self.wandb_project,
            wandb_run_name=f"{self.experiment_name}_train",
        )

    def to_dict(self) -> dict:
        """Serialize to a flat dict for W&B config logging."""
        return dataclasses.asdict(self)
