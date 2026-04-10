from __future__ import annotations

from dataclasses import dataclass

from crn_surrogate.configs.labeled_enum import LabeledEnum


class SchedulerType(LabeledEnum):
    """Learning-rate scheduler choice."""

    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class TrainingMode(LabeledEnum):
    """Training strategy for the neural SDE."""

    TEACHER_FORCING = "teacher_forcing"
    FULL_ROLLOUT = "full_rollout"
    SCHEDULED_SAMPLING = "scheduled_sampling"


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    batch_size: int = 16
    grad_clip_norm: float = 1.0
    n_sde_samples: int = 8  # K parallel SDE rollouts per training item
    n_trajectory_samples: int = 32  # M independent SSA trajectories per dataset item
    dt: float = 0.1  # Euler-Maruyama step size
    val_every: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    training_mode: TrainingMode = TrainingMode.TEACHER_FORCING
    scheduled_sampling_start_epoch: int = 50
    scheduled_sampling_end_epoch: int = 150
    checkpoint_every: int = (
        0  # Save a periodic checkpoint every N epochs (0 to disable)
    )
    # Weights & Biases integration (requires `pip install wandb`)
    use_wandb: bool = False
    wandb_project: str = "crn-surrogate"
    wandb_run_name: str | None = None

    # DataLoader settings
    num_workers: int = 4
    shuffle_train: bool = True

    # DataCache settings
    gpu_memory_fraction: float = 0.5  # fraction of free GPU memory to use for cache

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(lr={self.lr}, max_epochs={self.max_epochs}, "
            f"batch_size={self.batch_size}, dt={self.dt}, "
            f"n_trajectory_samples={self.n_trajectory_samples}, scheduler={self.scheduler_type.value}, "
            f"training_mode={self.training_mode.value}, use_wandb={self.use_wandb})"
        )
