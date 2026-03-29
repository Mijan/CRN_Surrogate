from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SchedulerType(Enum):
    """Learning-rate scheduler choice."""

    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    batch_size: int = 16
    grad_clip_norm: float = 1.0
    n_sde_samples: int = 8  # K parallel SDE rollouts per training item
    n_ssa_samples: int = 32  # M independent SSA trajectories per dataset item
    dt: float = 0.1  # Euler-Maruyama step size
    val_every: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(lr={self.lr}, max_epochs={self.max_epochs}, "
            f"batch_size={self.batch_size}, dt={self.dt}, "
            f"n_ssa_samples={self.n_ssa_samples}, scheduler={self.scheduler_type.value})"
        )
