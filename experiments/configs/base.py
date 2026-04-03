"""Base experiment configuration with shared fields and builder methods.

Every experiment config inherits from BaseExperimentConfig and adds
experiment-specific fields (dataset config, topology constraints, etc.).
The builder methods translate flat config fields into the library's
structured config objects (EncoderConfig, SDEConfig, TrainingConfig).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)


@dataclass(frozen=True)
class BaseExperimentConfig:
    """Shared experiment configuration.

    Subclasses override defaults and add experiment-specific fields
    (e.g. dataset config). All builder methods are inherited.

    Attributes:
        experiment_name: Identifier used in artifact names and file paths.
        wandb_project: W&B project name.
        wandb_group: W&B run group for organizing related runs.
        max_n_species: SDE state dimension (pad smaller CRNs to this).
        max_n_reactions: SDE noise channels (pad smaller CRNs to this).
        d_model: Hidden dimension for GNN node embeddings.
        n_encoder_layers: Number of bipartite message-passing rounds.
        d_hidden: Hidden dimension inside drift/diffusion MLPs.
        n_sde_hidden_layers: FiLM-conditioned hidden layers per MLP.
        d_protocol: Protocol embedding dimension (0 = no protocol encoder).
        n_ssa_samples: SSA trajectories per dataset item used during training.
        max_epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        dt: Euler-Maruyama integration step size.
        grad_clip_norm: Gradient clipping threshold.
        scheduler_type: LR scheduler ("cosine" or "reduce_on_plateau").
        val_every: Validation frequency in epochs.
    """

    # ── Identity ─────────────────────────────────────────────────────────
    experiment_name: str = ""
    wandb_project: str = "crn-surrogate"
    wandb_group: str = ""

    # ── Architecture ─────────────────────────────────────────────────────
    max_n_species: int = 3
    max_n_reactions: int = 6
    d_model: int = 64
    n_encoder_layers: int = 3
    d_hidden: int = 128
    n_sde_hidden_layers: int = 2
    d_protocol: int = 0
    context_dropout: float = 0.0
    mlp_dropout: float = 0.0

    # ── Training ─────────────────────────────────────────────────────────
    n_ssa_samples: int = 32
    max_epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    dt: float = 0.1
    grad_clip_norm: float = 1.0
    scheduler_type: str = "cosine"
    val_every: int = 10

    # ── Builders ─────────────────────────────────────────────────────────

    def build_encoder_config(self) -> EncoderConfig:
        """Build encoder configuration from experiment settings."""
        return EncoderConfig(
            d_model=self.d_model,
            n_layers=self.n_encoder_layers,
            use_attention=True,
            context_dropout=self.context_dropout,
        )

    def build_sde_config(self) -> SDEConfig:
        """Build SDE config sized for the maximum topology in this experiment."""
        return SDEConfig(
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            n_noise_channels=self.max_n_reactions,
            n_hidden_layers=self.n_sde_hidden_layers,
            clip_state=True,
            d_protocol=self.d_protocol,
            mlp_dropout=self.mlp_dropout,
        )

    def build_model_config(self) -> ModelConfig:
        """Build the full model config (encoder + SDE)."""
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
            n_ssa_samples=self.n_ssa_samples,
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
