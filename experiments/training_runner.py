"""Training runner for the CRN surrogate."""

from __future__ import annotations

from pathlib import Path

import torch

from crn_surrogate.configs.training_config import TrainingConfig
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.training.trainer import Trainer, TrainingResult
from experiments.builders import (
    build_model,
    build_model_config,
    build_simulator,
    build_training_config,
    select_device,
)
from experiments.experiment_context import ExperimentContext
from experiments.wandb_session import WandbSession


class TrainingRunner:
    """Runs the full training pipeline from a resolved ExperimentContext."""

    def __init__(self, ctx: ExperimentContext) -> None:
        """Args:
        ctx: Shared experiment context holding config and utilities.
        """
        self._ctx = ctx
        self._cfg = ctx.cfg

    def run(self) -> None:
        """Execute training end-to-end."""
        from experiments.dataset_loader import DatasetLoader

        device = select_device(self._cfg.device)
        run_cfg = self._cfg.run

        with self._ctx.wandb_session("training", "train") as session:
            # Data
            train_data, val_data = DatasetLoader(run_cfg.dataset_dir).load(
                run_cfg.wandb_artifact
            )

            # Model
            model_config = build_model_config(self._cfg)
            encoder = BipartiteGNNEncoder(model_config.encoder).to(device)
            model = build_model(self._cfg, device)
            simulator = build_simulator(self._cfg)
            train_config = build_training_config(self._cfg, use_wandb=session.active)

            print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
            print(f"Model params:   {sum(p.numel() for p in model.parameters()):,}")

            # Resume
            trainer = Trainer(
                encoder, model, model_config, train_config, simulator=simulator
            )
            start_epoch = self._handle_resume(trainer, encoder, model, device)

            # Train
            Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            result = trainer.train(train_data, val_data, start_epoch=start_epoch)

            # Final artifact
            self._save_final(session, encoder, model, train_config, result)
            print(f"Final train loss: {result.train_losses[-1]:.4f}")

    def _handle_resume(
        self,
        trainer: Trainer,
        encoder: BipartiteGNNEncoder,
        model: torch.nn.Module,
        device: torch.device,
    ) -> int:
        """Handle resume and resume_weights_only config fields.

        Args:
            trainer: Trainer instance for full checkpoint resume.
            encoder: Encoder module for weights-only resume.
            model: SDE/drift model for weights-only resume.
            device: Device to load checkpoint tensors onto.

        Returns:
            Starting epoch (1 if no checkpoint loaded).

        Raises:
            ValueError: If both resume and resume_weights_only are set.
        """
        run_cfg = self._cfg.run
        resume = run_cfg.resume
        resume_weights = run_cfg.resume_weights_only

        if resume and resume_weights:
            raise ValueError("resume and resume_weights_only are mutually exclusive.")
        if not resume and not resume_weights:
            return 1

        ref = resume_weights or resume
        artifact_name = f"{self._cfg.experiment_name}_train_periodic_checkpoint"
        path = self._ctx.resolve_checkpoint(ref, artifact_name)

        if path is None:
            print("No checkpoint found. Starting from scratch.")
            return 1

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if resume_weights:
            encoder.load_state_dict(checkpoint["encoder_state"])
            model.load_state_dict(checkpoint["model_state"])
            start = checkpoint.get("epoch", 0) + 1
            print(f"Loaded weights. Fresh training from epoch {start}.")
            return start

        return trainer.load_checkpoint(checkpoint)

    def _save_final(
        self,
        session: WandbSession,
        encoder: BipartiteGNNEncoder,
        model: torch.nn.Module,
        train_config: TrainingConfig,
        result: TrainingResult,
    ) -> None:
        """Save final model artifact to W&B.

        Args:
            session: Active WandbSession.
            encoder: Trained encoder module.
            model: Trained surrogate model.
            train_config: Training configuration (provides checkpoint_dir).
            result: TrainingResult with loss history.
        """
        if not session.active:
            return
        ckpt_path = Path(train_config.checkpoint_dir) / "final.pt"
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "model_state": model.state_dict(),
                "config": self._ctx.flat_config,
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
            },
            ckpt_path,
        )
        session.log_artifact(
            f"{self._cfg.experiment_name}_model",
            "model",
            ckpt_path,
            metadata={"final_train_loss": result.train_losses[-1]},
        )
