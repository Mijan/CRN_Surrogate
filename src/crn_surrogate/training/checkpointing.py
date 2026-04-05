"""Checkpoint saving, loading, and lifecycle management for the CRN surrogate."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """Manages model checkpoint saving, loading, and cleanup.

    Tracks best validation loss, saves best and periodic checkpoints,
    handles W&B artifact logging, and cleans up old periodic files.

    Two checkpoint streams are maintained independently:
    - Best-validation: saved to ``best_epochN.pt`` whenever val_loss improves.
    - Periodic: saved to ``periodic_epochN.pt`` every ``checkpoint_every`` epochs;
      only the most recent ``max_periodic_kept`` files are kept on disk, but all
      versions are preserved as versioned W&B artifacts.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        wandb_run_name: str | None = None,
        checkpoint_every: int = 0,
        max_periodic_kept: int = 3,
    ) -> None:
        """Args:
        checkpoint_dir: Directory for saving checkpoint files.
        wandb_run_name: Prefix for W&B artifact names. None disables W&B logging.
        checkpoint_every: Epoch interval for periodic checkpoints (0 = disabled).
        max_periodic_kept: Number of periodic checkpoint files to retain on disk.
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._wandb_run_name = wandb_run_name
        self._checkpoint_every = checkpoint_every
        self._max_periodic_kept = max_periodic_kept
        self._best_val_loss: float = float("inf")

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen so far."""
        return self._best_val_loss

    def save_best(self, state: dict, val_loss: float, epoch: int) -> None:
        """Save a checkpoint if val_loss improves on the current best.

        Args:
            state: Checkpoint dict built by the caller (encoder, SDE, optimizer,
                scheduler state dicts plus any extra metadata).
            val_loss: Current validation loss to compare against the best.
            epoch: Current epoch number (used in the filename).
        """
        if val_loss >= self._best_val_loss:
            return
        self._best_val_loss = val_loss
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"best_epoch{epoch}.pt"
        torch.save(state, path)
        if self._wandb_run_name is not None:
            self._log_artifact(
                path,
                artifact_name=f"{self._wandb_run_name}_model_checkpoint",
                metadata={"epoch": epoch, "val_loss": val_loss},
            )

    def save_periodic(self, state: dict, epoch: int, train_loss: float) -> None:
        """Save a periodic checkpoint regardless of validation performance.

        Does nothing if ``checkpoint_every`` is 0 or ``epoch`` is not a
        multiple of ``checkpoint_every``. Keeps only the most recent
        ``max_periodic_kept`` files on disk.

        Args:
            state: Checkpoint dict built by the caller.
            epoch: Current epoch number (used for interval check and filename).
            train_loss: Current training loss (stored in the checkpoint).
        """
        if self._checkpoint_every <= 0:
            return
        if epoch % self._checkpoint_every != 0:
            return

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"periodic_epoch{epoch}.pt"
        torch.save(state, path)

        # Keep only the most recent files on disk (sort numerically by epoch)
        all_periodic = sorted(
            self._checkpoint_dir.glob("periodic_epoch*.pt"),
            key=lambda p: int(p.stem.removeprefix("periodic_epoch")),
        )
        for old_file in all_periodic[: -self._max_periodic_kept]:
            old_file.unlink()

        if self._wandb_run_name is not None:
            self._log_artifact(
                path,
                artifact_name=f"{self._wandb_run_name}_periodic_checkpoint",
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "checkpoint_type": "periodic",
                },
            )

    def load(
        self,
        checkpoint: dict,
        encoder: nn.Module,
        sde: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> int:
        """Restore training state from a checkpoint dict.

        Loads model weights, optimizer state, scheduler state, and best_val_loss.
        Returns the next epoch to train (checkpoint epoch + 1).

        Args:
            checkpoint: Dict loaded from a checkpoint .pt file.
            encoder: Encoder module to restore weights into.
            sde: SDE module to restore weights into.
            optimizer: Optimizer to restore state into.
            scheduler: LR scheduler to restore state into.

        Returns:
            The epoch to resume from (checkpoint["epoch"] + 1).
        """
        encoder.load_state_dict(checkpoint["encoder_state"])
        sde.load_state_dict(checkpoint["sde_state"])

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "best_val_loss" in checkpoint:
            self._best_val_loss = checkpoint["best_val_loss"]

        epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {epoch} (best_val_loss={self._best_val_loss:.4f})")
        return epoch + 1

    def _log_artifact(
        self,
        path: Path,
        artifact_name: str,
        metadata: dict,
    ) -> None:
        """Log a checkpoint file as a W&B artifact.

        Args:
            path: Local path to the checkpoint file.
            artifact_name: W&B artifact name (versioned automatically).
            metadata: Key-value metadata to attach to the artifact.
        """
        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model-checkpoint",
            metadata=metadata,
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
