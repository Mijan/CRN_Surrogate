"""Resolve checkpoint references to loaded state dicts."""

from __future__ import annotations

from pathlib import Path

import torch


class CheckpointResolver:
    """Resolves checkpoint references (local path, W&B artifact, 'auto')
    into loaded checkpoint dicts.
    """

    def __init__(self, wandb_project: str, experiment_name: str) -> None:
        """Args:
        wandb_project: W&B project name for artifact lookups.
        experiment_name: Experiment name prefix for auto-discovery.
        """
        self._wandb_project = wandb_project
        self._experiment_name = experiment_name

    def resolve(self, reference: str, device: torch.device) -> dict | None:
        """Resolve a checkpoint reference to a loaded dict.

        Args:
            reference: One of:
                - "auto": search W&B for the latest periodic checkpoint.
                - A W&B artifact reference (contains ":" or "/").
                - A local file path.
            device: Device to map tensors onto.

        Returns:
            Loaded checkpoint dict, or None if not found.
        """
        if reference == "auto":
            return self._resolve_auto(device)
        elif ":" in reference or "/" in reference:
            return self._resolve_artifact(reference, device)
        else:
            return self._resolve_local(Path(reference), device)

    def _resolve_auto(self, device: torch.device) -> dict | None:
        """Search W&B for the latest periodic checkpoint artifact.

        Args:
            device: Device to map tensors onto.

        Returns:
            Loaded checkpoint dict, or None if not found.
        """
        artifact_ref = (
            f"{self._wandb_project}/"
            f"{self._experiment_name}_train_periodic_checkpoint:latest"
        )
        return self._resolve_artifact(artifact_ref, device)

    def _resolve_artifact(self, artifact_ref: str, device: torch.device) -> dict | None:
        """Download and load a W&B artifact checkpoint.

        Args:
            artifact_ref: W&B artifact reference string.
            device: Device to map tensors onto.

        Returns:
            Loaded checkpoint dict, or None if the artifact is not found.
        """
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.run.use_artifact(artifact_ref)
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_ref)
            artifact_dir = Path(artifact.download())
            ckpt_files = sorted(artifact_dir.glob("*.pt"))
            if not ckpt_files:
                print(f"No .pt files in artifact {artifact_ref}")
                return None
            path = ckpt_files[-1]
            print(f"Resolved checkpoint: {artifact_ref} -> {path.name}")
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f"Checkpoint artifact not found ({exc}). Starting fresh.")
            return None

    @staticmethod
    def _resolve_local(path: Path, device: torch.device) -> dict | None:
        """Load a checkpoint from a local file.

        Args:
            path: Path to the checkpoint file.
            device: Device to map tensors onto.

        Returns:
            Loaded checkpoint dict, or None if the file does not exist.
        """
        if not path.exists():
            print(f"Checkpoint not found: {path}. Starting fresh.")
            return None
        print(f"Loading checkpoint: {path}")
        return torch.load(path, map_location=device, weights_only=False)
