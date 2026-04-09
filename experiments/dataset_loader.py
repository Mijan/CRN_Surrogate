"""Load train/val datasets from local files or W&B artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from crn_surrogate.data.dataset import CRNTrajectoryDataset


class DatasetLoader:
    """Loads train/val CRNTrajectoryDataset splits.

    Supports two sources:
    - Local directory containing *_train.pt and *_val.pt files.
    - W&B artifact reference, downloaded and searched the same way.
    """

    def __init__(self, dataset_dir: str | Path = "experiments/datasets") -> None:
        """Args:
            dataset_dir: Default local directory to search for dataset files.
        """
        self._dataset_dir = Path(dataset_dir)

    def load(
        self, wandb_artifact: str | None = None
    ) -> tuple[CRNTrajectoryDataset, CRNTrajectoryDataset]:
        """Load train and validation datasets.

        Args:
            wandb_artifact: Optional W&B artifact reference. If provided,
                the artifact is downloaded and datasets are loaded from it.
                Otherwise, datasets are loaded from the local dataset_dir.

        Returns:
            (train_dataset, val_dataset) tuple.

        Raises:
            FileNotFoundError: If no *_train.pt or *_val.pt files are found.
        """
        if wandb_artifact is not None:
            search_dir = self._download_artifact(wandb_artifact)
        else:
            search_dir = self._dataset_dir

        train_files = sorted(search_dir.glob("*_train.pt"))
        val_files = sorted(search_dir.glob("*_val.pt"))

        if not train_files or not val_files:
            raise FileNotFoundError(
                f"Expected *_train.pt and *_val.pt in {search_dir}, "
                f"found: {list(search_dir.iterdir())}"
            )

        train_dataset = torch.load(train_files[0], weights_only=False)
        val_dataset = torch.load(val_files[0], weights_only=False)
        print(f"Train: {len(train_dataset)} items | Val: {len(val_dataset)} items")
        return train_dataset, val_dataset

    @staticmethod
    def _download_artifact(artifact_ref: str) -> Path:
        """Download a W&B artifact and return the local directory path.

        Args:
            artifact_ref: W&B artifact reference string.

        Returns:
            Local directory containing the downloaded artifact files.
        """
        import wandb

        if wandb.run is not None:
            artifact = wandb.run.use_artifact(artifact_ref)
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact_ref)
        return Path(artifact.download())
