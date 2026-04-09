"""Resolve artifact references (local path, W&B artifact, 'auto') to local files."""

from __future__ import annotations

from pathlib import Path


class CheckpointResolver:
    """Resolves checkpoint references to local .pt file paths.

    Accepts three kinds of references:
    - "auto": downloads the latest version of a named W&B artifact.
    - A W&B artifact reference (contains ":" or "/"): downloads that artifact.
    - A local file path: returns it directly.
    """

    def __init__(self, wandb_project: str) -> None:
        """Args:
        wandb_project: W&B project name for artifact lookups.
        """
        self._wandb_project = wandb_project

    def resolve(
        self,
        reference: str,
        artifact_name: str | None = None,
    ) -> Path | None:
        """Resolve a reference to a local .pt file path.

        Args:
            reference: "auto", a W&B artifact ref, or a local path.
            artifact_name: Required when reference is "auto". The artifact
                name without version suffix, e.g.
                "mass_action_3s_det_train_checkpoint".

        Returns:
            Path to the resolved .pt file, or None if not found.
        """
        if reference == "auto":
            if artifact_name is None:
                raise ValueError("artifact_name is required for auto-resolve.")
            return self._download_artifact(
                f"{self._wandb_project}/{artifact_name}:latest"
            )
        if ":" in reference or "/" in reference:
            return self._download_artifact(reference)
        path = Path(reference)
        if not path.exists():
            print(f"File not found: {path}")
            return None
        return path

    def _download_artifact(self, artifact_ref: str) -> Path | None:
        """Download a W&B artifact, return path to its latest .pt file.

        Args:
            artifact_ref: Full W&B artifact reference string.

        Returns:
            Path to the newest .pt file in the artifact, or None on failure.
        """
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.run.use_artifact(artifact_ref)
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_ref)
            artifact_dir = Path(artifact.download())
            pt_files = sorted(artifact_dir.glob("*.pt"))
            if not pt_files:
                print(f"No .pt files in artifact {artifact_ref}")
                return None
            print(f"Resolved: {artifact_ref} -> {pt_files[-1].name}")
            return pt_files[-1]
        except Exception as exc:
            print(f"Artifact not found ({exc}).")
            return None
