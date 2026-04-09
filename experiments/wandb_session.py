"""W&B session management as a context manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbSession:
    """Context manager for a W&B run.

    Usage:
        with WandbSession(project="x", name="y", config={...}) as session:
            session.log({"loss": 0.5})
            session.log_artifact("my_model", "model", path, metadata={})
        # run.finish() is called automatically
    """

    def __init__(
        self,
        project: str,
        name: str,
        group: str = "",
        job_type: str = "training",
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        """Args:
        project: W&B project name.
        name: W&B run name.
        group: Optional run group.
        job_type: W&B job type label.
        config: Config dict to log with the run.
        enabled: If False, all methods are no-ops.
        """
        self._project = project
        self._name = name
        self._group = group
        self._job_type = job_type
        self._config = config or {}
        self._enabled = enabled
        self._run: Any = None

    def __enter__(self) -> "WandbSession":
        if not self._enabled:
            return self
        import wandb

        self._run = wandb.init(
            project=self._project,
            group=self._group,
            job_type=self._job_type,
            name=self._name,
            config=self._config,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._run is not None:
            self._run.finish()

    @property
    def active(self) -> bool:
        """Whether a W&B run is active."""
        return self._run is not None

    def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics if W&B is active.

        Args:
            metrics: Key-value metrics to log.
        """
        if self._run is not None:
            self._run.log(metrics)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        file_path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        """Log a single file as a versioned W&B artifact.

        Args:
            name: Artifact name.
            artifact_type: Artifact type label (e.g. "model", "dataset").
            file_path: Path to the file to upload.
            metadata: Optional metadata dict.
        """
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        artifact.add_file(str(file_path))
        self._run.log_artifact(artifact)

    def log_multi_file_artifact(
        self,
        name: str,
        artifact_type: str,
        file_paths: list[str | Path],
        metadata: dict | None = None,
    ) -> None:
        """Log multiple files as a single versioned W&B artifact.

        Args:
            name: Artifact name.
            artifact_type: Artifact type label.
            file_paths: List of file paths to include in the artifact.
            metadata: Optional metadata dict.
        """
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(
            name=name, type=artifact_type, metadata=metadata or {}
        )
        for fp in file_paths:
            artifact.add_file(str(fp))
        self._run.log_artifact(artifact)

    def use_artifact(self, artifact_ref: str):
        """Declare artifact usage for lineage tracking.

        Args:
            artifact_ref: W&B artifact reference string.

        Returns:
            W&B artifact object, or None if no run is active.
        """
        if self._run is not None:
            return self._run.use_artifact(artifact_ref)
        return None
