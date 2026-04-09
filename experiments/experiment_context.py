"""Shared experiment context for all scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from experiments.checkpoint_resolver import CheckpointResolver
from experiments.wandb_session import WandbSession


class ExperimentContext:
    """Shared context for experiment scripts.

    Holds the Hydra config and provides access to common utilities
    (resolver, flat config dict, WandbSession creation).
    Calling setup() prints the config and seeds the RNG.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Args:
        cfg: Fully resolved Hydra config.
        """
        self.cfg = cfg
        self.flat_config: dict[str, Any] = cast(
            dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
        )
        self.resolver = CheckpointResolver(cfg.wandb_project)

    def setup(self) -> None:
        """Print config and seed RNG. Call once at script start."""
        print(OmegaConf.to_yaml(self.cfg))
        torch.manual_seed(self.cfg.seed)

    def wandb_session(self, job_type: str, name_suffix: str) -> WandbSession:
        """Create a WandbSession with standard project/group/name settings.

        Args:
            job_type: W&B job type label (e.g. "training", "data-generation").
            name_suffix: Appended to experiment_name to form the run name.

        Returns:
            Configured WandbSession (not yet entered).
        """
        return WandbSession(
            project=self.cfg.wandb_project,
            name=f"{self.cfg.experiment_name}_{name_suffix}",
            group=self.cfg.wandb_group,
            job_type=job_type,
            config=self.flat_config,
            enabled=not self.cfg.no_wandb,
        )

    def resolve_checkpoint(
        self, reference: str | None, artifact_name: str
    ) -> Path | None:
        """Resolve a checkpoint reference, returning None if ref is null/empty.

        Args:
            reference: "auto", a W&B artifact ref, a local path, or None.
            artifact_name: Artifact name used for "auto" resolution.

        Returns:
            Resolved local path, or None if reference is absent or not found.
        """
        if not reference:
            return None
        return self.resolver.resolve(reference, artifact_name=artifact_name)
