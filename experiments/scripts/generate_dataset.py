"""Generate a CRN dataset.

Usage:
    python experiments/scripts/generate_dataset.py
    python experiments/scripts/generate_dataset.py experiment=mass_action_3s_v7
    python experiments/scripts/generate_dataset.py dataset.n_train=100000
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dataset_generator import DatasetGenerator
from experiments.experiment_context import ExperimentContext


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for dataset generation."""
    ctx = ExperimentContext(cfg)
    ctx.setup()
    DatasetGenerator(ctx).run()


if __name__ == "__main__":
    main()
