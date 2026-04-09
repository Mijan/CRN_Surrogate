"""Train the CRN surrogate.

Usage:
    python experiments/scripts/train.py
    python experiments/scripts/train.py experiment=mass_action_3s_v5
    python experiments/scripts/train.py training.lr=5e-4 model.d_model=256
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.experiment_context import ExperimentContext
from experiments.training_runner import TrainingRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    ctx = ExperimentContext(cfg)
    ctx.setup()
    TrainingRunner(ctx).run()


if __name__ == "__main__":
    main()
