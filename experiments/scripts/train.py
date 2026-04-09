"""Train the CRN surrogate.

Usage:
    python experiments/scripts/train.py
    python experiments/scripts/train.py experiment=mass_action_3s_v5
    python experiments/scripts/train.py training.lr=5e-4 model.d_model=256
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.training.trainer import Trainer
from experiments.builders import (
    build_model,
    build_model_config,
    build_simulator,
    build_training_config,
    select_device,
)
from experiments.checkpoint_resolver import CheckpointResolver
from experiments.dataset_loader import DatasetLoader
from experiments.wandb_session import WandbSession


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    device = select_device(cfg.device)
    use_wandb = not cfg.no_wandb
    flat_config: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )

    with WandbSession(
        project=cfg.wandb_project,
        name=f"{cfg.experiment_name}_train",
        group=cfg.wandb_group,
        job_type="training",
        config=flat_config,
        enabled=use_wandb,
    ) as session:
        # Data
        loader = DatasetLoader(cfg.dataset_dir)
        train_data, val_data = loader.load(cfg.wandb_artifact)

        # Model
        model_config = build_model_config(cfg)
        encoder = BipartiteGNNEncoder(model_config.encoder).to(device)
        model = build_model(cfg, device)
        simulator = build_simulator(cfg)
        train_config = build_training_config(cfg, use_wandb=use_wandb)

        print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"Model params:   {sum(p.numel() for p in model.parameters()):,}")

        # Resume
        trainer = Trainer(
            encoder, model, model_config, train_config, simulator=simulator
        )
        start_epoch = _handle_resume(cfg, trainer, encoder, model, device)

        # Train
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        result = trainer.train(train_data, val_data, start_epoch=start_epoch)

        # Save final artifact
        if session.active:
            ckpt_path = Path(train_config.checkpoint_dir) / "final.pt"
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "model_state": model.state_dict(),
                    "config": flat_config,
                    "train_losses": result.train_losses,
                    "val_losses": result.val_losses,
                },
                ckpt_path,
            )
            session.log_artifact(
                f"{cfg.experiment_name}_model",
                "model",
                ckpt_path,
                metadata={"final_train_loss": result.train_losses[-1]},
            )

        print(f"Final train loss: {result.train_losses[-1]:.4f}")


def _handle_resume(
    cfg: DictConfig, trainer, encoder, model, device: torch.device
) -> int:
    """Handle resume and resume_weights_only config fields.

    Args:
        cfg: Fully resolved Hydra config.
        trainer: Trainer instance (used for full checkpoint resume).
        encoder: Encoder module (for weights-only resume).
        model: SDE/drift model (for weights-only resume).
        device: Device to load checkpoint tensors onto.

    Returns:
        Starting epoch (1 if no checkpoint loaded).

    Raises:
        ValueError: If both resume and resume_weights_only are set.
    """
    resume = cfg.resume
    resume_weights = cfg.resume_weights_only

    if resume and resume_weights:
        raise ValueError("resume and resume_weights_only are mutually exclusive.")

    if not resume and not resume_weights:
        return 1

    resolver = CheckpointResolver(cfg.wandb_project, cfg.experiment_name)
    ref = resume_weights or resume
    checkpoint = resolver.resolve(ref, device)

    if checkpoint is None:
        print("No checkpoint loaded. Starting from scratch.")
        return 1

    if resume_weights:
        encoder.load_state_dict(checkpoint["encoder_state"])
        model.load_state_dict(checkpoint["model_state"])
        start = checkpoint.get("epoch", 0) + 1
        print(f"Loaded weights. Fresh training from epoch {start}.")
        return start

    return trainer.load_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
