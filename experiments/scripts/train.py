"""Train the CRN surrogate on a generated mass-action dataset.

Usage:
    python experiments/scripts/train.py [--dataset-dir DIR] [--device auto|cpu|cuda|mps]
                                        [--no-wandb] [--seed N] [--max-epochs N]
                                        [--wandb-artifact ARTIFACT_REF]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.dataset import CRNTrajectoryDataset
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.training.trainer import Trainer
from experiments.configs.mass_action_3s import MassAction3sConfig


def _select_device(device_arg: str) -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device_arg: One of "auto", "cpu", "cuda", "mps".

    Returns:
        Selected torch.device.
    """
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Parse CLI args and run training."""
    parser = argparse.ArgumentParser(description="Train the CRN surrogate model.")
    parser.add_argument("--dataset-dir", default="experiments/datasets")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--wandb-artifact",
        default=None,
        help="W&B artifact reference (e.g. 'mass_action_3s_v1_dataset:latest')",
    )
    args = parser.parse_args()

    cfg = MassAction3sConfig()
    torch.manual_seed(args.seed)

    device = _select_device(args.device)
    print(f"Device: {device}")

    use_wandb = not args.no_wandb

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.wandb_artifact and use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type="training",
            name=f"{cfg.experiment_name}_train",
            config=cfg.to_dict(),
        )
        artifact = run.use_artifact(args.wandb_artifact)
        artifact_dir = artifact.download()
        train_dataset = torch.load(
            Path(artifact_dir) / f"{cfg.experiment_name}_train.pt",
            weights_only=False,
        )
        val_dataset = torch.load(
            Path(artifact_dir) / f"{cfg.experiment_name}_val.pt",
            weights_only=False,
        )
    else:
        dataset_dir = Path(args.dataset_dir)
        train_dataset = torch.load(
            dataset_dir / f"{cfg.experiment_name}_train.pt", weights_only=False
        )
        val_dataset = torch.load(
            dataset_dir / f"{cfg.experiment_name}_val.pt", weights_only=False
        )
        if use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                job_type="training",
                name=f"{cfg.experiment_name}_train",
                config=cfg.to_dict(),
            )

    print(f"Train: {len(train_dataset)} items | Val: {len(val_dataset)} items")

    # ── Build model ──────────────────────────────────────────────────────────
    model_config = cfg.build_model_config()
    encoder = BipartiteGNNEncoder(model_config.encoder).to(device)
    sde = CRNNeuralSDE(model_config.sde, n_species=cfg.max_n_species).to(device)

    train_config = cfg.build_training_config(use_wandb=use_wandb)
    if args.max_epochs is not None:
        train_config = dataclasses.replace(train_config, max_epochs=args.max_epochs)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"SDE params:     {sum(p.numel() for p in sde.parameters()):,}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(encoder, sde, model_config, train_config)
    result = trainer.train(train_dataset, val_dataset)

    # ── Save checkpoint as W&B artifact ──────────────────────────────────────
    if use_wandb:
        ckpt_path = Path(train_config.checkpoint_dir) / "final.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "sde_state": sde.state_dict(),
                "config": cfg.to_dict(),
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
            },
            ckpt_path,
        )
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_model",
            type="model",
            metadata={"final_train_loss": result.train_losses[-1]},
        )
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Final train loss: {result.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
