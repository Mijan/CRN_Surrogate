"""Train the CRN surrogate on a generated dataset.

Usage:
    python experiments/scripts/train.py [--config NAME]
                                        [--dataset-dir DIR]
                                        [--device auto|cpu|cuda|mps]
                                        [--no-wandb]
                                        [--seed N]
                                        [--max-epochs N]
                                        [--wandb-artifact ARTIFACT_REF]
                                        [--resume PATH|ARTIFACT_REF|auto]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.training.trainer import Trainer
from experiments.configs.registry import available_configs, get_config


def _resolve_checkpoint(
    resume_arg: str,
    cfg,
    device: torch.device,
    use_wandb: bool,
) -> dict | None:
    """Resolve a --resume argument to a loaded checkpoint dict.

    Args:
        resume_arg: One of:
            - "auto": search W&B for the latest checkpoint artifact
            - A W&B artifact reference (contains ":" or "/")
            - A local file path
        cfg: Experiment config (for constructing artifact names).
        device: Device to load tensors onto.
        use_wandb: Whether W&B is available.

    Returns:
        Loaded checkpoint dict, or None if no checkpoint found.
    """
    ckpt_path: Path | None = None

    if resume_arg == "auto" or ":" in resume_arg or "/" in resume_arg:
        if not use_wandb:
            print("--resume with W&B artifact requires W&B. Skipping.")
            return None
        import wandb

        if resume_arg == "auto":
            # Prefer best-validation checkpoint; fall back to periodic checkpoint
            candidates = [
                # f"{cfg.wandb_project}/{cfg.experiment_name}_train_model_checkpoint:latest",
                f"{cfg.wandb_project}/{cfg.experiment_name}_train_periodic_checkpoint:latest",
            ]
            ckpt_path = None
            for artifact_ref in candidates:
                try:
                    if wandb.run is not None:
                        artifact = wandb.run.use_artifact(artifact_ref)
                    else:
                        api = wandb.Api()
                        artifact = api.artifact(artifact_ref)
                    artifact_dir = Path(artifact.download())
                    ckpt_files = sorted(artifact_dir.glob("*.pt"))
                    if ckpt_files:
                        ckpt_path = ckpt_files[-1]
                        print(f"Auto-resume: {artifact_ref} -> {ckpt_path.name}")
                        break
                except Exception:
                    continue
            if ckpt_path is None:
                print("No checkpoint artifact found. Starting fresh.")
                return None
        else:
            artifact_ref = resume_arg
            try:
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
                ckpt_path = ckpt_files[-1]
                print(f"Resume (artifact): {artifact_ref} -> {ckpt_path.name}")
            except Exception as exc:  # wandb.errors.CommError or similar
                print(f"No checkpoint artifact found ({exc}). Starting fresh.")
                return None
    else:
        ckpt_path = Path(resume_arg)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}. Starting fresh.")
            return None
        print(f"Resume from local: {ckpt_path}")

    return torch.load(ckpt_path, map_location=device, weights_only=False)


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
    parser.add_argument(
        "--config",
        default="mass_action_3s",
        choices=available_configs(),
        help="Experiment config name",
    )
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
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume from a checkpoint. Accepts: "
            "(1) local path to a .pt file, "
            "(2) W&B artifact reference (e.g. 'mass_action_3s_v3_train_checkpoint:latest'), "
            "(3) 'auto' to automatically find the latest W&B checkpoint for this experiment"
        ),
    )
    parser.add_argument(
        "--resume-weights-only",
        default=None,
        help=(
            "Load model weights from a checkpoint but start training fresh "
            "(new optimizer, scheduler, epoch counter). Accepts the same values "
            "as --resume: local path, W&B artifact reference, or 'auto'."
        ),
    )
    args = parser.parse_args()

    if args.resume and args.resume_weights_only:
        print("Error: --resume and --resume-weights-only are mutually exclusive.")
        sys.exit(1)

    cfg = get_config(args.config)
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
        artifact_path = Path(artifact_dir)
        train_files = sorted(artifact_path.glob("*_train.pt"))
        val_files = sorted(artifact_path.glob("*_val.pt"))
        if not train_files or not val_files:
            raise FileNotFoundError(
                f"Expected *_train.pt and *_val.pt in {artifact_path}, "
                f"found: {list(artifact_path.iterdir())}"
            )
        train_dataset = torch.load(train_files[0], weights_only=False)
        val_dataset = torch.load(val_files[0], weights_only=False)
    else:
        dataset_dir = Path(args.dataset_dir)
        train_files = sorted(dataset_dir.glob("*_train.pt"))
        val_files = sorted(dataset_dir.glob("*_val.pt"))
        if not train_files or not val_files:
            raise FileNotFoundError(
                f"Expected *_train.pt and *_val.pt in {dataset_dir}, "
                f"found: {list(dataset_dir.iterdir())}"
            )
        train_dataset = torch.load(train_files[0], weights_only=False)
        val_dataset = torch.load(val_files[0], weights_only=False)
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
    model = cfg.build_model(device)
    simulator = cfg.build_simulator()

    train_config = cfg.build_training_config(use_wandb=use_wandb)
    if args.max_epochs is not None:
        train_config = dataclasses.replace(train_config, max_epochs=args.max_epochs)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Model params:   {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume from checkpoint if requested ──────────────────────────────────
    trainer = Trainer(encoder, model, model_config, train_config, simulator=simulator)
    start_epoch = 1
    if args.resume_weights_only:
        checkpoint = _resolve_checkpoint(args.resume_weights_only, cfg, device, use_wandb)
        if checkpoint is not None:
            encoder.load_state_dict(checkpoint["encoder_state"])
            model.load_state_dict(checkpoint["sde_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Loaded model weights from checkpoint. Starting fresh training from epoch {start_epoch} with config LR={train_config.lr}, scheduler={train_config.scheduler_type}")
        else:
            print("No checkpoint loaded. Starting from scratch.")
    if args.resume:
        checkpoint = _resolve_checkpoint(args.resume, cfg, device, use_wandb)
        if checkpoint is not None:
            start_epoch = trainer.load_checkpoint(checkpoint)
            print(f"Will resume training from epoch {start_epoch}")
        else:
            print("No checkpoint loaded. Starting from scratch.")

    # ── Train ────────────────────────────────────────────────────────────────
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    result = trainer.train(train_dataset, val_dataset, start_epoch=start_epoch)

    # ── Save checkpoint as W&B artifact ──────────────────────────────────────
    if use_wandb:
        ckpt_path = Path(train_config.checkpoint_dir) / "final.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "sde_state": model.state_dict(),
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
