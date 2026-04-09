"""Train the CRN surrogate on a generated dataset.

Usage:
    python experiments/scripts/train.py                          # defaults
    python experiments/scripts/train.py experiment=mass_action_3s_v5
    python experiments/scripts/train.py training.lr=5e-4 model.d_model=256
    python experiments/scripts/train.py experiment=mass_action_3s_det device=cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _resolve_checkpoint(
    resume_arg: str,
    cfg: DictConfig,
    device: torch.device,
    use_wandb: bool,
) -> dict | None:
    """Resolve a resume argument to a loaded checkpoint dict.

    Accepts a local path, a W&B artifact reference, or 'auto'.

    Args:
        resume_arg: One of 'auto', a W&B artifact reference, or a local path.
        cfg: Fully resolved Hydra config.
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
            candidates = [
                f"{cfg.wandb_project}/{cfg.experiment_name}_train_periodic_checkpoint:latest",
            ]
            for artifact_ref in candidates:
                try:
                    api = wandb.Api() if wandb.run is None else None
                    artifact = (
                        wandb.run.use_artifact(artifact_ref)
                        if wandb.run is not None
                        else api.artifact(artifact_ref)
                    )
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
            try:
                api = wandb.Api() if wandb.run is None else None
                artifact = (
                    wandb.run.use_artifact(resume_arg)
                    if wandb.run is not None
                    else api.artifact(resume_arg)
                )
                artifact_dir = Path(artifact.download())
                ckpt_files = sorted(artifact_dir.glob("*.pt"))
                if not ckpt_files:
                    print(f"No .pt files in artifact {resume_arg}")
                    return None
                ckpt_path = ckpt_files[-1]
            except Exception as exc:
                print(f"No checkpoint artifact found ({exc}). Starting fresh.")
                return None
    else:
        ckpt_path = Path(resume_arg)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}. Starting fresh.")
            return None

    return torch.load(ckpt_path, map_location=device, weights_only=False)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    device = select_device(cfg.device)
    print(f"Device: {device}")

    use_wandb = not cfg.no_wandb

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset_dir = Path(cfg.dataset_dir)
    train_files = sorted(dataset_dir.glob("*_train.pt"))
    val_files = sorted(dataset_dir.glob("*_val.pt"))

    wandb_artifact = cfg.wandb_artifact
    if wandb_artifact and use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type="training",
            name=f"{cfg.experiment_name}_train",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        artifact = run.use_artifact(wandb_artifact)
        artifact_path = Path(artifact.download())
        train_files = sorted(artifact_path.glob("*_train.pt"))
        val_files = sorted(artifact_path.glob("*_val.pt"))

    if not train_files or not val_files:
        search_dir = dataset_dir if not wandb_artifact else artifact_path
        raise FileNotFoundError(
            f"Expected *_train.pt and *_val.pt in {search_dir}"
        )

    train_dataset = torch.load(train_files[0], weights_only=False)
    val_dataset = torch.load(val_files[0], weights_only=False)

    if use_wandb and not wandb_artifact:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type="training",
            name=f"{cfg.experiment_name}_train",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    print(f"Train: {len(train_dataset)} items | Val: {len(val_dataset)} items")

    # ── Build model ──────────────────────────────────────────────────────
    model_config = build_model_config(cfg)
    encoder = BipartiteGNNEncoder(model_config.encoder).to(device)
    model = build_model(cfg, device)
    simulator = build_simulator(cfg)
    train_config = build_training_config(cfg, use_wandb=use_wandb)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Model params:   {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume ───────────────────────────────────────────────────────────
    trainer = Trainer(encoder, model, model_config, train_config, simulator=simulator)
    start_epoch = 1

    resume = cfg.resume
    resume_weights_only = cfg.resume_weights_only

    if resume and resume_weights_only:
        raise ValueError("resume and resume_weights_only are mutually exclusive.")

    if resume_weights_only:
        checkpoint = _resolve_checkpoint(resume_weights_only, cfg, device, use_wandb)
        if checkpoint is not None:
            encoder.load_state_dict(checkpoint["encoder_state"])
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Loaded weights. Fresh training from epoch {start_epoch}.")

    if resume:
        checkpoint = _resolve_checkpoint(resume, cfg, device, use_wandb)
        if checkpoint is not None:
            start_epoch = trainer.load_checkpoint(checkpoint)
            print(f"Resuming from epoch {start_epoch}")

    # ── Train ────────────────────────────────────────────────────────────
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    result = trainer.train(train_dataset, val_dataset, start_epoch=start_epoch)

    # ── Final artifact ───────────────────────────────────────────────────
    if use_wandb:
        ckpt_path = Path(train_config.checkpoint_dir) / "final.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "model_state": model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
            },
            ckpt_path,
        )
        import wandb

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
