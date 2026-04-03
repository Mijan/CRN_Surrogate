"""Generate a CRN dataset and log it as a W&B artifact.

Usage:
    python experiments/scripts/generate_dataset.py [--config NAME]
                                                   [--output-dir DIR]
                                                   [--no-wandb]
                                                   [--seed N]
                                                   [--checkpoint-every N]
                                                   [--resume PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.configs import CurationConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.mass_action_generator import MassActionCRNGenerator
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.training.normalization import TrajectoryNormalizer
from experiments.configs.registry import available_configs, get_config


def _make_checkpoint_fn(
    output_dir: Path,
    experiment_name: str,
    split_name: str,
    use_wandb: bool,
) -> Callable[[list[TrajectoryItem], str], None]:
    """Create a checkpoint callback that saves intermediate items to disk.

    Args:
        output_dir: Directory to write checkpoint files.
        experiment_name: Used as filename prefix.
        split_name: "train" or "val".
        use_wandb: Whether to log checkpoint progress to W&B.

    Returns:
        Callable that accepts (items, label) and saves a CRNTrajectoryDataset.
    """
    def _checkpoint(items: list[TrajectoryItem], label: str) -> None:
        path = output_dir / f"{experiment_name}_{split_name}_{label}.pt"
        torch.save(CRNTrajectoryDataset(items), path)
        print(f"  Checkpoint: {len(items)} items -> {path.name}")

        if use_wandb:
            import wandb
            # Log as a versioned artifact so it survives Colab disconnects
            artifact = wandb.Artifact(
                name=f"{experiment_name}_{split_name}_checkpoint",
                type="dataset-checkpoint",
                metadata={"n_items": len(items), "label": label},
            )
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
            wandb.log({f"data/{split_name}_items": len(items)})

    return _checkpoint


def _generate_split(
    gen: MassActionCRNGenerator,
    ssa: GillespieSSA,
    time_grid: torch.Tensor,
    *,
    n_items: int,
    n_ssa_trajectories: int,
    n_workers: int,
    initial_state_mean: float,
    initial_state_spread: float,
    t_max: float,
    use_wandb: bool = False,
    checkpoint_fn: Callable[[list[TrajectoryItem], str], None] | None = None,
    checkpoint_every: int = 50,
    resume_items: list[TrajectoryItem] | None = None,
) -> tuple[list[TrajectoryItem], dict]:
    """Generate one dataset split, returning (items, metadata_dict).

    Args:
        gen: Configured CRN generator.
        ssa: Gillespie SSA simulator.
        time_grid: (T,) shared time grid for all trajectories.
        n_items: Target number of items to generate.
        n_ssa_trajectories: SSA trajectories per CRN.
        n_workers: Parallel workers for SSA simulation.
        initial_state_mean: Geometric mean of initial molecule counts.
        initial_state_spread: Geometric standard deviation for initial states.
        t_max: Simulation end time.
        use_wandb: Whether to log progress metrics to W&B.
        checkpoint_fn: Optional callback called every checkpoint_every items.
        checkpoint_every: Checkpoint interval in number of accepted items.
        resume_items: Pre-existing items to resume from.

    Returns:
        Tuple of (items, stats_dict).
    """
    items: list[TrajectoryItem] = list(resume_items) if resume_items else []
    if items:
        print(f"  Resuming from {len(items)} existing items")

    stats: dict = {
        "n_attempted": 0,
        "n_curated_pass": 0,
        "n_species_dist": {},
        "n_reactions_dist": {},
    }
    filter_ = ViabilityFilter(CurationConfig())
    max_attempts = n_items * 5

    pbar = tqdm(total=n_items, initial=len(items), desc="generating", unit="item")

    while len(items) < n_items and stats["n_attempted"] < max_attempts:
        crn = gen.sample()
        init_state = gen.sample_initial_state(
            crn,
            mean_molecules=initial_state_mean,
            spread=initial_state_spread,
        )
        crn_repr = crn_to_tensor_repr(crn)

        trajectories_list = ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init_state,
            t_max=t_max,
            n_trajectories=n_ssa_trajectories,
            n_workers=min(n_workers, n_ssa_trajectories),
        )
        traj_tensor = Trajectory.stack_on_grid(trajectories_list, time_grid)

        stats["n_attempted"] += 1
        result = filter_.check(traj_tensor)

        if not result.viable:
            pbar.set_postfix(
                attempts=stats["n_attempted"],
                rate=f"{stats['n_curated_pass']}/{stats['n_attempted']}",
            )
            continue

        stats["n_curated_pass"] += 1
        items.append(
            TrajectoryItem(
                crn_repr=crn_repr,
                initial_state=init_state,
                trajectories=traj_tensor,
                times=time_grid,
                scale=TrajectoryNormalizer().compute_scale(traj_tensor),
                motif_label="mass_action",
            )
        )

        ns = str(crn.n_species)
        nr = str(crn.n_reactions)
        stats["n_species_dist"][ns] = stats["n_species_dist"].get(ns, 0) + 1
        stats["n_reactions_dist"][nr] = stats["n_reactions_dist"].get(nr, 0) + 1

        pbar.update(1)
        pbar.set_postfix(
            pass_rate=f"{stats['n_curated_pass']/stats['n_attempted']:.0%}",
            species=ns,
        )

        if use_wandb and len(items) % 10 == 0:
            import wandb
            wandb.log({
                "data/items_generated": len(items),
                "data/attempts": stats["n_attempted"],
                "data/pass_rate": stats["n_curated_pass"] / stats["n_attempted"],
            })

        if (
            checkpoint_fn is not None
            and checkpoint_every > 0
            and len(items) % checkpoint_every == 0
        ):
            checkpoint_fn(items, f"checkpoint_{len(items)}")

    pbar.close()

    stats["pass_rate"] = stats["n_curated_pass"] / max(stats["n_attempted"], 1)
    print(
        f"  Generated {len(items)}/{n_items} items "
        f"({stats['n_attempted']} attempted, {stats['pass_rate']:.0%} pass rate)"
    )
    print(f"  Species distribution:   {stats['n_species_dist']}")
    print(f"  Reactions distribution: {stats['n_reactions_dist']}")

    if len(items) < n_items:
        warnings.warn(
            f"Only generated {len(items)}/{n_items} items after {stats['n_attempted']} attempts.",
            RuntimeWarning,
            stacklevel=2,
        )

    return items, stats


def generate(
    cfg,
    output_dir: Path,
    *,
    use_wandb: bool,
    seed: int,
    checkpoint_every: int,
    resume_train: Path | None = None,
    resume_val: Path | None = None,
) -> None:
    """Run dataset generation and optionally log as a W&B artifact.

    Args:
        cfg: Experiment configuration (must have a .dataset attribute).
        output_dir: Directory to write dataset files.
        use_wandb: Whether to log an artifact to W&B.
        seed: Random seed for reproducibility.
        checkpoint_every: Save intermediate dataset every N accepted items.
        resume_train: Optional checkpoint path to resume training split from.
        resume_val: Optional checkpoint path to resume validation split from.
    """
    torch.manual_seed(seed)

    if use_wandb:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type="data-generation",
            name=f"{cfg.experiment_name}_data",
            config=cfg.to_dict(),
        )

    gen = MassActionCRNGenerator(cfg.dataset.generator)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, cfg.dataset.t_max, cfg.dataset.n_time_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_train_items = None
    if resume_train is not None:
        loaded = torch.load(resume_train, weights_only=False)
        resume_train_items = list(loaded)
        print(f"Loaded {len(resume_train_items)} items from {resume_train.name}")

    resume_val_items = None
    if resume_val is not None:
        loaded = torch.load(resume_val, weights_only=False)
        resume_val_items = list(loaded)
        print(f"Loaded {len(resume_val_items)} items from {resume_val.name}")

    print(f"Generating {cfg.dataset.n_train} training items...")
    train_items, train_meta = _generate_split(
        gen, ssa, time_grid,
        n_items=cfg.dataset.n_train,
        n_ssa_trajectories=cfg.dataset.n_ssa_trajectories,
        n_workers=cfg.dataset.n_workers,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        use_wandb=use_wandb,
        checkpoint_fn=_make_checkpoint_fn(output_dir, cfg.experiment_name, "train", use_wandb),
        checkpoint_every=checkpoint_every,
        resume_items=resume_train_items,
    )

    print(f"Generating {cfg.dataset.n_val} validation items...")
    val_items, val_meta = _generate_split(
        gen, ssa, time_grid,
        n_items=cfg.dataset.n_val,
        n_ssa_trajectories=cfg.dataset.n_ssa_trajectories,
        n_workers=cfg.dataset.n_workers,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        use_wandb=use_wandb,
        checkpoint_fn=_make_checkpoint_fn(output_dir, cfg.experiment_name, "val", use_wandb),
        checkpoint_every=checkpoint_every,
        resume_items=resume_val_items,
    )

    train_path = output_dir / f"{cfg.experiment_name}_train.pt"
    val_path = output_dir / f"{cfg.experiment_name}_val.pt"
    meta_path = output_dir / f"{cfg.experiment_name}_meta.json"

    torch.save(CRNTrajectoryDataset(train_items), train_path)
    torch.save(CRNTrajectoryDataset(val_items), val_path)

    metadata = {
        "experiment": cfg.experiment_name,
        "seed": seed,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "config": cfg.to_dict(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved: {train_path}, {val_path}, {meta_path}")

    if use_wandb:
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_dataset",
            type="dataset",
            metadata=metadata,
        )
        artifact.add_file(str(train_path))
        artifact.add_file(str(val_path))
        artifact.add_file(str(meta_path))
        run.log_artifact(artifact)
        run.finish()
        print(f"Logged W&B artifact: {cfg.experiment_name}_dataset")


def main() -> None:
    """Parse CLI args and run generation."""
    parser = argparse.ArgumentParser(description="Generate a CRN dataset.")
    parser.add_argument(
        "--config",
        default="mass_action_3s",
        choices=available_configs(),
        help="Experiment config name",
    )
    parser.add_argument("--output-dir", default="experiments/datasets")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save intermediate dataset every N accepted items (0 to disable)",
    )
    parser.add_argument(
        "--resume-train",
        default=None,
        help="Path to a training split checkpoint .pt file to resume from",
    )
    parser.add_argument(
        "--resume-val",
        default=None,
        help="Path to a validation split checkpoint .pt file to resume from",
    )
    args = parser.parse_args()

    cfg = get_config(args.config)
    generate(
        cfg,
        output_dir=Path(args.output_dir),
        use_wandb=not args.no_wandb,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        resume_train=Path(args.resume_train) if args.resume_train else None,
        resume_val=Path(args.resume_val) if args.resume_val else None,
    )


if __name__ == "__main__":
    main()
