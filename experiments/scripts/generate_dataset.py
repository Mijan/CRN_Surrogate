"""Generate a mass-action CRN dataset and log it as a W&B artifact.

Usage:
    python experiments/scripts/generate_dataset.py [--output-dir DIR] [--no-wandb] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.configs import CurationConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.mass_action_generator import MassActionCRNGenerator
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from experiments.configs.mass_action_3s import MassAction3sConfig


def _generate_split(
    gen: MassActionCRNGenerator,
    ssa: GillespieSSA,
    time_grid: torch.Tensor,
    cfg: MassAction3sConfig,
    n_items: int,
) -> tuple[list[TrajectoryItem], dict]:
    """Generate one dataset split, returning (items, metadata_dict).

    Args:
        gen: Configured CRN generator.
        ssa: Gillespie SSA simulator.
        time_grid: (T,) shared time grid for all trajectories.
        cfg: Experiment configuration.
        n_items: Target number of items to generate.

    Returns:
        Tuple of (items, stats_dict).
    """
    items: list[TrajectoryItem] = []
    stats: dict = {
        "n_attempted": 0,
        "n_curated_pass": 0,
        "n_species_dist": {},
        "n_reactions_dist": {},
    }
    filter_ = ViabilityFilter(CurationConfig())
    max_attempts = n_items * 5

    while len(items) < n_items and stats["n_attempted"] < max_attempts:
        crn = gen.sample()
        init_state = gen.sample_initial_state(
            crn,
            mean_molecules=cfg.dataset.initial_state_mean,
            spread=cfg.dataset.initial_state_spread,
        )
        crn_repr = crn_to_tensor_repr(crn)

        trajectories_list = ssa.simulate_batch(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init_state,
            t_max=cfg.dataset.t_max,
            n_trajectories=cfg.dataset.n_ssa_trajectories,
            n_workers=min(cfg.dataset.n_workers, cfg.dataset.n_ssa_trajectories),
        )
        traj_tensor = Trajectory.stack_on_grid(trajectories_list, time_grid)

        stats["n_attempted"] += 1

        result = filter_.check(traj_tensor)
        if not result.viable:
            continue

        stats["n_curated_pass"] += 1
        items.append(
            TrajectoryItem(
                crn_repr=crn_repr,
                initial_state=init_state,
                trajectories=traj_tensor,
                times=time_grid,
                motif_label="mass_action",
            )
        )

        ns = str(crn.n_species)
        nr = str(crn.n_reactions)
        stats["n_species_dist"][ns] = stats["n_species_dist"].get(ns, 0) + 1
        stats["n_reactions_dist"][nr] = stats["n_reactions_dist"].get(nr, 0) + 1

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
    cfg: MassAction3sConfig,
    output_dir: Path,
    *,
    use_wandb: bool,
    seed: int,
) -> None:
    """Run dataset generation and optionally log as a W&B artifact.

    Args:
        cfg: Experiment configuration.
        output_dir: Directory to write dataset files.
        use_wandb: Whether to log an artifact to W&B.
        seed: Random seed for reproducibility.
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

    print(f"Generating {cfg.dataset.n_train} training items...")
    train_items, train_meta = _generate_split(gen, ssa, time_grid, cfg, cfg.dataset.n_train)

    print(f"Generating {cfg.dataset.n_val} validation items...")
    val_items, val_meta = _generate_split(gen, ssa, time_grid, cfg, cfg.dataset.n_val)

    output_dir.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Generate a mass-action CRN dataset.")
    parser.add_argument("--output-dir", default="experiments/datasets")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = MassAction3sConfig()
    generate(
        cfg,
        output_dir=Path(args.output_dir),
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
