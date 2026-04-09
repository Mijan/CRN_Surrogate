"""Generate a CRN dataset.

Usage:
    python experiments/scripts/generate_dataset.py
    python experiments/scripts/generate_dataset.py experiment=mass_action_3s_v7
    python experiments/scripts/generate_dataset.py dataset.n_train=100000
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.configs import CurationConfig, ODEPreScreenConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.mass_action_generator import MassActionCRNGenerator
from crn_surrogate.data.generation.ode_prescreen import ODEPreScreen
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.data_simulator import DataSimulator
from experiments.builders import build_data_simulator, build_dataset_generator_config
from experiments.wandb_session import WandbSession


def _make_checkpoint_fn(
    output_dir: Path,
    experiment_name: str,
    split_name: str,
    session: WandbSession,
) -> Callable[[list[TrajectoryItem], str], None]:
    """Create a checkpoint callback that saves intermediate items to disk.

    Args:
        output_dir: Directory to write checkpoint files.
        experiment_name: Used as filename prefix.
        split_name: "train" or "val".
        session: Active WandbSession for logging (no-op if inactive).

    Returns:
        Callable that accepts (items, label) and saves a CRNTrajectoryDataset.
    """

    def _checkpoint(items: list[TrajectoryItem], label: str) -> None:
        path = output_dir / f"{experiment_name}_{split_name}_{label}.pt"
        torch.save(CRNTrajectoryDataset(items), path)
        print(f"  Checkpoint: {len(items)} items -> {path.name}")
        session.log_artifact(
            f"{experiment_name}_{split_name}_checkpoint",
            "dataset-checkpoint",
            path,
            metadata={"n_items": len(items), "label": label},
        )
        session.log({f"data/{split_name}_items": len(items)})

    return _checkpoint


def _generate_split(
    gen: MassActionCRNGenerator,
    simulator: DataSimulator,
    time_grid: torch.Tensor,
    *,
    n_items: int,
    n_trajectories: int,
    initial_state_mean: float,
    initial_state_spread: float,
    t_max: float,
    n_init_conditions: int = 1,
    session: WandbSession,
    checkpoint_fn: Callable[[list[TrajectoryItem], str], None] | None = None,
    checkpoint_every: int = 50,
    resume_items: list[TrajectoryItem] | None = None,
    ode_prescreen: ODEPreScreen | None = None,
) -> tuple[list[TrajectoryItem], dict]:
    """Generate one dataset split, returning (items, metadata_dict).

    Args:
        gen: Configured CRN generator.
        simulator: DataSimulator instance to produce trajectories.
        time_grid: (T,) shared time grid for all trajectories.
        n_items: Target number of items to generate.
        n_trajectories: Trajectories per CRN (split across init conditions).
        initial_state_mean: Geometric mean of initial molecule counts.
        initial_state_spread: Geometric standard deviation for initial states.
        t_max: Simulation end time.
        n_init_conditions: Number of distinct initial conditions per CRN topology.
        session: WandbSession for logging progress metrics.
        checkpoint_fn: Optional callback called every checkpoint_every items.
        checkpoint_every: Checkpoint interval in number of accepted items.
        resume_items: Pre-existing items to resume from.
        ode_prescreen: Optional pre-screener; rejects boring dynamics before SSA.

    Returns:
        Tuple of (items, stats_dict).
    """
    items: list[TrajectoryItem] = list(resume_items) if resume_items else []
    if items:
        print(f"  Resuming from {len(items)} existing items")

    stats: dict = {
        "n_attempted": 0,
        "n_curated_pass": 0,
        "n_timeout": 0,
        "n_species_dist": {},
        "n_reactions_dist": {},
    }
    filter_ = ViabilityFilter(CurationConfig())
    n_trajs_per_init = max(1, n_trajectories // n_init_conditions)
    max_attempts = n_items * 10

    pbar = tqdm(total=n_items, initial=len(items), desc="generating", unit="item")

    while len(items) < n_items and stats["n_attempted"] < max_attempts:
        crn = gen.sample()
        crn_repr = crn_to_tensor_repr(crn)

        for _ in range(n_init_conditions):
            if len(items) >= n_items:
                break

            init_state = gen.sample_initial_state(
                crn,
                mean_molecules=initial_state_mean,
                spread=initial_state_spread,
            )

            stats["n_attempted"] += 1

            if ode_prescreen is not None:
                prescreen_result = ode_prescreen.check(crn, init_state)
                if not prescreen_result.accepted:
                    stats.setdefault("n_ode_rejected", 0)
                    stats["n_ode_rejected"] += 1
                    pbar.set_postfix(
                        ode_rej=stats["n_ode_rejected"],
                        rate=f"{stats['n_curated_pass']}/{stats['n_attempted']}",
                    )
                    continue
                stats.setdefault("n_ode_accepted", 0)
                stats["n_ode_accepted"] += 1

            traj_tensor = simulator.simulate(
                crn, init_state, t_max, n_trajs_per_init, time_grid
            )
            if traj_tensor is None:
                stats["n_timeout"] += 1
                pbar.set_postfix(timeouts=stats["n_timeout"])
                continue

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
                    motif_label="mass_action",
                )
            )

            ns = str(crn.n_species)
            nr = str(crn.n_reactions)
            stats["n_species_dist"][ns] = stats["n_species_dist"].get(ns, 0) + 1
            stats["n_reactions_dist"][nr] = stats["n_reactions_dist"].get(nr, 0) + 1

            pbar.update(1)
            pbar.set_postfix(
                pass_rate=f"{stats['n_curated_pass'] / stats['n_attempted']:.0%}",
                species=ns,
            )

            if session.active and len(items) % 10 == 0:
                session.log(
                    {
                        "data/items_generated": len(items),
                        "data/attempts": stats["n_attempted"],
                        "data/pass_rate": stats["n_curated_pass"]
                        / stats["n_attempted"],
                    }
                )

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
        f"({stats['n_attempted']} attempted, {stats['pass_rate']:.0%} pass rate, "
        f"{stats['n_timeout']} timeouts)"
    )
    print(f"  Species distribution:   {stats['n_species_dist']}")
    print(f"  Reactions distribution: {stats['n_reactions_dist']}")
    if "n_ode_rejected" in stats:
        print(
            f"  ODE pre-screen: {stats.get('n_ode_accepted', 0)} accepted, "
            f"{stats['n_ode_rejected']} rejected "
            f"({stats['n_ode_rejected'] / stats['n_attempted']:.0%} rejection rate)"
        )

    if len(items) < n_items:
        warnings.warn(
            f"Only generated {len(items)}/{n_items} items after {stats['n_attempted']} attempts.",
            RuntimeWarning,
            stacklevel=2,
        )

    return items, stats


def generate(
    cfg: DictConfig,
    *,
    simulator: DataSimulator,
    session: WandbSession,
    output_dir: Path,
    seed: int,
    checkpoint_every: int,
    n_init_conditions: int = 1,
    use_ode_prescreen: bool = True,
    resume_train: Path | None = None,
    resume_val: Path | None = None,
) -> None:
    """Run dataset generation and log as a W&B artifact via the session.

    Args:
        cfg: Fully resolved Hydra config.
        simulator: DataSimulator to use for trajectory generation.
        session: WandbSession for artifact logging (no-op if inactive).
        output_dir: Directory to write dataset files.
        seed: Random seed for reproducibility.
        checkpoint_every: Save intermediate dataset every N accepted items.
        n_init_conditions: Number of distinct initial conditions per CRN topology.
        use_ode_prescreen: Whether to run ODE pre-screening to reject boring dynamics.
        resume_train: Optional checkpoint path to resume training split from.
        resume_val: Optional checkpoint path to resume validation split from.
    """
    torch.manual_seed(seed)

    generator_config = build_dataset_generator_config(cfg)
    gen = MassActionCRNGenerator(generator_config)
    time_grid = torch.linspace(0.0, cfg.dataset.t_max, cfg.dataset.n_time_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    ode_prescreen = None
    if use_ode_prescreen:
        ode_prescreen = ODEPreScreen(ODEPreScreenConfig(t_max=cfg.dataset.t_max))
        print("ODE pre-screening enabled")

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
        gen,
        simulator,
        time_grid,
        n_items=cfg.dataset.n_train,
        n_trajectories=cfg.dataset.n_ssa_trajectories,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        n_init_conditions=n_init_conditions,
        session=session,
        checkpoint_fn=_make_checkpoint_fn(
            output_dir, cfg.experiment_name, "train", session
        ),
        checkpoint_every=checkpoint_every,
        resume_items=resume_train_items,
        ode_prescreen=ode_prescreen,
    )

    print(f"Generating {cfg.dataset.n_val} validation items...")
    val_items, val_meta = _generate_split(
        gen,
        simulator,
        time_grid,
        n_items=cfg.dataset.n_val,
        n_trajectories=cfg.dataset.n_ssa_trajectories,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        n_init_conditions=n_init_conditions,
        session=session,
        checkpoint_fn=_make_checkpoint_fn(
            output_dir, cfg.experiment_name, "val", session
        ),
        checkpoint_every=checkpoint_every,
        resume_items=resume_val_items,
        ode_prescreen=ode_prescreen,
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
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved: {train_path}, {val_path}, {meta_path}")

    session.log_multi_file_artifact(
        f"{cfg.experiment_name}_dataset",
        "dataset",
        [train_path, val_path, meta_path],
        metadata=metadata,
    )
    if session.active:
        print(f"Logged W&B artifact: {cfg.experiment_name}_dataset")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for dataset generation."""
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    gen_cfg = cfg.generation
    flat_config: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )
    simulator = build_data_simulator(cfg)

    with WandbSession(
        project=cfg.wandb_project,
        name=f"{cfg.experiment_name}_data",
        group=cfg.wandb_group,
        job_type="data-generation",
        config=flat_config,
        enabled=not cfg.no_wandb,
    ) as session:
        generate(
            cfg,
            simulator=simulator,
            session=session,
            output_dir=Path(gen_cfg.generation.output_dir),
            seed=cfg.seed,
            checkpoint_every=gen_cfg.generation.checkpoint_every,
            n_init_conditions=gen_cfg.generation.n_init_conditions,
            use_ode_prescreen=gen_cfg.generation.use_ode_prescreen,
        )


if __name__ == "__main__":
    main()
