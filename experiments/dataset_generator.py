"""Dataset generation runner for CRN surrogate."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.configs import CurationConfig, ODEPreScreenConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.mass_action_generator import MassActionCRNGenerator
from crn_surrogate.data.generation.ode_prescreen import ODEPreScreen
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.data_simulator import DataSimulator
from experiments.builders import build_data_simulator, build_dataset_generator_config
from experiments.experiment_context import ExperimentContext
from experiments.wandb_session import WandbSession


class DatasetGenerator:
    """Generates train/val CRN trajectory datasets.

    All configuration is absorbed at construction via ExperimentContext.
    The run() method executes generation end-to-end.
    """

    def __init__(self, ctx: ExperimentContext) -> None:
        """Args:
        ctx: Shared experiment context holding config and utilities.
        """
        self._ctx = ctx
        self._cfg = ctx.cfg
        self._gen_cfg = ctx.cfg.generation
        self._ds_cfg = ctx.cfg.dataset

    def run(self) -> None:
        """Execute dataset generation end-to-end."""
        torch.manual_seed(self._cfg.seed)
        simulator = build_data_simulator(self._cfg)

        with self._ctx.wandb_session("data-generation", "data") as session:
            gen = MassActionCRNGenerator(build_dataset_generator_config(self._cfg))
            time_grid = torch.linspace(
                0.0, self._ds_cfg.t_max, self._ds_cfg.n_time_points
            )
            output_dir = Path(self._gen_cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            prescreen = self._build_prescreen()

            # Train split
            resume_train = self._resolve_resume("train")
            print(f"Generating {self._ds_cfg.n_train} training items...")
            train_items, train_meta = self._generate_split(
                gen,
                simulator,
                time_grid,
                session,
                n_items=self._ds_cfg.n_train,
                split_name="train",
                output_dir=output_dir,
                prescreen=prescreen,
                resume_items=resume_train,
            )

            # Val split
            resume_val = self._resolve_resume("val")
            print(f"Generating {self._ds_cfg.n_val} validation items...")
            val_items, val_meta = self._generate_split(
                gen,
                simulator,
                time_grid,
                session,
                n_items=self._ds_cfg.n_val,
                split_name="val",
                output_dir=output_dir,
                prescreen=prescreen,
                resume_items=resume_val,
            )

            self._save(
                output_dir, train_items, val_items, train_meta, val_meta, session
            )

    def _resolve_resume(self, split: str) -> list[TrajectoryItem] | None:
        """Resolve resume checkpoint for a split, returning loaded items or None.

        Args:
            split: "train" or "val".

        Returns:
            Loaded items list, or None if no resume path configured.
        """
        ref = getattr(self._gen_cfg, f"resume_{split}", None)
        if not ref:
            return None
        path = self._ctx.resolve_checkpoint(
            ref,
            artifact_name=f"{self._cfg.experiment_name}_{split}_checkpoint",
        )
        if path is None:
            return None
        loaded = torch.load(path, weights_only=False)
        items = list(loaded)
        print(f"Resumed {split}: {len(items)} items from {path.name}")
        return items

    def _build_prescreen(self) -> ODEPreScreen | None:
        """Build the ODE pre-screener if enabled, else return None."""
        if not self._gen_cfg.use_ode_prescreen:
            return None
        print("ODE pre-screening enabled")
        return ODEPreScreen(ODEPreScreenConfig(t_max=self._ds_cfg.t_max))

    def _generate_split(
        self,
        gen: MassActionCRNGenerator,
        simulator: DataSimulator,
        time_grid: torch.Tensor,
        session: WandbSession,
        *,
        n_items: int,
        split_name: str,
        output_dir: Path,
        prescreen: ODEPreScreen | None,
        resume_items: list[TrajectoryItem] | None,
    ) -> tuple[list[TrajectoryItem], dict]:
        """Generate one dataset split.

        Args:
            gen: Configured CRN generator.
            simulator: DataSimulator for trajectory generation.
            time_grid: (T,) shared time grid.
            session: Active WandbSession for progress logging.
            n_items: Target number of accepted items.
            split_name: "train" or "val" (used for checkpoint filenames).
            output_dir: Directory for intermediate checkpoints.
            prescreen: Optional ODE pre-screener.
            resume_items: Pre-existing items to start from.

        Returns:
            Tuple of (items, stats_dict).
        """
        items: list[TrajectoryItem] = list(resume_items) if resume_items else []
        if items:
            print(f"  Resuming from {len(items)} existing items")

        checkpoint_every = self._gen_cfg.checkpoint_every
        n_init_conditions = self._gen_cfg.n_init_conditions
        n_trajs_per_init = max(1, self._ds_cfg.n_ssa_trajectories // n_init_conditions)

        stats: dict = {
            "n_attempted": 0,
            "n_curated_pass": 0,
            "n_timeout": 0,
            "n_species_dist": {},
            "n_reactions_dist": {},
        }
        filter_ = ViabilityFilter(CurationConfig())
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
                    mean_molecules=self._ds_cfg.initial_state_mean,
                    spread=self._ds_cfg.initial_state_spread,
                )
                stats["n_attempted"] += 1

                if prescreen is not None:
                    result = prescreen.check(crn, init_state)
                    if not result.accepted:
                        stats.setdefault("n_ode_rejected", 0)
                        stats["n_ode_rejected"] += 1
                        continue
                    stats.setdefault("n_ode_accepted", 0)
                    stats["n_ode_accepted"] += 1

                traj_tensor = simulator.simulate(
                    crn, init_state, self._ds_cfg.t_max, n_trajs_per_init, time_grid
                )
                if traj_tensor is None:
                    stats["n_timeout"] += 1
                    continue

                if not filter_.check(traj_tensor).viable:
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
                            "data/pass_rate": stats["n_curated_pass"]
                            / stats["n_attempted"],
                        }
                    )

                if checkpoint_every > 0 and len(items) % checkpoint_every == 0:
                    self._save_checkpoint(items, split_name, output_dir, session)

        pbar.close()
        stats["pass_rate"] = stats["n_curated_pass"] / max(stats["n_attempted"], 1)
        self._print_stats(n_items, items, stats)

        if len(items) < n_items:
            warnings.warn(
                f"Only generated {len(items)}/{n_items} items.",
                RuntimeWarning,
                stacklevel=2,
            )
        return items, stats

    def _save_checkpoint(
        self,
        items: list[TrajectoryItem],
        split_name: str,
        output_dir: Path,
        session: WandbSession,
    ) -> None:
        """Save intermediate checkpoint to disk and W&B.

        Args:
            items: Current list of accepted items.
            split_name: "train" or "val".
            output_dir: Directory to write the checkpoint file.
            session: Active WandbSession for artifact logging.
        """
        name = f"{self._cfg.experiment_name}_{split_name}"
        path = output_dir / f"{name}_checkpoint_{len(items)}.pt"
        torch.save(CRNTrajectoryDataset(items), path)
        print(f"  Checkpoint: {len(items)} items -> {path.name}")
        session.log_artifact(
            f"{name}_checkpoint",
            "dataset-checkpoint",
            path,
            metadata={"n_items": len(items)},
        )

    def _save(
        self,
        output_dir: Path,
        train_items: list[TrajectoryItem],
        val_items: list[TrajectoryItem],
        train_meta: dict,
        val_meta: dict,
        session: WandbSession,
    ) -> None:
        """Save final datasets and metadata to disk and W&B.

        Args:
            output_dir: Directory to write output files.
            train_items: Accepted training items.
            val_items: Accepted validation items.
            train_meta: Generation statistics for the train split.
            val_meta: Generation statistics for the val split.
            session: Active WandbSession for artifact logging.
        """
        name = self._cfg.experiment_name
        train_path = output_dir / f"{name}_train.pt"
        val_path = output_dir / f"{name}_val.pt"
        meta_path = output_dir / f"{name}_meta.json"

        torch.save(CRNTrajectoryDataset(train_items), train_path)
        torch.save(CRNTrajectoryDataset(val_items), val_path)

        metadata = {
            "experiment": name,
            "seed": self._cfg.seed,
            "n_train": len(train_items),
            "n_val": len(val_items),
            "train_meta": train_meta,
            "val_meta": val_meta,
            "config": self._ctx.flat_config,
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"Saved: {train_path}, {val_path}, {meta_path}")

        session.log_multi_file_artifact(
            f"{name}_dataset",
            "dataset",
            [train_path, val_path, meta_path],
            metadata=metadata,
        )

    @staticmethod
    def _print_stats(n_target: int, items: list, stats: dict) -> None:
        """Print generation summary statistics.

        Args:
            n_target: Target number of items requested.
            items: Accepted items list.
            stats: Stats dict from the generation loop.
        """
        print(
            f"  Generated {len(items)}/{n_target} items "
            f"({stats['n_attempted']} attempted, {stats['pass_rate']:.0%} pass rate, "
            f"{stats['n_timeout']} timeouts)"
        )
        print(f"  Species distribution:   {stats['n_species_dist']}")
        print(f"  Reactions distribution: {stats['n_reactions_dist']}")
        if "n_ode_rejected" in stats:
            print(
                f"  ODE pre-screen: {stats.get('n_ode_accepted', 0)} accepted, "
                f"{stats['n_ode_rejected']} rejected"
            )
