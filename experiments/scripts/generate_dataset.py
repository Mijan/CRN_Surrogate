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

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crn_surrogate.data.dataset import CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.configs import CurationConfig, ODEPreScreenConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.ode_prescreen import ODEPreScreen
from crn_surrogate.data.generation.mass_action_generator import MassActionCRNGenerator
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from experiments.builders import build_dataset_generator_config
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


def _simulate_with_timeout(
    ssa: GillespieSSA,
    timeout: int,
    **kwargs,
) -> list | None:
    """Run SSA ensemble with a wall-clock timeout. Returns None on timeout.

    Args:
        ssa: Configured Gillespie SSA simulator.
        timeout: Wall-clock timeout in seconds. 0 disables the timeout.
        **kwargs: Forwarded to ssa.simulate_batch.

    Returns:
        List of trajectories, or None if the timeout was exceeded.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    if timeout <= 0:
        return ssa.simulate_batch(**kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ssa.simulate_batch, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            return None


def _simulate_ode(
    crn,
    initial_state: torch.Tensor,
    t_max: float,
    time_grid: torch.Tensor,
    *,
    n_substeps: int = 10,
    blowup_threshold: float = 1e5,
) -> torch.Tensor | None:
    """Integrate the deterministic mass-action ODE via forward Euler.

    Produces a single deterministic trajectory. Returns a (1, T, n_species)
    tensor to match the stochastic TrajectoryItem format (M=1).

    Args:
        crn: The CRN to simulate.
        initial_state: (n_species,) initial molecule counts.
        t_max: End time (unused, derived from time_grid).
        time_grid: (T,) time points at which to record the state.
        n_substeps: Euler substeps between consecutive grid points.
        blowup_threshold: Abort and return None if any state exceeds this.

    Returns:
        (1, T, n_species) trajectory tensor, or None if blowup detected.
    """
    import numpy as np

    T = len(time_grid)
    n_species = initial_state.shape[0]
    stoich = crn.stoichiometry_matrix.numpy().T  # (n_species, n_reactions)

    x = initial_state.numpy().astype(np.float64).copy()
    recorded = np.zeros((T, n_species), dtype=np.float64)
    recorded[0] = x

    for t_idx in range(1, T):
        dt_segment = (time_grid[t_idx] - time_grid[t_idx - 1]).item()
        dt_sub = dt_segment / n_substeps

        for _ in range(n_substeps):
            x_clamped = np.maximum(x, 0.0)
            x_tensor = torch.tensor(x_clamped, dtype=torch.float32)
            props = crn.evaluate_propensities(x_tensor, 0.0).numpy()
            dx = stoich @ props
            x = x + dt_sub * dx
            x = np.maximum(x, 0.0)

            if np.any(x > blowup_threshold) or np.any(np.isnan(x)):
                return None

        recorded[t_idx] = x

    return torch.tensor(recorded, dtype=torch.float32).unsqueeze(0)


def _make_simulator(
    use_fast_ssa: bool,
    ssa: GillespieSSA,
    fast_ssa: object | None,
) -> Callable:
    """Return a callable (crn, init_state, t_max, n_traj, time_grid, timeout) -> Tensor | None.

    Args:
        use_fast_ssa: Whether to attempt the Numba-accelerated path.
        ssa: Standard Gillespie SSA simulator (always available as fallback).
        fast_ssa: FastMassActionSSA instance, or None if unavailable.

    Returns:
        Callable that runs SSA and returns a (n_traj, T, n_species) tensor,
        or None if the simulation times out.
    """
    from crn_surrogate.simulation.fast_ssa import FastMassActionSSA

    def _simulate_standard(crn, init_state, t_max, n_traj, time_grid, timeout):
        trajs = _simulate_with_timeout(
            ssa,
            timeout,
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=init_state,
            t_max=t_max,
            n_trajectories=n_traj,
            n_workers=1,
        )
        if trajs is None:
            return None
        return Trajectory.stack_on_grid(trajs, time_grid)

    def _simulate_fast(crn, init_state, t_max, n_traj, time_grid, timeout):
        try:
            arrays = FastMassActionSSA.extract_topology_arrays(crn)
        except (AttributeError, ValueError):
            return _simulate_standard(crn, init_state, t_max, n_traj, time_grid, timeout)

        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        if timeout <= 0:
            return fast_ssa.simulate_batch(
                stoichiometry=arrays["stoichiometry"],
                reactant_matrix=arrays["reactant_matrix"],
                rate_constants=arrays["rate_constants"],
                initial_state=init_state,
                t_max=t_max,
                time_grid=time_grid,
                n_trajectories=n_traj,
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                fast_ssa.simulate_batch,
                stoichiometry=arrays["stoichiometry"],
                reactant_matrix=arrays["reactant_matrix"],
                rate_constants=arrays["rate_constants"],
                initial_state=init_state,
                t_max=t_max,
                time_grid=time_grid,
                n_trajectories=n_traj,
            )
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout:
                return None

    if use_fast_ssa and fast_ssa is not None:
        return _simulate_fast
    return _simulate_standard


def _generate_split(
    gen: MassActionCRNGenerator,
    simulate_fn: Callable,
    time_grid: torch.Tensor,
    *,
    n_items: int,
    n_ssa_trajectories: int,
    initial_state_mean: float,
    initial_state_spread: float,
    t_max: float,
    sim_timeout: int = 30,
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
        simulate_fn: Callable (crn, init_state, t_max, n_traj, time_grid, timeout)
            -> (n_traj, T, n_species) tensor or None on timeout.
        time_grid: (T,) shared time grid for all trajectories.
        n_items: Target number of items to generate.
        n_ssa_trajectories: SSA trajectories per CRN (split across init conditions).
        initial_state_mean: Geometric mean of initial molecule counts.
        initial_state_spread: Geometric standard deviation for initial states.
        t_max: Simulation end time.
        sim_timeout: Per-CRN wall-clock timeout in seconds (0 to disable).
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
    n_trajs_per_init = max(1, n_ssa_trajectories // n_init_conditions)
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

            traj_tensor = simulate_fn(
                crn, init_state, t_max, n_trajs_per_init, time_grid, sim_timeout
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
                pass_rate=f"{stats['n_curated_pass']/stats['n_attempted']:.0%}",
                species=ns,
            )

            if session.active and len(items) % 10 == 0:
                session.log({
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
    session: WandbSession,
    output_dir: Path,
    seed: int,
    checkpoint_every: int,
    sim_timeout: int = 30,
    n_init_conditions: int = 1,
    use_fast_ssa: bool = True,
    use_ode_prescreen: bool = True,
    resume_train: Path | None = None,
    resume_val: Path | None = None,
    deterministic: bool = False,
) -> None:
    """Run dataset generation and log as a W&B artifact via the session.

    Args:
        cfg: Fully resolved Hydra config.
        session: WandbSession for artifact logging (no-op if inactive).
        output_dir: Directory to write dataset files.
        seed: Random seed for reproducibility.
        checkpoint_every: Save intermediate dataset every N accepted items.
        sim_timeout: Per-CRN wall-clock timeout in seconds (0 to disable).
        n_init_conditions: Number of distinct initial conditions per CRN topology.
        use_fast_ssa: Whether to attempt the Numba-accelerated fast SSA path.
        use_ode_prescreen: Whether to run ODE pre-screening to reject boring dynamics.
        resume_train: Optional checkpoint path to resume training split from.
        resume_val: Optional checkpoint path to resume validation split from.
        deterministic: If True, generate deterministic ODE trajectories instead
            of stochastic SSA (M=1 per item).
    """
    torch.manual_seed(seed)

    generator_config = build_dataset_generator_config(cfg)
    gen = MassActionCRNGenerator(generator_config)
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, cfg.dataset.t_max, cfg.dataset.n_time_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_ssa = None
    if use_fast_ssa:
        try:
            from crn_surrogate.simulation.fast_ssa import (
                NUMBA_AVAILABLE,
                FastMassActionSSA,
                _gillespie_mass_action_inner,
            )
            if not NUMBA_AVAILABLE:
                print("Numba not available, using standard SSA")
                use_fast_ssa = False
            else:
                fast_ssa = FastMassActionSSA()
                print("Using Numba-accelerated SSA")
                print("Warming up Numba JIT (first call compiles)...")
                import numpy as np
                _gillespie_mass_action_inner(
                    np.zeros((2, 1), dtype=np.float64),
                    np.zeros((2, 1), dtype=np.float64),
                    np.ones(2, dtype=np.float64),
                    np.array([1.0], dtype=np.float64),
                    1.0, 100, 42,
                )
                print("JIT warm-up complete")
        except ImportError:
            print("fast_ssa module not available, using standard SSA")
            use_fast_ssa = False

    simulate_fn = _make_simulator(use_fast_ssa, ssa, fast_ssa)

    if deterministic:
        def det_simulate_fn(crn, init_state, t_max, n_traj, time_grid, timeout):
            return _simulate_ode(crn, init_state, t_max, time_grid)
        simulate_fn = det_simulate_fn
        print("Using deterministic ODE simulation (M=1 per item)")

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
        gen, simulate_fn, time_grid,
        n_items=cfg.dataset.n_train,
        n_ssa_trajectories=cfg.dataset.n_ssa_trajectories,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        sim_timeout=sim_timeout,
        n_init_conditions=n_init_conditions,
        session=session,
        checkpoint_fn=_make_checkpoint_fn(output_dir, cfg.experiment_name, "train", session),
        checkpoint_every=checkpoint_every,
        resume_items=resume_train_items,
        ode_prescreen=ode_prescreen,
    )

    print(f"Generating {cfg.dataset.n_val} validation items...")
    val_items, val_meta = _generate_split(
        gen, simulate_fn, time_grid,
        n_items=cfg.dataset.n_val,
        n_ssa_trajectories=cfg.dataset.n_ssa_trajectories,
        initial_state_mean=cfg.dataset.initial_state_mean,
        initial_state_spread=cfg.dataset.initial_state_spread,
        t_max=cfg.dataset.t_max,
        sim_timeout=sim_timeout,
        n_init_conditions=n_init_conditions,
        session=session,
        checkpoint_fn=_make_checkpoint_fn(output_dir, cfg.experiment_name, "val", session),
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
    use_wandb = not cfg.no_wandb
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    with WandbSession(
        project=cfg.wandb_project,
        name=f"{cfg.experiment_name}_data",
        group=cfg.wandb_group,
        job_type="data-generation",
        config=flat_config,
        enabled=use_wandb,
    ) as session:
        generate(
            cfg,
            session=session,
            output_dir=Path(gen_cfg.output_dir),
            seed=cfg.seed,
            checkpoint_every=gen_cfg.checkpoint_every,
            sim_timeout=gen_cfg.sim_timeout,
            n_init_conditions=gen_cfg.n_init_conditions,
            use_fast_ssa=gen_cfg.use_fast_ssa,
            use_ode_prescreen=gen_cfg.use_ode_prescreen,
            deterministic=cfg.solver.deterministic,
        )


if __name__ == "__main__":
    main()
