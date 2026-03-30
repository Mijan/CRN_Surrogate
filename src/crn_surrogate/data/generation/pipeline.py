"""End-to-end data generation pipeline for CRN surrogate training data."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from crn_surrogate.data.dataset import TrajectoryItem
from crn_surrogate.data.generation.configs import GenerationConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.motif_registry import get_factory
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory
from crn_surrogate.data.generation.parameter_sampling import ParameterSampler
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.interpolation import interpolate_to_grid

logger = logging.getLogger(__name__)


@dataclass
class DatasetSummary:
    """Summary statistics for a completed data generation run.

    Attributes:
        total_items: Total number of viable items saved to disk.
        counts_per_motif: Maps motif label to item count after balancing.
        curation_stats: Per-motif stats including n_sampled, n_passed, n_rejected,
            and per-criterion rejection counts.
        cluster_id_map: Maps motif label string to integer cluster ID.
    """

    total_items: int
    counts_per_motif: dict[str, int]
    curation_stats: dict[str, dict[str, int]]
    cluster_id_map: dict[str, int]


class DataGenerationPipeline:
    """Generates, curates, and saves CRN trajectory datasets.

    For each motif type, the pipeline:
        1. Samples kinetic parameters.
        2. Simulates an ensemble of SSA trajectories.
        3. Applies viability filters.
        4. Converts viable CRNs to tensor representations.
        5. Saves the dataset and metadata to output_dir.
    """

    def __init__(self, config: GenerationConfig) -> None:
        """Args:
        config: Top-level generation configuration.
        """
        self._config = config
        self._sampler = ParameterSampler(config.sampling)
        self._filter = ViabilityFilter(config.curation)
        self._ssa = GillespieSSA()

    def run(self) -> DatasetSummary:
        """Execute the full data generation pipeline.

        Returns:
            DatasetSummary with counts, curation statistics, and cluster ID mapping.
        """
        cfg = self._config
        all_items: list[TrajectoryItem] = []
        curation_stats: dict[str, dict[str, int]] = {}

        time_grid = torch.linspace(0.0, cfg.simulation_time, cfg.n_timepoints)

        for motif_type in MotifType:
            factory = get_factory(motif_type)
            label = motif_type.value
            param_list = self._sampler.sample(factory, cfg.n_samples_per_motif)

            stats: dict[str, int] = {
                "n_sampled": len(param_list),
                "n_passed": 0,
                "n_rejected": 0,
            }
            rejection_counts: dict[str, int] = {}

            for params in param_list:
                crn = factory.create(params)
                initial_state_dict = self._sampler.sample_initial_states(
                    factory, n_samples=1
                )[0]
                initial_state = self._initial_state_tensor(factory, initial_state_dict)

                trajs = self._simulate_batch(
                    crn,
                    initial_state,
                    time_grid,
                    cfg.n_ssa_trajectories,
                    cfg.simulation_time,
                )
                result = self._filter.check(trajs)

                if result.viable:
                    crn_repr = crn_to_tensor_repr(crn, max_params=8)
                    item = TrajectoryItem(
                        crn_repr=crn_repr,
                        initial_state=initial_state,
                        trajectories=trajs,
                        times=time_grid,
                        motif_label=label,
                        cluster_id=-1,
                        params=dataclasses.asdict(params),
                    )
                    all_items.append(item)
                    stats["n_passed"] += 1
                else:
                    stats["n_rejected"] += 1
                    rejection_counts[result.rejection_reason] = (
                        rejection_counts.get(result.rejection_reason, 0) + 1
                    )

            stats.update(rejection_counts)
            curation_stats[label] = stats
            logger.info(
                "%s: %d/%d viable", label, stats["n_passed"], stats["n_sampled"]
            )

        all_items = self._balance(all_items)

        labels = sorted({item.motif_label for item in all_items})
        cluster_id_map = {lbl: i for i, lbl in enumerate(labels)}

        for item in all_items:
            item.cluster_id = cluster_id_map[item.motif_label]

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(all_items, output_dir / "dataset.pt")

        counts_per_motif: dict[str, int] = {}
        for item in all_items:
            counts_per_motif[item.motif_label] = (
                counts_per_motif.get(item.motif_label, 0) + 1
            )

        summary = DatasetSummary(
            total_items=len(all_items),
            counts_per_motif=counts_per_motif,
            curation_stats=curation_stats,
            cluster_id_map=cluster_id_map,
        )

        meta = {
            "total_items": summary.total_items,
            "counts_per_motif": summary.counts_per_motif,
            "cluster_id_map": summary.cluster_id_map,
            "curation_stats": summary.curation_stats,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return summary

    def _simulate_batch(
        self,
        crn: object,
        initial_state: torch.Tensor,
        time_grid: torch.Tensor,
        n_traj: int,
        t_max: float,
    ) -> torch.Tensor:
        """Run n_traj independent SSA simulations and interpolate to the time grid.

        Args:
            crn: CRN instance with stoichiometry_matrix and evaluate_propensities.
            initial_state: (n_species,) initial molecule counts.
            time_grid: (T,) evenly-spaced output time points.
            n_traj: Number of independent SSA trajectories.
            t_max: Simulation end time.

        Returns:
            (n_traj, T, n_species) stacked trajectory tensor.
        """
        from crn_surrogate.crn.crn import CRN

        assert isinstance(crn, CRN)
        trajs = []
        for _ in range(n_traj):
            result = self._ssa.simulate(
                stoichiometry=crn.stoichiometry_matrix,
                propensity_fn=crn.evaluate_propensities,
                initial_state=initial_state.clone(),
                t_max=t_max,
            )
            traj = interpolate_to_grid(result.times, result.states, time_grid)
            trajs.append(traj)
        return torch.stack(trajs, dim=0)  # (M, T, n_species)

    def _initial_state_tensor(
        self,
        factory: MotifFactory,
        initial_state: dict[str, int],
    ) -> torch.Tensor:
        """Build the initial state tensor from a sampled initial state dict.

        Args:
            factory: MotifFactory providing ordered species names.
            initial_state: Dict mapping species name to initial molecule count.

        Returns:
            (n_species,) float32 tensor of initial molecule counts.
        """
        values = [float(initial_state[name]) for name in factory.species_names]
        return torch.tensor(values, dtype=torch.float32)

    def _balance(self, items: list[TrajectoryItem]) -> list[TrajectoryItem]:
        """Cap over-represented motif classes at 2x the median count.

        Args:
            items: All viable items before balancing.

        Returns:
            Balanced list of items with per-class caps applied.
        """
        rng = random.Random(self._config.random_seed)
        counts = Counter(item.motif_label for item in items)
        if not counts:
            return items

        sorted_counts = sorted(counts.values())
        median_count = sorted_counts[len(sorted_counts) // 2]
        cap = 2 * median_count

        for label, count in counts.items():
            if count < 50:
                logger.warning(
                    "Motif %r has only %d viable configs (< 50)", label, count
                )

        by_label: dict[str, list[TrajectoryItem]] = {}
        for item in items:
            by_label.setdefault(item.motif_label, []).append(item)

        result: list[TrajectoryItem] = []
        for label_items in by_label.values():
            if len(label_items) > cap:
                label_items = rng.sample(label_items, cap)
            result.extend(label_items)
        return result
