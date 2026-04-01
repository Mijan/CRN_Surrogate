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

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.dataset import TrajectoryItem
from crn_surrogate.data.generation.configs import GenerationConfig
from crn_surrogate.data.generation.curation import ViabilityFilter
from crn_surrogate.data.generation.motifs.base import MotifFactory
from crn_surrogate.data.generation.parameter_sampling import ParameterSampler
from crn_surrogate.data.generation.task import GenerationTask
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.interpolation import interpolate_to_grid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result of evaluating one parameter config against the viability filter.

    Attributes:
        item: The viable TrajectoryItem, or None if rejected.
        rejection_reason: The reason for rejection, or None if viable.
    """

    item: TrajectoryItem | None
    rejection_reason: str | None

    @property
    def viable(self) -> bool:
        """True if the config produced a viable TrajectoryItem."""
        return self.item is not None


@dataclass
class MotifResult:
    """Aggregated generation results for one motif type.

    Attributes:
        motif_label: The motif type string label.
        items: All viable TrajectoryItems collected.
        n_attempted: Total number of parameter configs evaluated.
        rejection_counts: Maps rejection reason to count.
    """

    motif_label: str
    items: list[TrajectoryItem]
    n_attempted: int
    rejection_counts: dict[str, int]

    @property
    def n_viable(self) -> int:
        """Number of viable items collected."""
        return len(self.items)

    @property
    def pass_rate(self) -> float:
        """Fraction of attempted configs that were viable."""
        if self.n_attempted == 0:
            return 0.0
        return self.n_viable / self.n_attempted

    def to_curation_stats(self) -> dict[str, int]:
        """Convert to the curation_stats dict format used by DatasetSummary.

        Returns:
            Dict with n_sampled, n_passed, n_rejected, and per-reason counts.
        """
        stats: dict[str, int] = {
            "n_sampled": self.n_attempted,
            "n_passed": self.n_viable,
            "n_rejected": self.n_attempted - self.n_viable,
        }
        stats.update(self.rejection_counts)
        return stats


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

    Accepts a list of GenerationTasks specifying which motifs to generate
    and how many of each. The pipeline is agnostic to whether tasks involve
    elementary or composed motifs.
    """

    def __init__(
        self,
        config: GenerationConfig,
        tasks: list[GenerationTask],
    ) -> None:
        """Args:
            config: Simulation, curation, and output configuration.
            tasks: What to generate. Each task specifies a factory, a
                target viable count, and a label.

        Raises:
            ValueError: If tasks is empty.
        """
        if not tasks:
            raise ValueError("tasks must be non-empty")
        self._config = config
        self._tasks = tasks
        self._sampler = ParameterSampler(config.sampling)
        self._filter = ViabilityFilter(config.curation)
        self._ssa = GillespieSSA()
        self._time_grid = torch.linspace(
            0.0, config.simulation_time, config.n_timepoints
        )

    def run(self) -> DatasetSummary:
        """Execute the full pipeline: generate, curate, balance, save.

        Returns:
            DatasetSummary with counts, curation statistics, and cluster ID mapping.
        """
        motif_results = self._generate_all_tasks()
        all_items = self._collect_items(motif_results)
        all_items = self._balance(all_items)
        cluster_id_map = self._assign_cluster_ids(all_items)
        self._save(all_items)
        summary = self._build_summary(all_items, motif_results, cluster_id_map)
        self._save_metadata(summary)
        return summary

    # --- Per-task generation ---

    def _generate_all_tasks(self) -> list[MotifResult]:
        """Generate viable items for every task.

        Returns:
            List of MotifResult, one per task.
        """
        return [
            self._generate_motif(task.factory, target=task.target, label=task.label)
            for task in self._tasks
        ]

    def _generate_motif(
        self,
        factory: MotifFactory,
        target: int,
        label: str,
    ) -> MotifResult:
        """Sample parameters until target viable items are collected.

        Samples in batches of config.batch_size. Stops early if
        target * max_attempts_multiplier total attempts are exhausted.

        Args:
            factory: The motif factory to generate configs for.
            target: Desired number of viable items.
            label: Label string for this motif in the dataset metadata.

        Returns:
            MotifResult with viable items and curation statistics.
        """
        max_attempts = target * self._config.max_attempts_multiplier
        viable_items: list[TrajectoryItem] = []
        n_attempted = 0
        rejection_counts: dict[str, int] = {}

        while len(viable_items) < target and n_attempted < max_attempts:
            batch_size = min(
                self._config.batch_size,
                target - len(viable_items),
                max_attempts - n_attempted,
            )
            params_batch = self._sampler.sample(factory, n_samples=batch_size)
            for params in params_batch:
                n_attempted += 1
                outcome = self._evaluate_config(factory, params)
                if outcome.viable:
                    viable_items.append(outcome.item)  # type: ignore[arg-type]
                else:
                    reason = outcome.rejection_reason or "unknown"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                if len(viable_items) >= target:
                    break

        self._log_motif_progress(label, len(viable_items), target, n_attempted)
        return MotifResult(
            motif_label=label,
            items=viable_items,
            n_attempted=n_attempted,
            rejection_counts=rejection_counts,
        )

    # --- Single-config evaluation ---

    def _evaluate_config(
        self, factory: MotifFactory, params: object
    ) -> EvaluationOutcome:
        """Simulate one parameter config and check viability.

        Args:
            factory: Motif factory used to build the CRN.
            params: Typed params instance for this config.

        Returns:
            EvaluationOutcome with a viable TrajectoryItem or a rejection reason.
        """
        crn = factory.create(params)  # type: ignore[arg-type]
        initial_state = self._sample_initial_state(factory)
        try:
            trajectories = self._simulate_ensemble(crn, initial_state)
        except RuntimeError:
            return EvaluationOutcome(item=None, rejection_reason="simulation_error")
        curation_result = self._filter.check(trajectories)
        if not curation_result.viable:
            return EvaluationOutcome(
                item=None, rejection_reason=curation_result.rejection_reason
            )
        item = self._build_trajectory_item(
            crn=crn,
            params=params,
            initial_state=initial_state,
            trajectories=trajectories,
            motif_label=factory.motif_type.value,
        )
        return EvaluationOutcome(item=item, rejection_reason=None)

    # --- Simulation ---

    def _simulate_ensemble(self, crn: CRN, initial_state: torch.Tensor) -> torch.Tensor:
        """Run M independent SSA simulations, interpolated to the time grid.

        Args:
            crn: CRN to simulate.
            initial_state: (n_species,) initial molecule counts.

        Returns:
            (M, T, n_species) trajectory tensor.
        """
        trajectories = [
            self._simulate_single(crn, initial_state)
            for _ in range(self._config.n_ssa_trajectories)
        ]
        return torch.stack(trajectories, dim=0)

    def _simulate_single(self, crn: CRN, initial_state: torch.Tensor) -> torch.Tensor:
        """Run one SSA simulation and interpolate to the time grid.

        Args:
            crn: CRN to simulate.
            initial_state: (n_species,) initial molecule counts.

        Returns:
            (T, n_species) interpolated trajectory.
        """
        result = self._ssa.simulate(
            stoichiometry=crn.stoichiometry_matrix,
            propensity_fn=crn.evaluate_propensities,
            initial_state=initial_state.clone(),
            t_max=self._config.simulation_time,
        )
        return interpolate_to_grid(result.times, result.states, self._time_grid)

    # --- Initial state ---

    def _sample_initial_state(self, factory: MotifFactory) -> torch.Tensor:
        """Sample one initial state and convert to a float32 tensor.

        Args:
            factory: Motif factory providing species names and state ranges.

        Returns:
            (n_species,) float32 tensor of initial molecule counts.
        """
        state_dict = self._sampler.sample_initial_states(factory, n_samples=1)[0]
        return torch.tensor(
            [float(state_dict[name]) for name in factory.species_names],
            dtype=torch.float32,
        )

    # --- Item construction ---

    def _build_trajectory_item(
        self,
        crn: CRN,
        params: object,
        initial_state: torch.Tensor,
        trajectories: torch.Tensor,
        motif_label: str,
    ) -> TrajectoryItem:
        """Wrap simulation results into a TrajectoryItem for storage.

        Args:
            crn: The CRN instance that was simulated.
            params: Typed params dataclass used to create this CRN.
            initial_state: (n_species,) initial molecule counts.
            trajectories: (M, T, n_species) SSA trajectories.
            motif_label: Motif type string label.

        Returns:
            TrajectoryItem with cluster_id=-1 (assigned later).
        """
        crn_repr = crn_to_tensor_repr(crn, max_params=8)
        return TrajectoryItem(
            crn_repr=crn_repr,
            initial_state=initial_state,
            trajectories=trajectories,
            times=self._time_grid,
            motif_label=motif_label,
            cluster_id=-1,
            params=dataclasses.asdict(params),  # type: ignore[call-overload]
        )

    # --- Post-processing ---

    def _collect_items(self, results: list[MotifResult]) -> list[TrajectoryItem]:
        """Flatten all viable items from per-motif results.

        Args:
            results: Per-motif generation results.

        Returns:
            Flat list of all viable TrajectoryItems.
        """
        return [item for result in results for item in result.items]

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

    def _assign_cluster_ids(self, items: list[TrajectoryItem]) -> dict[str, int]:
        """Assign integer cluster IDs to each item based on motif label.

        Args:
            items: All items to assign cluster IDs to (mutated in-place).

        Returns:
            Mapping from motif label to cluster ID.
        """
        labels = sorted({item.motif_label for item in items})
        cluster_map = {label: i for i, label in enumerate(labels)}
        for item in items:
            item.cluster_id = cluster_map[item.motif_label]
        return cluster_map

    # --- Persistence ---

    def _save(self, items: list[TrajectoryItem]) -> None:
        """Save the dataset to output_dir/dataset.pt.

        Args:
            items: Balanced, cluster-assigned items to save.
        """
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(items, output_dir / "dataset.pt")

    def _save_metadata(self, summary: DatasetSummary) -> None:
        """Write summary metadata to output_dir/metadata.json.

        Args:
            summary: Completed DatasetSummary to serialize.
        """
        output_dir = Path(self._config.output_dir)
        meta = {
            "total_items": summary.total_items,
            "counts_per_motif": summary.counts_per_motif,
            "cluster_id_map": summary.cluster_id_map,
            "curation_stats": summary.curation_stats,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _build_summary(
        self,
        items: list[TrajectoryItem],
        motif_results: list[MotifResult],
        cluster_id_map: dict[str, int],
    ) -> DatasetSummary:
        """Construct the DatasetSummary from generation results.

        Args:
            items: Balanced, cluster-assigned items.
            motif_results: Per-motif generation results.
            cluster_id_map: Motif label to cluster ID mapping.

        Returns:
            Populated DatasetSummary.
        """
        counts_per_motif = dict(Counter(item.motif_label for item in items))
        curation_stats = {r.motif_label: r.to_curation_stats() for r in motif_results}
        return DatasetSummary(
            total_items=len(items),
            counts_per_motif=counts_per_motif,
            curation_stats=curation_stats,
            cluster_id_map=cluster_id_map,
        )

    # --- Logging ---

    def _log_motif_progress(
        self,
        label: str,
        n_viable: int,
        target: int,
        n_attempted: int,
    ) -> None:
        """Log completion status for one motif.

        Args:
            label: Motif type string label.
            n_viable: Number of viable items collected.
            target: Target number of viable items.
            n_attempted: Total configs evaluated.
        """
        pass_rate = 100 * n_viable / n_attempted if n_attempted > 0 else 0.0
        if n_viable < target:
            logger.warning(
                "%s: only %d/%d viable after %d attempts (%.1f%% pass rate)",
                label,
                n_viable,
                target,
                n_attempted,
                pass_rate,
            )
        else:
            logger.info(
                "%s: %d/%d viable in %d attempts (%.1f%% pass rate)",
                label,
                n_viable,
                target,
                n_attempted,
                pass_rate,
            )
