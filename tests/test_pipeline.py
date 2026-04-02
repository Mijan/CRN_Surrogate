"""Tests for DataGenerationPipeline: sample-until-viable, decomposition, and integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from crn_surrogate.data.dataset import TrajectoryItem
from crn_surrogate.data.generation.configs import (
    CurationConfig,
    GenerationConfig,
    SamplingConfig,
)
from crn_surrogate.data.generation.motif_registry import get_factory
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory
from crn_surrogate.data.generation.motifs.birth_death import BirthDeathFactory
from crn_surrogate.data.generation.pipeline import (
    DataGenerationPipeline,
    EvaluationOutcome,
    MotifResult,
)
from crn_surrogate.data.generation.task import GenerationTask

# --- Fixtures -------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary directory for pipeline output."""
    return tmp_path / "generated"


@pytest.fixture
def fast_config(output_dir: Path) -> GenerationConfig:
    """Minimal GenerationConfig for fast integration tests."""
    return GenerationConfig(
        sampling=SamplingConfig(random_seed=7),
        curation=CurationConfig(
            blowup_threshold=1e7,
            min_coefficient_of_variation=0.001,
            max_zero_fraction=0.99,
            min_reactions_fired=1,
            max_final_population=1e6,
        ),
        n_ssa_trajectories=4,
        simulation_time=10.0,
        n_timepoints=20,
        batch_size=20,
        max_attempts_multiplier=5,
        output_dir=str(output_dir),
        random_seed=7,
    )


# --- Test-only pipeline subclasses ----------------------------------------------


class _AlwaysRejectPipeline(DataGenerationPipeline):
    """Pipeline that rejects every config, for testing the max-attempts cap."""

    def _evaluate_config(
        self, factory: MotifFactory, params: object, motif_label: str = ""
    ) -> EvaluationOutcome:
        return EvaluationOutcome(item=None, rejection_reason="always_rejected")


class _AlternatingPipeline(DataGenerationPipeline):
    """Pipeline that accepts every other config, for testing sample-until-viable."""

    def __init__(self, config: GenerationConfig, tasks: list[GenerationTask]) -> None:
        super().__init__(config, tasks)
        self._call_count = 0

    def _evaluate_config(
        self, factory: MotifFactory, params: object, motif_label: str = ""
    ) -> EvaluationOutcome:
        self._call_count += 1
        if self._call_count % 2 == 0:
            return EvaluationOutcome(item=None, rejection_reason="alternating_reject")
        crn = factory.create(params)  # type: ignore[arg-type]
        initial_state = self._sample_initial_state(factory)
        trajectories = torch.zeros(
            self._config.n_ssa_trajectories,
            self._config.n_timepoints,
            crn.n_species,
        )
        item = self._build_trajectory_item(
            crn=crn,
            params=params,
            initial_state=initial_state,
            trajectories=trajectories,
            motif_label=factory.motif_type.value,
        )
        return EvaluationOutcome(item=item, rejection_reason=None)


# --- Unit: sample-until-viable --------------------------------------------------


def test_generate_motif_collects_target_viable_count(output_dir: Path) -> None:
    """_generate_motif returns exactly target items when ~50% pass rate."""
    config = GenerationConfig(
        batch_size=4,
        max_attempts_multiplier=20,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=10)]
    pipeline = _AlternatingPipeline(config, tasks)
    result = pipeline._generate_motif(
        BirthDeathFactory(), target=10, label="birth_death"
    )
    assert result.n_viable == 10
    assert result.n_attempted > 10


def test_generate_motif_n_attempted_exceeds_viable(output_dir: Path) -> None:
    """n_attempted > n_viable when some configs are rejected."""
    config = GenerationConfig(
        batch_size=4,
        max_attempts_multiplier=20,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=10)]
    pipeline = _AlternatingPipeline(config, tasks)
    result = pipeline._generate_motif(
        BirthDeathFactory(), target=10, label="birth_death"
    )
    assert result.n_attempted > result.n_viable


def test_generate_motif_stops_at_max_attempts(output_dir: Path) -> None:
    """_generate_motif stops at target * max_attempts_multiplier when all are rejected."""
    target = 10
    multiplier = 2
    config = GenerationConfig(
        batch_size=5,
        max_attempts_multiplier=multiplier,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=target)]
    pipeline = _AlwaysRejectPipeline(config, tasks)
    result = pipeline._generate_motif(
        BirthDeathFactory(), target=target, label="birth_death"
    )
    assert result.n_attempted == target * multiplier
    assert result.n_viable == 0


def test_generate_motif_logs_warning_when_target_not_met(
    output_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_generate_motif logs a warning when the target cannot be reached."""
    config = GenerationConfig(
        batch_size=5,
        max_attempts_multiplier=2,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=10)]
    pipeline = _AlwaysRejectPipeline(config, tasks)
    with caplog.at_level(
        logging.WARNING, logger="crn_surrogate.data.generation.pipeline"
    ):
        pipeline._generate_motif(BirthDeathFactory(), target=10, label="birth_death")
    assert any("only" in msg for msg in caplog.messages)


def test_generate_motif_logs_info_when_target_met(
    output_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_generate_motif logs INFO (not WARNING) when the target is met."""
    config = GenerationConfig(
        batch_size=4,
        max_attempts_multiplier=20,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=5)]
    pipeline = _AlternatingPipeline(config, tasks)
    with caplog.at_level(logging.INFO, logger="crn_surrogate.data.generation.pipeline"):
        pipeline._generate_motif(BirthDeathFactory(), target=5, label="birth_death")
    warning_msgs = [m for m in caplog.messages if "only" in m]
    assert len(warning_msgs) == 0


def test_generate_motif_rejection_counts_populated(output_dir: Path) -> None:
    """rejection_counts in MotifResult maps reason to count correctly."""
    config = GenerationConfig(
        batch_size=5,
        max_attempts_multiplier=2,
        output_dir=str(output_dir),
    )
    tasks = [GenerationTask(BirthDeathFactory(), target=5)]
    pipeline = _AlwaysRejectPipeline(config, tasks)
    result = pipeline._generate_motif(
        BirthDeathFactory(), target=5, label="birth_death"
    )
    assert "always_rejected" in result.rejection_counts
    assert result.rejection_counts["always_rejected"] == result.n_attempted


# --- Unit: task label is threaded into TrajectoryItem --------------------------


def test_evaluate_config_uses_task_label_not_motif_type(output_dir: Path) -> None:
    """_evaluate_config attaches the task label to the TrajectoryItem, not factory.motif_type.value."""
    config = GenerationConfig(
        curation=CurationConfig(
            blowup_threshold=1e9,
            min_coefficient_of_variation=0.0,
            max_zero_fraction=1.0,
            min_reactions_fired=0,
            max_final_population=1e9,
        ),
        n_ssa_trajectories=2,
        simulation_time=5.0,
        n_timepoints=10,
        output_dir=str(output_dir),
    )
    factory = BirthDeathFactory()
    tasks = [GenerationTask(factory, target=1)]
    pipeline = DataGenerationPipeline(config, tasks)
    from crn_surrogate.data.generation.motifs.birth_death import BirthDeathParams

    custom_label = "my_custom_label"
    params = BirthDeathParams(k_prod=10.0, k_deg=0.5)
    outcome = pipeline._evaluate_config(factory, params, motif_label=custom_label)
    assert outcome.viable
    assert outcome.item is not None
    assert outcome.item.motif_label == custom_label
    assert outcome.item.motif_label != factory.motif_type.value


# --- Unit: pipeline constructor -------------------------------------------------


def test_pipeline_requires_non_empty_tasks(output_dir: Path) -> None:
    """DataGenerationPipeline raises ValueError when tasks is empty."""
    config = GenerationConfig(output_dir=str(output_dir))
    with pytest.raises(ValueError, match="tasks must be non-empty"):
        DataGenerationPipeline(config, tasks=[])


# --- Unit: _evaluate_config -----------------------------------------------------


def test_evaluate_config_viable_for_known_good_params(output_dir: Path) -> None:
    """_evaluate_config returns viable=True for a birth-death config with loose curation."""
    config = GenerationConfig(
        curation=CurationConfig(
            blowup_threshold=1e9,
            min_coefficient_of_variation=0.0,
            max_zero_fraction=1.0,
            min_reactions_fired=0,
            max_final_population=1e9,
        ),
        n_ssa_trajectories=2,
        simulation_time=5.0,
        n_timepoints=10,
        output_dir=str(output_dir),
    )
    factory = BirthDeathFactory()
    tasks = [GenerationTask(factory, target=1)]
    pipeline = DataGenerationPipeline(config, tasks)
    from crn_surrogate.data.generation.motifs.birth_death import BirthDeathParams

    params = BirthDeathParams(k_prod=10.0, k_deg=0.5)
    outcome = pipeline._evaluate_config(factory, params, motif_label="birth_death")
    assert outcome.viable
    assert outcome.item is not None
    assert outcome.rejection_reason is None


def test_evaluate_config_rejected_for_impossible_curation(output_dir: Path) -> None:
    """_evaluate_config returns viable=False when curation is impossible to satisfy."""
    config = GenerationConfig(
        curation=CurationConfig(min_coefficient_of_variation=1e9),
        n_ssa_trajectories=2,
        simulation_time=5.0,
        n_timepoints=10,
        output_dir=str(output_dir),
    )
    factory = BirthDeathFactory()
    tasks = [GenerationTask(factory, target=1)]
    pipeline = DataGenerationPipeline(config, tasks)
    from crn_surrogate.data.generation.motifs.birth_death import BirthDeathParams

    params = BirthDeathParams(k_prod=10.0, k_deg=0.5)
    outcome = pipeline._evaluate_config(factory, params, motif_label="birth_death")
    assert not outcome.viable
    assert outcome.item is None
    assert outcome.rejection_reason is not None


# --- Unit: MotifResult ----------------------------------------------------------


def test_motif_result_pass_rate() -> None:
    """MotifResult.pass_rate equals n_viable / n_attempted."""
    result = MotifResult(
        motif_label="birth_death",
        items=[],
        n_attempted=100,
        rejection_counts={"low_cv": 100},
    )
    assert result.pass_rate == pytest.approx(0.0)


def test_motif_result_to_curation_stats_includes_rejection_counts() -> None:
    """to_curation_stats merges rejection_counts into the stats dict."""
    result = MotifResult(
        motif_label="birth_death",
        items=["x"] * 7,  # type: ignore[list-item]
        n_attempted=10,
        rejection_counts={"low_cv": 2, "blowup": 1},
    )
    stats = result.to_curation_stats()
    assert stats["n_sampled"] == 10
    assert stats["n_passed"] == 7
    assert stats["n_rejected"] == 3
    assert stats["low_cv"] == 2
    assert stats["blowup"] == 1


# --- Integration: _balance unchanged -------------------------------------------


def test_balance_caps_over_represented_labels(output_dir: Path) -> None:
    """_balance limits any label to 2x the median count."""
    config = GenerationConfig(random_seed=0, output_dir=str(output_dir))
    tasks = [GenerationTask(BirthDeathFactory(), target=1)]
    pipeline = DataGenerationPipeline(config, tasks)

    def _make_items(label: str, n: int) -> list[TrajectoryItem]:
        return [
            TrajectoryItem(
                crn_repr=None,  # type: ignore[arg-type]
                initial_state=torch.tensor([0.0]),
                trajectories=torch.zeros(1, 1, 1),
                times=torch.tensor([0.0]),
                motif_label=label,
            )
            for _ in range(n)
        ]

    items = _make_items("a", 10) + _make_items("b", 10) + _make_items("c", 100)
    balanced = pipeline._balance(items)
    counts = {
        lbl: sum(1 for i in balanced if i.motif_label == lbl) for lbl in ("a", "b", "c")
    }
    median = 10
    assert counts["c"] <= 2 * median
    assert counts["a"] == 10
    assert counts["b"] == 10


# --- Integration: full pipeline (slow) ------------------------------------------


@pytest.mark.slow
def test_pipeline_output_file_exists(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """Pipeline creates dataset.pt in the configured output directory."""
    tasks = [
        GenerationTask(
            get_factory(MotifType.BIRTH_DEATH), target=fast_config.batch_size
        ),
        GenerationTask(
            get_factory(MotifType.AUTO_CATALYSIS), target=fast_config.batch_size
        ),
    ]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    pipeline.run()
    assert (output_dir / "dataset.pt").exists()


@pytest.mark.slow
def test_pipeline_dataset_contains_trajectory_items(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """Loaded dataset.pt is a list of TrajectoryItem objects."""
    tasks = [
        GenerationTask(
            get_factory(MotifType.BIRTH_DEATH), target=fast_config.batch_size
        ),
        GenerationTask(
            get_factory(MotifType.AUTO_CATALYSIS), target=fast_config.batch_size
        ),
    ]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    pipeline.run()
    items = torch.load(output_dir / "dataset.pt", weights_only=False)
    assert isinstance(items, list)
    for item in items:
        assert isinstance(item, TrajectoryItem)


@pytest.mark.slow
def test_pipeline_trajectory_shape(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """Each item's trajectory has shape (M, T, n_species)."""
    n_target = 5
    tasks = [GenerationTask(get_factory(MotifType.BIRTH_DEATH), target=n_target)]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    pipeline.run()
    items = torch.load(output_dir / "dataset.pt", weights_only=False)
    assert len(items) > 0
    for item in items:
        M, T, n_species = item.trajectories.shape
        assert M == fast_config.n_ssa_trajectories
        assert T == fast_config.n_timepoints
        assert n_species == item.crn_repr.n_species


@pytest.mark.slow
def test_pipeline_cluster_ids_assigned(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """All items have cluster_id >= 0 after pipeline run."""
    tasks = [
        GenerationTask(
            get_factory(MotifType.BIRTH_DEATH), target=fast_config.batch_size
        ),
        GenerationTask(
            get_factory(MotifType.AUTO_CATALYSIS), target=fast_config.batch_size
        ),
    ]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    pipeline.run()
    items = torch.load(output_dir / "dataset.pt", weights_only=False)
    for item in items:
        assert item.cluster_id >= 0


@pytest.mark.slow
def test_pipeline_metadata_json_valid(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """metadata.json is valid JSON with required top-level keys."""
    tasks = [
        GenerationTask(
            get_factory(MotifType.BIRTH_DEATH), target=fast_config.batch_size
        ),
        GenerationTask(
            get_factory(MotifType.AUTO_CATALYSIS), target=fast_config.batch_size
        ),
    ]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    summary = pipeline.run()
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    required_keys = {
        "total_items",
        "counts_per_motif",
        "cluster_id_map",
        "curation_stats",
    }
    assert required_keys.issubset(meta.keys())
    assert meta["total_items"] == summary.total_items


@pytest.mark.slow
def test_pipeline_summary_counts_match_items(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """DatasetSummary counts match the actual items saved to disk."""
    n_target = 5
    tasks = [GenerationTask(get_factory(MotifType.BIRTH_DEATH), target=n_target)]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    summary = pipeline.run()
    items = torch.load(output_dir / "dataset.pt", weights_only=False)
    assert summary.total_items == len(items)


@pytest.mark.slow
def test_pipeline_cluster_id_map_in_summary(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """cluster_id_map in summary has string keys and non-negative integer values."""
    tasks = [
        GenerationTask(
            get_factory(MotifType.BIRTH_DEATH), target=fast_config.batch_size
        ),
        GenerationTask(
            get_factory(MotifType.AUTO_CATALYSIS), target=fast_config.batch_size
        ),
    ]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    summary = pipeline.run()
    for label, cid in summary.cluster_id_map.items():
        assert isinstance(label, str)
        assert isinstance(cid, int)
        assert cid >= 0


@pytest.mark.slow
def test_pipeline_params_serialized_as_dict(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """TrajectoryItem.params is a dict serialized from the typed params dataclass."""
    n_target = 5
    tasks = [GenerationTask(get_factory(MotifType.BIRTH_DEATH), target=n_target)]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    pipeline.run()
    items = torch.load(output_dir / "dataset.pt", weights_only=False)
    assert len(items) > 0
    for item in items:
        assert isinstance(item.params, dict)
        assert "k_prod" in item.params
        assert "k_deg" in item.params


@pytest.mark.slow
def test_pipeline_viable_count_meets_target(
    fast_config: GenerationConfig, output_dir: Path
) -> None:
    """Pipeline collects up to n_target items for birth-death."""
    n_target = 10
    tasks = [GenerationTask(get_factory(MotifType.BIRTH_DEATH), target=n_target)]
    pipeline = DataGenerationPipeline(fast_config, tasks=tasks)
    summary = pipeline.run()
    bd_count = summary.counts_per_motif.get("birth_death", 0)
    assert bd_count == n_target
