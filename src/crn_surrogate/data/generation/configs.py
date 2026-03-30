"""Configuration dataclasses for the data generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for kinetic parameter sampling.

    Attributes:
        n_samples_per_motif: Number of parameter configurations to sample per motif type.
        random_seed: Seed for the random number generator ensuring reproducibility.
    """

    n_samples_per_motif: int = 500
    random_seed: int = 42


@dataclass(frozen=True)
class CurationConfig:
    """Configuration for trajectory viability filtering.

    Attributes:
        blowup_threshold: Trajectories with any state value above this are rejected.
        min_coefficient_of_variation: Minimum per-species coefficient of variation
            (max over species) required for acceptance.
        max_zero_fraction: Maximum fraction of timepoints where all species are zero
            before rejecting as stuck-at-zero.
        min_reactions_fired: Minimum total number of state transitions across all
            trajectories for the configuration to be considered active.
        max_final_population: Maximum mean population in the last 10 timepoints
            before rejecting as unbounded.
    """

    blowup_threshold: float = 1e6
    min_coefficient_of_variation: float = 0.01
    max_zero_fraction: float = 0.95
    min_reactions_fired: int = 10
    max_final_population: float = 1e5


@dataclass(frozen=True)
class GenerationConfig:
    """Top-level configuration for the full data generation run.

    Attributes:
        sampling: Configuration for kinetic parameter sampling.
        curation: Configuration for trajectory viability filtering.
        n_ssa_trajectories: Number of independent SSA runs per CRN configuration.
        simulation_time: End time for each SSA simulation.
        n_timepoints: Number of evenly-spaced timepoints in the output grid.
        n_samples_per_motif: Parameter configurations to attempt for each elementary motif.
        n_samples_per_composed: Parameter configurations to attempt for composed motifs.
        target_total_viable: Desired total number of viable items in the final dataset.
        output_dir: Directory path where dataset.pt and metadata.json are written.
        random_seed: Master random seed for reproducibility.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    n_ssa_trajectories: int = 32
    simulation_time: float = 100.0
    n_timepoints: int = 200
    n_samples_per_motif: int = 500
    n_samples_per_composed: int = 300
    target_total_viable: int = 15000
    output_dir: str = "data/generated"
    random_seed: int = 42
