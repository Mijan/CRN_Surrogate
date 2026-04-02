"""Data generation sub-package for CRN surrogate training data.

Provides factories for all supported CRN motif types, parameter sampling,
trajectory curation, and an end-to-end pipeline that writes dataset.pt and
metadata.json to disk.
"""

from crn_surrogate.data.generation.composer import (
    ComposedMotifFactory,
    ComposedParams,
    CompositionSpec,
)
from crn_surrogate.data.generation.configs import (
    CurationConfig,
    GenerationConfig,
    SamplingConfig,
)
from crn_surrogate.data.generation.curation import CurationResult, ViabilityFilter
from crn_surrogate.data.generation.motif_registry import get_factory
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.pipeline import (
    DataGenerationPipeline,
    DatasetSummary,
)
from crn_surrogate.data.generation.task import (
    GenerationTask,
    all_elementary_tasks,
    default_tasks,
)

__all__ = [
    "ComposedMotifFactory",
    "ComposedParams",
    "CompositionSpec",
    "CurationConfig",
    "CurationResult",
    "DataGenerationPipeline",
    "DatasetSummary",
    "GenerationConfig",
    "GenerationTask",
    "MotifType",
    "SamplingConfig",
    "ViabilityFilter",
    "all_elementary_tasks",
    "default_tasks",
    "get_factory",
]
