"""Data generation sub-package for CRN surrogate training data.

Provides factories for all supported CRN motif types, parameter sampling,
trajectory curation, and an end-to-end pipeline that writes dataset.pt and
metadata.json to disk.
"""

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

__all__ = [
    "CurationConfig",
    "CurationResult",
    "DataGenerationPipeline",
    "DatasetSummary",
    "GenerationConfig",
    "MotifType",
    "SamplingConfig",
    "ViabilityFilter",
    "get_factory",
]
