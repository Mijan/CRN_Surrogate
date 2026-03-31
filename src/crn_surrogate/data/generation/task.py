"""GenerationTask dataclass and convenience task-list builders."""

from __future__ import annotations

from dataclasses import dataclass

from crn_surrogate.data.generation.motif_registry import get_factory
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.base import MotifFactory


@dataclass(frozen=True)
class GenerationTask:
    """Specification for generating a batch of CRN configurations.

    Attributes:
        factory: The motif factory to use for CRN construction.
        target: Number of viable configurations to collect.
        label: String label for this task in the dataset metadata. Defaults
            to the factory's motif_type.value if not provided.
    """

    factory: MotifFactory
    target: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.target <= 0:
            raise ValueError(f"target must be positive, got {self.target}")
        if not self.label:
            object.__setattr__(self, "label", self.factory.motif_type.value)


def all_elementary_tasks(target_per_motif: int = 500) -> list[GenerationTask]:
    """Create one task per elementary motif type, all with the same target.

    Args:
        target_per_motif: Target viable count for each motif.

    Returns:
        List of 8 GenerationTasks (one per elementary MotifType).
    """
    return [
        GenerationTask(factory=get_factory(mt), target=target_per_motif)
        for mt in MotifType
        if mt is not MotifType.COMPOSED
    ]


def default_tasks(
    target_per_motif: int = 500,
) -> list[GenerationTask]:
    """Create the standard full task list: all elementary motifs.

    Args:
        target_per_motif: Target viable count for each elementary motif.

    Returns:
        List of GenerationTasks for all elementary motif types.
    """
    return all_elementary_tasks(target_per_motif=target_per_motif)
