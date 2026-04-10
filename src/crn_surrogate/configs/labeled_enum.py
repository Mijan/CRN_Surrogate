"""Base class for enums with string values that support config deserialization."""

from __future__ import annotations

from enum import Enum
from typing import Self


class LabeledEnum(Enum):
    """Enum whose members have string values and can be parsed from strings.

    Use this as the base class for any enum that appears in Hydra YAML
    configs. The string values serve as both the serialized form and the
    human-readable label.

    Usage:
        class TrainingMode(LabeledEnum):
            TEACHER_FORCING = "teacher_forcing"
            FULL_ROLLOUT = "full_rollout"

        mode = TrainingMode.from_str("full_rollout")
    """

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Parse a string into an enum member.

        Args:
            value: The string label to look up.

        Returns:
            The matching enum member.

        Raises:
            ValueError: If no member has the given value, listing all
                valid options.
        """
        try:
            return cls(value)
        except ValueError:
            options = [e.value for e in cls]
            raise ValueError(
                f"Unknown {cls.__name__} {value!r}. Valid options: {options}"
            ) from None
