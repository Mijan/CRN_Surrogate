"""Enum of all supported CRN motif types for training data generation."""

from __future__ import annotations

import enum


class MotifType(enum.Enum):
    """Supported elementary CRN motif types.

    Each value is a human-readable string label used as the motif_label
    in TrajectoryItem and as keys in dataset metadata.
    """

    BIRTH_DEATH = "birth_death"
    AUTO_CATALYSIS = "auto_catalysis"
    NEGATIVE_AUTOREGULATION = "negative_autoregulation"
    TOGGLE_SWITCH = "toggle_switch"
    ENZYMATIC_CATALYSIS = "enzymatic_catalysis"
    INCOHERENT_FEEDFORWARD = "incoherent_feedforward"
    REPRESSILATOR = "repressilator"
    SUBSTRATE_INHIBITION = "substrate_inhibition"
