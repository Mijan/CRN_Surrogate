"""Tests for TrainingConfig, SchedulerType, and TrainingMode."""

from __future__ import annotations

import pytest

from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)


def test_default_values():
    cfg = TrainingConfig()
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.training_mode == TrainingMode.TEACHER_FORCING
    assert cfg.use_wandb is False


def test_scheduler_type_enum():
    assert SchedulerType.COSINE.value == "cosine"
    assert SchedulerType.REDUCE_ON_PLATEAU.value == "reduce_on_plateau"


def test_training_mode_enum():
    assert TrainingMode.TEACHER_FORCING.value == "teacher_forcing"
    assert TrainingMode.FULL_ROLLOUT.value == "full_rollout"
    assert TrainingMode.SCHEDULED_SAMPLING.value == "scheduled_sampling"


def test_frozen():
    cfg = TrainingConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.lr = 0.5  # type: ignore[misc]
