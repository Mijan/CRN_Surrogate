"""Tests for measurement config enums and dataclasses."""

from __future__ import annotations

import pytest

from crn_surrogate.measurement.config import (
    MeasurementConfig,
    NoiseConfig,
    NoiseMode,
    NoiseSharing,
)


def test_noise_mode_enum() -> None:
    assert NoiseMode.LEARNED
    assert NoiseMode.FIXED


def test_noise_sharing_enum() -> None:
    assert NoiseSharing.SHARED
    assert NoiseSharing.PER_SPECIES


def test_noise_config_defaults() -> None:
    cfg = NoiseConfig()
    assert cfg.mode == NoiseMode.LEARNED
    assert cfg.sharing == NoiseSharing.SHARED
    assert cfg.init_value == pytest.approx(0.02)


def test_measurement_config_defaults() -> None:
    cfg = MeasurementConfig()
    assert cfg.min_variance == pytest.approx(1e-2)
    assert isinstance(cfg.noise, NoiseConfig)
    assert cfg.noise.mode == NoiseMode.LEARNED


def test_frozen() -> None:
    cfg = MeasurementConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.min_variance = 0.5  # type: ignore[misc]
