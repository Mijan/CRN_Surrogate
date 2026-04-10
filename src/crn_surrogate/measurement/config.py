"""Measurement model configuration."""

from __future__ import annotations

from dataclasses import dataclass

from crn_surrogate.configs.labeled_enum import LabeledEnum


class NoiseMode(LabeledEnum):
    """Whether observation noise parameters are learned or fixed."""

    FIXED = "fixed"
    LEARNED = "learned"


class NoiseSharing(LabeledEnum):
    """Whether observation noise is shared across species or per-species."""

    SHARED = "shared"
    PER_SPECIES = "per_species"


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for observation noise.

    The observation noise represents irreducible measurement/model error,
    separate from the CLE process noise. It is parameterized as a relative
    error: sigma_obs = eps * x, where eps is either fixed or learned.

    At high molecule counts (X ~ 100k), CLE process noise is tiny relative
    to the state magnitude, so even small drift prediction errors produce
    enormous NLL. The proportional observation noise absorbs this gracefully:
    sigma_obs ~ 1000 at X=100k but ~ 0.1 at X=10, where the CLE noise
    already dominates.

    Attributes:
        mode: LEARNED makes eps a trainable nn.Parameter; FIXED registers
            it as a non-trainable buffer. LEARNED is recommended for training.
        sharing: PER_SPECIES gives each species its own eps (vector of length
            n_species). SHARED uses a single scalar eps for all species.
        init_value: Initial relative error tolerance. softplus(raw_param) will
            equal this value at construction. Reasonable range is 0.01 to 0.10.
    """

    mode: NoiseMode = NoiseMode.LEARNED
    sharing: NoiseSharing = NoiseSharing.SHARED
    init_value: float = 0.02


@dataclass(frozen=True)
class MeasurementConfig:
    """Top-level measurement model configuration.

    Attributes:
        noise: Observation noise configuration.
        min_variance: Floor for total variance (process + observation)
            to prevent log(0) and 1/v explosions. Default 1e-2 is
            appropriate for molecular count data.
    """

    noise: NoiseConfig = NoiseConfig()
    min_variance: float = 1e-2
