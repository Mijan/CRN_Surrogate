"""Direct (identity) observation with learned proportional noise."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from crn_surrogate.measurement.base import MeasurementModel
from crn_surrogate.measurement.config import (
    MeasurementConfig,
    NoiseConfig,
    NoiseMode,
    NoiseSharing,
)


class DirectObservation(MeasurementModel):
    """Identity measurement transform with state-proportional Gaussian noise.

    The observation model is:
        y = x + eps * x * z,    z ~ N(0, I)

    equivalently:
        y | x ~ N(x, (eps * x)^2)

    where eps is the relative error tolerance (scalar or per-species vector),
    either learned via gradient descent or held fixed.

    The combined log-likelihood (required by the ABC) adds process and
    observation variance:
        v_total = process_variance + (eps * x_predicted)^2

    Physical interpretation: each molecule emits light (as through GFP),
    and the total fluorescence signal has noise proportional to the
    molecule count. The eps parameter captures the coefficient of
    variation of this measurement process.

    This class is constructed from a MeasurementConfig via the from_config
    class method. Do not construct directly unless you have a good reason.
    """

    def __init__(
        self,
        noise_config: NoiseConfig,
        min_variance: float,
        n_species: int | None = None,
    ) -> None:
        """
        Args:
            noise_config: Observation noise configuration.
            min_variance: Floor for total variance.
            n_species: Required when noise_config.sharing is PER_SPECIES.
                Ignored when SHARED.

        Raises:
            ValueError: If min_variance <= 0.
            ValueError: If noise_config.init_value <= 0.
            ValueError: If PER_SPECIES sharing but n_species is None.
        """
        super().__init__()

        if min_variance <= 0:
            raise ValueError(f"min_variance must be positive, got {min_variance}")
        if noise_config.init_value <= 0:
            raise ValueError(
                f"noise_config.init_value must be positive, got {noise_config.init_value}"
            )

        self._min_variance = min_variance

        if noise_config.sharing == NoiseSharing.PER_SPECIES:
            if n_species is None:
                raise ValueError("n_species is required for PER_SPECIES noise sharing")
            size = n_species
        else:
            size = 1

        # Invert softplus: raw = log(exp(init_value) - 1)
        raw_init = math.log(math.expm1(noise_config.init_value))
        raw_tensor = torch.full((size,), raw_init)

        if noise_config.mode == NoiseMode.LEARNED:
            self._raw_eps = nn.Parameter(raw_tensor)
        else:
            self.register_buffer("_raw_eps", raw_tensor)

    @classmethod
    def from_config(
        cls,
        config: MeasurementConfig,
        n_species: int | None = None,
    ) -> DirectObservation:
        """Construct from a MeasurementConfig.

        This is the recommended construction path. The Trainer uses this.

        Args:
            config: Full measurement configuration.
            n_species: Required for PER_SPECIES noise sharing.
        """
        return cls(
            noise_config=config.noise,
            min_variance=config.min_variance,
            n_species=n_species,
        )

    @property
    def eps(self) -> Tensor:
        """Current relative error tolerance (always positive)."""
        return torch.nn.functional.softplus(self._raw_eps)

    def predict(self, x_latent: Tensor) -> Tensor:
        """Identity transform."""
        return x_latent

    def sample(self, x_latent: Tensor) -> Tensor:
        """Sample: x + eps * x * z, z ~ N(0, I)."""
        eps = self.eps
        sigma = eps * x_latent
        noise = torch.randn_like(x_latent)
        return x_latent + sigma * noise

    def log_likelihood(
        self,
        y_observed: Tensor,
        x_predicted: Tensor,
        process_variance: Tensor,
    ) -> Tensor:
        """Gaussian log p(y | x_predicted) combining process and observation noise.

        Total variance: v_total = process_variance + (eps * x_predicted)^2

        Args:
            y_observed: (..., n_species) observed next state.
            x_predicted: (..., n_species) predicted next state (mu from EM).
            process_variance: (..., n_species) process variance from SDE diffusion.

        Returns:
            (..., n_species) per-element log-likelihood.
        """
        eps = self.eps
        obs_variance = (eps * x_predicted) ** 2
        total_variance = process_variance + obs_variance
        total_variance = total_variance.clamp(min=self._min_variance)
        residual = y_observed - x_predicted

        return -0.5 * (residual**2 / total_variance + total_variance.log())

    @property
    def n_observed(self) -> int | None:
        return None  # same as input
