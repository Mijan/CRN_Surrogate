"""Measurement model interface.

A measurement model maps latent SDE states to observable space and
provides the combined log-likelihood log p(y | x, process_variance). It is
the standard state-space model decomposition: the SDE is the process model,
the measurement model is the observation model.

During surrogate training (Components 1-2), the measurement model is
always DirectObservation (identity transform + learned proportional
noise). The SDE is trained on fully observed SSA data via teacher forcing.

Richer measurement models (partial observation, FRET, flow cytometry)
are used at inference time in Component 3, where candidate CRNs are
scored against real experimental data. The surrogate forward-simulates
the full latent state, then the measurement model computes the
likelihood against partial/transformed observations. This separation
means training never sees partial observations; partial observability
is handled entirely in the scoring pipeline.

Design constraint: the measurement model must NOT absorb process
variance that the SDE diffusion should be learning. The proportional
noise parameterization (sigma_obs = eps * x) naturally satisfies this
because at low molecule counts where CLE noise dominates, the
observation noise is negligible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class MeasurementModel(ABC, nn.Module):
    """Maps latent SDE state to observed space and provides the combined log-likelihood.

    Subclasses must implement:
        predict: deterministic transform from latent to observed space
        sample: predict + noise (for forward simulation / posterior predictive checks)
        log_likelihood: log p(y | x_predicted, process_variance)
        n_observed: dimensionality of the observation space
    """

    @abstractmethod
    def predict(self, x_latent: Tensor) -> Tensor:
        """Deterministic transform from latent state to observed space.

        Args:
            x_latent: (..., n_species) latent molecular state.

        Returns:
            (..., n_observed) predicted observation.
        """

    @abstractmethod
    def sample(self, x_latent: Tensor) -> Tensor:
        """Sample a noisy observation given the latent state.

        Equivalent to predict(x_latent) + noise drawn from the
        observation noise model. Used for forward simulation and
        posterior predictive checks.

        Args:
            x_latent: (..., n_species) latent molecular state.

        Returns:
            (..., n_observed) noisy observation sample.
        """

    @abstractmethod
    def log_likelihood(
        self,
        y_observed: Tensor,
        x_predicted: Tensor,
        process_variance: Tensor,
    ) -> Tensor:
        """Log-likelihood of observed data combining process and observation noise.

        Computes log p(y | x_predicted) where the total variance combines the
        SDE process variance (from the Euler-Maruyama step) with the observation
        model's own noise. Evaluated per-element; the caller is responsible for
        any summation or averaging.

        Args:
            y_observed: (..., n_observed) observed data.
            x_predicted: (..., n_species) predicted next state (mu from EM step).
            process_variance: (..., n_species) process variance from SDE diffusion.

        Returns:
            (..., n_observed) log-likelihood per observed dimension.
        """

    @property
    @abstractmethod
    def n_observed(self) -> int | None:
        """Number of observed dimensions, or None if same as input."""
