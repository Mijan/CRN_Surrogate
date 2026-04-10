"""Loss functions for CRN surrogate training.

StepLoss implementations score one-step (teacher-forcing) predictions.
RolloutLoss implementations score full trajectory rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from crn_surrogate.measurement.base import MeasurementModel


class RolloutLoss(ABC):
    """Abstract base class for trajectory-matching loss functions.

    Both pred_states and true_states must always be 3D tensors.
    """

    @abstractmethod
    def compute(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_states: (K, T, n_species) K samples from the neural SDE.
            true_states: (M, T, n_species) M ground-truth Gillespie trajectories.
            mask: (n_species,) optional bool mask; True = valid species.

        Returns:
            Scalar loss tensor.
        """

    def _validate_3d(
        self, pred_states: torch.Tensor, true_states: torch.Tensor
    ) -> None:
        """Raise ValueError if either input is not 3D."""
        if pred_states.dim() != 3:
            raise ValueError(
                f"pred_states must be 3D (K, T, n_species), got shape {tuple(pred_states.shape)}"
            )
        if true_states.dim() != 3:
            raise ValueError(
                f"true_states must be 3D (M, T, n_species), got shape {tuple(true_states.shape)}"
            )


# Backward-compatible alias
TrajectoryLoss = RolloutLoss


class MeanMatchingLoss(RolloutLoss):
    """Compares mean of K predicted trajectories against mean of M true trajectories.

    L = (1/T) * sum_t || E_pred[X(t)] - E_true[X(t)] ||^2
    """

    def compute(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mean-matching loss.

        Args:
            pred_states: (K, T, n_species) predicted states.
            true_states: (M, T, n_species) ground-truth states.
            mask: (n_species,) optional bool mask.

        Returns:
            Scalar loss tensor.
        """
        self._validate_3d(pred_states, true_states)
        pred_mean = pred_states.mean(dim=0)  # (T, n_species)
        true_mean = true_states.mean(dim=0)  # (T, n_species)
        diff = pred_mean - true_mean
        if mask is not None:
            diff = diff[..., mask]
        return (diff**2).mean()


class VarianceMatchingLoss(RolloutLoss):
    """Compares variance of K predicted trajectories against variance of M true trajectories.

    L_var = mean_t( || Var_pred[X(t)] - Var_true[X(t)] ||^2 ) / scale

    where scale = (true_mean.abs().mean().clamp(min=1.0))^2
    so the loss is dimensionless and balanced with MeanMatchingLoss.

    Requires K >= 2 and M >= 2.
    """

    def compute(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute variance-matching loss.

        Args:
            pred_states: (K, T, n_species) predicted states. Must have K >= 2.
            true_states: (M, T, n_species) ground-truth states. Must have M >= 2.
            mask: (n_species,) optional bool mask.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If inputs are not 3D or have fewer than 2 samples.
        """
        self._validate_3d(pred_states, true_states)
        K, M = pred_states.shape[0], true_states.shape[0]
        if K < 2:
            raise ValueError(
                f"VarianceMatchingLoss requires K >= 2 SDE samples, got K={K}"
            )
        if M < 2:
            raise ValueError(
                f"VarianceMatchingLoss requires M >= 2 true trajectories, got M={M}"
            )
        pred_var = pred_states.var(dim=0, correction=1)  # (T, n_species)
        true_var = true_states.var(dim=0, correction=1)  # (T, n_species)
        true_mean = true_states.mean(dim=0)
        true_mean_for_scale = true_mean[..., mask] if mask is not None else true_mean
        scale = true_mean_for_scale.abs().mean().clamp(min=1.0) ** 2

        diff = pred_var - true_var
        if mask is not None:
            diff = diff[..., mask]
        return (diff**2).mean() / scale


class StepLoss(ABC):
    """Per-transition loss on batched (N, S) tensors.

    Implementations define how to score the one-step prediction
    y_next vs mu given the process variance from the SDE diffusion.
    """

    @abstractmethod
    def compute(
        self,
        y_next: torch.Tensor,
        mu: torch.Tensor,
        process_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-element loss.

        Args:
            y_next: (N, S) observed next state.
            mu: (N, S) predicted next state.
            process_variance: (N, S) process variance from SDE diffusion.

        Returns:
            (N, S) per-element loss (not reduced).
        """

    def parameters(self) -> list[torch.nn.Parameter]:
        """Trainable parameters owned by this loss. Empty by default."""
        return []

    def state_dict(self) -> dict:
        """Serializable state for checkpointing. Empty by default."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. No-op by default."""

    def extra_metrics(self) -> dict[str, float]:
        """Per-epoch metrics to log (e.g. obs_eps). Empty by default."""
        return {}


# Backward-compatible alias
BatchedStepLoss = StepLoss


class RelativeMSEStepLoss(StepLoss):
    """Scale-invariant MSE: ((y_next - mu) / (|y_t| + eps))^2 loss for deterministic (ODE) surrogates.""

    Ignores process_variance entirely. No noise model, no variance floor.
    """

    def __init__(self, eps: float = 1.0) -> None:
        self._eps = eps

    def compute(
        self,
        y_next: torch.Tensor,
        mu: torch.Tensor,
        process_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute squared residuals.

        Args:
            y_next: (N, S) observed next state.
            mu: (N, S) predicted next state.
            process_variance: (N, S) ignored.

        Returns:
            (N, S) squared residuals.
        """

        residual = y_next - mu
        scale = y_next.abs() + self._eps
        return (residual / scale) ** 2


class NLLStepLoss(StepLoss):
    """Gaussian NLL loss for stochastic (SDE) surrogates.

    Combines process variance from the SDE diffusion with observation
    variance from a MeasurementModel. Falls back to process-only Gaussian
    NLL when no measurement model is provided.
    """

    def __init__(
        self,
        measurement_model: MeasurementModel | None = None,
        min_variance: float = 1e-2,
    ) -> None:
        """Args:
        measurement_model: Optional model for combining process and observation
            variance. When None, uses process variance only (Gaussian NLL).
        min_variance: Floor applied to process variance before NLL computation.
        """
        self._measurement_model = measurement_model
        self._min_variance = min_variance

    def compute(
        self,
        y_next: torch.Tensor,
        mu: torch.Tensor,
        process_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL, combining process and observation variance.

        Args:
            y_next: (N, S) observed next state.
            mu: (N, S) predicted next state.
            process_variance: (N, S) process variance from SDE diffusion.

        Returns:
            (N, S) per-element NLL.
        """
        variance = process_variance.clamp(min=self._min_variance)
        if self._measurement_model is not None:
            return -self._measurement_model.log_likelihood(
                y_observed=y_next,
                x_predicted=mu,
                process_variance=variance,
            )
        residual = y_next - mu
        return 0.5 * (residual**2 / variance + variance.log())

    def parameters(self) -> list[torch.nn.Parameter]:
        """Trainable parameters: measurement model params if present."""
        if self._measurement_model is not None:
            return list(self._measurement_model.parameters())
        return []

    def state_dict(self) -> dict:
        """Serializable state including measurement model weights."""
        if self._measurement_model is not None:
            return {"measurement_model": self._measurement_model.state_dict()}
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore measurement model weights from checkpoint state."""
        if self._measurement_model is not None and "measurement_model" in state:
            self._measurement_model.load_state_dict(state["measurement_model"])

    def extra_metrics(self) -> dict[str, float]:
        """Log obs_eps if the measurement model exposes it."""
        if self._measurement_model is not None and hasattr(
            self._measurement_model, "eps"
        ):
            return {"obs_eps": self._measurement_model.eps.mean().item()}
        return {}


class CombinedRolloutLoss(RolloutLoss):
    """Weighted sum of multiple RolloutLoss instances.

    Default configuration: MeanMatchingLoss (weight=1.0) +
    VarianceMatchingLoss (weight=var_weight, default 0.5).
    """

    def __init__(
        self,
        losses: list[tuple[RolloutLoss, float]] | None = None,
        var_weight: float = 0.5,
    ) -> None:
        """Args:
        losses: Explicit list of (loss_fn, weight) pairs. If None, uses default
            MeanMatchingLoss + VarianceMatchingLoss pair.
        var_weight: Weight for VarianceMatchingLoss in the default configuration.
        """
        if losses is not None:
            self._losses: list[tuple[RolloutLoss, float]] = losses
        else:
            self._losses = [
                (MeanMatchingLoss(), 1.0),
                (VarianceMatchingLoss(), var_weight),
            ]

    def compute(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted sum of all component losses.

        Args:
            pred_states: (K, T, n_species) predicted states.
            true_states: (M, T, n_species) ground-truth states.
            mask: (n_species,) optional bool mask.

        Returns:
            Scalar combined loss tensor.
        """
        total = pred_states.new_zeros(())
        for loss_fn, weight in self._losses:
            total = total + weight * loss_fn.compute(pred_states, true_states, mask)
        return total


# Backward-compatible alias
CombinedTrajectoryLoss = CombinedRolloutLoss
