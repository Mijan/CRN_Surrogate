from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from crn_surrogate.measurement.base import MeasurementModel

if TYPE_CHECKING:
    from crn_surrogate.crn.inputs import ResolvedProtocol
    from crn_surrogate.encoder.bipartite_gnn import CRNContext
    from crn_surrogate.simulator.neural_sde import CRNNeuralSDE


class TrajectoryLoss(ABC):
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


class MeanMatchingLoss(TrajectoryLoss):
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


class VarianceMatchingLoss(TrajectoryLoss):
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


class TransitionNLL(nn.Module):
    """One-step negative log-likelihood for SDE teacher-forcing training.

    Evaluates the log-likelihood of observed transitions under the
    Euler-Maruyama model. Two modes are supported:

    **Without measurement model (legacy)**: Gaussian NLL using process variance only.
        NLL = ½ Σ_s [ (y_{s,t+1} - μ_s)² / σ²_proc,s + log(σ²_proc,s) ]

    **With measurement model (e.g., DirectObservation)**: combines process and
    observation variance via the measurement model's ``log_likelihood`` method.
        v_total = σ²_proc + (eps · μ)²
        NLL = ½ Σ_s [ (y_{s,t+1} - μ_s)² / v_total,s + log(v_total,s) ]

    The observation noise is crucial at high molecule counts (X ~ 100k) where CLE
    process noise is negligible relative to the state magnitude. Even a 1% drift
    prediction error produces a residual far larger than the process variance,
    causing catastrophic NLL spikes. The proportional observation noise
    (sigma_obs = eps * x) absorbs moderate drift errors gracefully.

    In both cases, teacher forcing is used: each step starts from the true
    observed state, not the predicted state.

    Attributes:
        _measurement_model: Optional measurement model for combining variances.
        _min_variance: Floor for process variance.
    """

    def __init__(
        self,
        *,
        measurement_model: MeasurementModel | None = None,
        # Default 1e-2: SSA data is inherently noisy. A variance floor below 1.0
        # is not physically meaningful for molecule counts and causes catastrophic
        # NLL spikes when residuals are even moderately large.
        min_variance: float = 1e-2,
    ) -> None:
        """Args:
        measurement_model: If None, legacy behavior (process variance only, Gaussian
            NLL). If provided, its ``log_likelihood`` method is called to
            combine process and observation variance.
        min_variance: Floor for process variance from the SDE diffusion.
        """
        super().__init__()
        self._measurement_model = measurement_model
        self._min_variance = min_variance

    def compute(
        self,
        sde: CRNNeuralSDE,
        crn_context: CRNContext,
        true_trajectory: torch.Tensor,
        times: torch.Tensor,
        dt: float,
        mask: torch.Tensor | None = None,
        resolved_protocol: ResolvedProtocol | None = None,
    ) -> torch.Tensor:
        """Compute mean NLL over all transitions in the trajectory.

        All M*(T-1) transitions are batched into a single forward pass through
        the drift and diffusion networks (no Python loop over M or T).

        Args:
            sde: The neural SDE model (provides drift and diffusion).
            crn_context: CRN encoder output.
            true_trajectory: (T, n_species) observed states on a regular
                time grid. Can also be (M, T, n_species) in which case
                transitions from ALL M trajectories are used.
            times: (T,) time points corresponding to the trajectory.
            dt: Time step between consecutive observations.
            mask: (n_species,) optional bool mask; True = valid (internal) species.
            resolved_protocol: Optional bundle containing the protocol embedding
                for FiLM conditioning. Only the embedding field is used here;
                species clamping is the solver's responsibility.

        Returns:
            Scalar mean NLL loss.

        Raises:
            ValueError: If T < 2 (no transitions to evaluate).
        """
        protocol_embedding = (
            resolved_protocol.embedding if resolved_protocol is not None else None
        )

        if true_trajectory.dim() == 2:
            true_trajectory = true_trajectory.unsqueeze(0)  # (1, T, n_species)

        M, T, n_species = true_trajectory.shape
        if T < 2:
            raise ValueError(f"TransitionNLL requires T >= 2 time steps, got T={T}")

        # Reshape all M*(T-1) transitions into a single batch
        all_y_t = true_trajectory[:, :-1, :].reshape(
            -1, n_species
        )  # (M*(T-1), n_species)
        all_y_next = true_trajectory[:, 1:, :].reshape(
            -1, n_species
        )  # (M*(T-1), n_species)
        all_times = times[:-1].repeat(M)  # (M*(T-1),)

        # Two batched forward passes instead of M*(T-1) individual ones.
        if protocol_embedding is not None:
            all_drift = sde.drift(all_times, all_y_t, crn_context, protocol_embedding)
            all_G = sde.diffusion(all_times, all_y_t, crn_context, protocol_embedding)
        else:
            all_drift = sde.drift(all_times, all_y_t, crn_context)
            all_G = sde.diffusion(all_times, all_y_t, crn_context)

        mu = all_y_t + all_drift * dt  # (M*(T-1), n_species)
        variance = (all_G**2).sum(dim=-1) * dt  # (M*(T-1), n_species)
        variance = variance.clamp(min=self._min_variance)

        residual = all_y_next - mu  # (M*(T-1), n_species)

        if self._measurement_model is not None:
            log_lik = self._measurement_model.log_likelihood(
                y_observed=all_y_next,
                x_predicted=mu,
                process_variance=variance,
            )
            nll = -log_lik  # (M*(T-1), n_species)
        else:
            # Legacy: Gaussian NLL with process variance only
            nll = 0.5 * (
                residual**2 / variance + variance.log()
            )  # (M*(T-1), n_species)

        if mask is not None:
            nll = nll * mask.float()

        n_dims = int(mask.sum().item()) if mask is not None else n_species
        return nll.sum() / (M * (T - 1) * n_dims)


GaussianTransitionNLL = TransitionNLL  # backward compatibility


class CombinedTrajectoryLoss(TrajectoryLoss):
    """Weighted sum of multiple TrajectoryLoss instances.

    Default configuration: MeanMatchingLoss (weight=1.0) +
    VarianceMatchingLoss (weight=var_weight, default 0.5).
    """

    def __init__(
        self,
        losses: list[tuple[TrajectoryLoss, float]] | None = None,
        var_weight: float = 0.5,
    ) -> None:
        """Args:
        losses: Explicit list of (loss_fn, weight) pairs. If None, uses default
            MeanMatchingLoss + VarianceMatchingLoss pair.
        var_weight: Weight for VarianceMatchingLoss in the default configuration.
        """
        if losses is not None:
            self._losses: list[tuple[TrajectoryLoss, float]] = losses
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
