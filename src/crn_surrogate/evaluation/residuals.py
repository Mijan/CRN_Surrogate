"""Transition residual analysis for validating the Gaussian SDE assumption."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal

from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr
from crn_surrogate.simulator.base import StochasticSurrogate

_MIN_VARIANCE: float = 1e-6


@dataclass(frozen=True)
class ResidualReport:
    """Standardized residuals and summary statistics.

    If the Euler-Maruyama Gaussian model is correct, ``residuals`` should
    be approximately i.i.d. N(0, 1).

    Attributes:
        residuals: (M*(T-1), n_species) standardized residuals z_s = (y_next - μ_s) / σ_s.
        raw_residuals: (M*(T-1), n_species) non-standardized residuals y_next - μ_s.
        predicted_means: (M*(T-1), n_species) predicted means μ_s.
        predicted_stds: (M*(T-1), n_species) predicted standard deviations σ_s.
        mean: (n_species,) mean of standardized residuals — should be ~0.
        std: (n_species,) std of standardized residuals — should be ~1.
        kurtosis: (n_species,) kurtosis of standardized residuals — should be ~3.
    """

    residuals: torch.Tensor
    raw_residuals: torch.Tensor
    predicted_means: torch.Tensor
    predicted_stds: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    kurtosis: torch.Tensor


class ResidualAnalyzer:
    """Analyze standardized residuals of one-step Gaussian predictions.

    Uses batched forward passes (no Python loop over transitions).
    """

    def __init__(
        self,
        encoder: BipartiteGNNEncoder,
        sde: StochasticSurrogate,
        crn_repr: CRNTensorRepr,
    ) -> None:
        """Args:
        encoder: Trained bipartite GNN encoder.
        sde: Trained neural SDE.
        crn_repr: Tensor representation of the CRN.
        """
        self._encoder = encoder
        self._sde = sde
        self._crn_repr = crn_repr

    def compute_residuals(
        self,
        trajectories: torch.Tensor,
        times: torch.Tensor,
        dt: float,
    ) -> ResidualReport:
        """Compute standardized residuals for all transitions.

        For each transition (y_t, y_{t+1}):
            μ_s = y_{t,s} + F_θ[s](y_t, t) · dt
            σ_s = sqrt( ||G_θ[s,:](y_t, t)||² · dt )
            z_s = (y_{t+1,s} − μ_s) / σ_s

        If the model is correct, z_s ~ N(0, 1).

        Args:
            trajectories: (M, T, n_species) observed trajectories.
            times: (T,) time grid.
            dt: Time step between consecutive observations.

        Returns:
            ResidualReport with residuals and diagnostic statistics.

        Raises:
            ValueError: If trajectories is not 3D or has fewer than 2 time steps.
        """
        if trajectories.dim() != 3:
            raise ValueError(
                f"trajectories must be 3D (M, T, n_species), got shape {tuple(trajectories.shape)}"
            )
        M, T, n_species = trajectories.shape
        if T < 2:
            raise ValueError(f"Need T >= 2 time steps, got T={T}")

        device = next(self._encoder.parameters()).device
        trajectories = trajectories.to(device)
        times = times.to(device)
        crn_repr = self._crn_repr.to(device)

        self._encoder.eval()
        self._sde.eval()
        with torch.no_grad():
            ctx = self._encoder(crn_repr)

            # Batch all M*(T-1) transitions
            all_y_t = trajectories[:, :-1, :].reshape(
                -1, n_species
            )  # (M*(T-1), n_species)
            all_y_next = trajectories[:, 1:, :].reshape(
                -1, n_species
            )  # (M*(T-1), n_species)
            all_times = times[:-1].repeat(M)  # (M*(T-1),)

            all_drift = self._sde.drift(all_times, all_y_t, ctx)  # (M*(T-1), n_species)
            all_G = self._sde.diffusion(
                all_times, all_y_t, ctx
            )  # (M*(T-1), n_species, n_noise)

            mu = all_y_t + all_drift * dt  # (M*(T-1), n_species)
            variance = (all_G**2).sum(dim=-1) * dt  # (M*(T-1), n_species)
            variance = variance.clamp(min=_MIN_VARIANCE)
            std = variance.sqrt()

            raw_residuals = all_y_next - mu  # (M*(T-1), n_species)
            standardized = raw_residuals / std  # (M*(T-1), n_species)

        mean_z = standardized.mean(dim=0)  # (n_species,)
        std_z = standardized.std(dim=0, correction=1)  # (n_species,)
        z_centered = standardized - mean_z
        kurtosis = (z_centered**4).mean(dim=0) / (z_centered**2).mean(dim=0).pow(2)

        return ResidualReport(
            residuals=standardized,
            raw_residuals=raw_residuals,
            predicted_means=mu,
            predicted_stds=std,
            mean=mean_z,
            std=std_z,
            kurtosis=kurtosis,
        )

    def plot_histogram(
        self,
        report: ResidualReport,
        *,
        species_index: int = 0,
        label: str = "Residuals",
        color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Histogram of standardized residuals with N(0, 1) overlay.

        Args:
            report: Output of compute_residuals.
            species_index: Which species to plot.
            label: Legend label for the histogram.
            color: Bar color.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        z = report.residuals[:, species_index].numpy()
        ax.hist(z, bins=50, density=True, alpha=0.6, color=color, label=label)

        x_range = torch.linspace(float(z.min()), float(z.max()), 200)
        pdf = Normal(0, 1).log_prob(x_range).exp().numpy()
        ax.plot(x_range.numpy(), pdf, "k-", lw=2, label="N(0,1)")

        mean_val = report.mean[species_index].item()
        std_val = report.std[species_index].item()
        kurt_val = report.kurtosis[species_index].item()
        ax.set_title(
            f"Standardized residuals — species {species_index}\n"
            f"mean={mean_val:.3f}  std={std_val:.3f}  kurt={kurt_val:.2f}  (ideal: 0, 1, 3)"
        )
        ax.set_xlabel("z")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_qq(
        self,
        report: ResidualReport,
        *,
        species_index: int = 0,
        label: str = "Residuals",
        color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """QQ plot of standardized residuals against N(0, 1).

        Points near the diagonal indicate the Gaussian assumption is met.

        Args:
            report: Output of compute_residuals.
            species_index: Which species to plot.
            label: Legend label for the scatter points.
            color: Marker color.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        z = report.residuals[:, species_index]
        n = z.shape[0]
        probs = (torch.arange(1, n + 1, dtype=torch.float32) - 0.375) / (n + 0.25)
        theoretical = Normal(0, 1).icdf(probs)
        empirical = z.sort().values

        ax.scatter(
            theoretical.numpy(),
            empirical.numpy(),
            s=4,
            alpha=0.4,
            color=color,
            label=label,
        )
        lo = min(theoretical[0].item(), empirical[0].item())
        hi = max(theoretical[-1].item(), empirical[-1].item())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x (ideal)")
        ax.set_xlabel("Theoretical N(0,1) quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title(f"QQ plot — species {species_index}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax
