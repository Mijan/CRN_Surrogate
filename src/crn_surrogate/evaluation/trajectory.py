"""Trajectory-level statistics and comparison plots for SDE vs SSA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class TrajectoryComparator:
    """Compare SDE rollouts against SSA ground truth.

    Accepts pre-generated trajectory tensors and provides plot methods
    and a summary metrics dictionary.
    """

    def __init__(
        self,
        sde_trajectories: torch.Tensor,
        ssa_trajectories: torch.Tensor,
        times: torch.Tensor,
        *,
        analytical_mean: float | None = None,
        analytical_var: float | None = None,
    ) -> None:
        """Args:
        sde_trajectories: (K, T, n_species) SDE rollouts.
        ssa_trajectories: (M, T, n_species) SSA ground truth.
        times: (T,) time grid.
        analytical_mean: Known stationary mean (for reference line).
        analytical_var: Known stationary variance (for reference line).
        """
        if sde_trajectories.dim() != 3:
            raise ValueError(
                f"sde_trajectories must be 3D (K, T, n_species), "
                f"got shape {tuple(sde_trajectories.shape)}"
            )
        if ssa_trajectories.dim() != 3:
            raise ValueError(
                f"ssa_trajectories must be 3D (M, T, n_species), "
                f"got shape {tuple(ssa_trajectories.shape)}"
            )
        self._sde = sde_trajectories
        self._ssa = ssa_trajectories
        self._times = times
        self._analytical_mean = analytical_mean
        self._analytical_var = analytical_var

    def plot_mean_std(
        self,
        *,
        species_index: int = 0,
        sde_label: str = "SDE",
        sde_color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Mean ± std band for both SDE and SSA.

        Args:
            species_index: Which species to plot.
            sde_label: Legend label for the SDE curve.
            sde_color: Color for the SDE curve.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        t = self._times.numpy()
        sde_mean, sde_std = self._trajectory_stats(self._sde, species_index)
        ssa_mean, ssa_std = self._trajectory_stats(self._ssa, species_index)

        ax.plot(t, ssa_mean, "k-", lw=2, label="Gillespie")
        ax.fill_between(
            t, ssa_mean - ssa_std, ssa_mean + ssa_std, alpha=0.10, color="black"
        )
        ax.plot(t, sde_mean, "--", color=sde_color, lw=2, label=sde_label)
        ax.fill_between(
            t, sde_mean - sde_std, sde_mean + sde_std, alpha=0.20, color=sde_color
        )

        if self._analytical_mean is not None:
            ax.axhline(
                self._analytical_mean,
                color="red",
                linestyle=":",
                lw=1.5,
                label=f"E[X]={self._analytical_mean:.1f}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Mean \u00b1 Std")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_variance(
        self,
        *,
        species_index: int = 0,
        sde_label: str = "SDE Var",
        sde_color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Variance over time for both SDE and SSA.

        Args:
            species_index: Which species to plot.
            sde_label: Legend label for the SDE variance curve.
            sde_color: Color for the SDE curve.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        t = self._times.numpy()
        _, sde_std = self._trajectory_stats(self._sde, species_index)
        _, ssa_std = self._trajectory_stats(self._ssa, species_index)

        ax.plot(t, ssa_std**2, "k-", lw=2, label="Gillespie Var")
        ax.plot(t, sde_std**2, "--", color=sde_color, lw=2, label=sde_label)

        if self._analytical_var is not None:
            ax.axhline(
                self._analytical_var,
                color="red",
                linestyle=":",
                lw=1.5,
                label=f"Var[X]={self._analytical_var:.1f}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Variance")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_sample_paths(
        self,
        *,
        n_paths: int = 10,
        species_index: int = 0,
        sde_label: str = "SDE",
        sde_color: str = "steelblue",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Overlay sample paths from the SDE.

        Args:
            n_paths: Number of paths to draw.
            species_index: Which species to plot.
            sde_label: Legend label (shown on first path only).
            sde_color: Color for the SDE paths.
            ax: Matplotlib axes. Created if None.

        Returns:
            The populated axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        t = self._times.numpy()
        k_plot = min(n_paths, self._sde.shape[0])
        for k in range(k_plot):
            ax.plot(
                t,
                self._sde[k, :, species_index].numpy(),
                alpha=0.35,
                lw=0.9,
                color=sde_color,
                label=sde_label if k == 0 else "",
            )

        if self._analytical_mean is not None:
            ax.axhline(
                self._analytical_mean,
                color="red",
                linestyle=":",
                lw=1.5,
                label=f"E[X]={self._analytical_mean:.1f}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_summary(
        self,
        *,
        species_index: int = 0,
        sde_label: str = "SDE",
        sde_color: str = "steelblue",
    ) -> "Figure":
        """Three-panel figure: mean±std, variance, sample paths.

        Args:
            species_index: Which species to plot in all panels.
            sde_label: Legend label for the SDE.
            sde_color: Color for the SDE.

        Returns:
            The populated figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        self.plot_mean_std(
            species_index=species_index,
            sde_label=sde_label,
            sde_color=sde_color,
            ax=axes[0],
        )
        self.plot_variance(
            species_index=species_index,
            sde_label=sde_label,
            sde_color=sde_color,
            ax=axes[1],
        )
        self.plot_sample_paths(
            species_index=species_index,
            sde_label=sde_label,
            sde_color=sde_color,
            ax=axes[2],
        )
        axes[0].set_title("Mean ± Std")
        axes[1].set_title("Variance")
        axes[2].set_title("Sample Paths")
        fig.tight_layout()
        return fig

    def metrics(self) -> dict[str, float]:
        """Summary metrics comparing SDE trajectories to SSA ground truth.

        Returns:
            Dict with keys: mean_mse, var_mse, final_mean, final_var,
            mean_sde_std, diffusion_collapsed.
        """
        s = 0  # summarise over species 0 for scalar metrics
        sde_mean, sde_std = self._trajectory_stats(self._sde, s)
        ssa_mean, ssa_std = self._trajectory_stats(self._ssa, s)

        mean_mse = float(
            ((torch.tensor(sde_mean) - torch.tensor(ssa_mean)) ** 2).mean()
        )
        var_mse = float(
            ((torch.tensor(sde_std**2) - torch.tensor(ssa_std**2)) ** 2).mean()
        )
        final_mean = float(self._sde[:, -10:, s].mean())
        final_var = float(self._sde[:, -10:, s].var())
        mean_sde_std = float(self._sde[:, :, s].std(dim=0).mean())
        return {
            "mean_mse": mean_mse,
            "var_mse": var_mse,
            "final_mean": final_mean,
            "final_var": final_var,
            "mean_sde_std": mean_sde_std,
            "diffusion_collapsed": float(mean_sde_std < 0.1),
        }

    @staticmethod
    def _trajectory_stats(
        trajs: torch.Tensor, species_index: int
    ) -> tuple[float, float]:
        """Return (mean, std) numpy arrays over the sample dimension."""
        s = trajs[:, :, species_index]  # (K, T)
        return s.mean(dim=0).numpy(), s.std(dim=0).numpy()
