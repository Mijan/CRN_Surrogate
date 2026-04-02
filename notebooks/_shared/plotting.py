"""Plotting utilities for CRN trajectory visualization."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# Default color palette for species
SPECIES_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]
INPUT_COLOR = "#5B9BD5"


def setup_style() -> None:
    """Apply the standard notebook plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 100, "font.size": 11})


def plot_trajectory_ensemble(
    ax: plt.Axes,
    t_grid: np.ndarray,
    states: np.ndarray,
    species_idx: int = 0,
    color: str = "tab:blue",
    label: str | None = None,
    alpha: float = 0.25,
    linewidth_sample: float = 0.8,
    linewidth_mean: float = 2.5,
    show_mean: bool = True,
) -> None:
    """Plot M trajectories as thin lines with a bold mean line.

    Args:
        ax: Matplotlib axes to plot on.
        t_grid: (T,) time points.
        states: (M, T, n_species) trajectory ensemble.
        species_idx: Which species to plot.
        color: Color for all lines.
        label: Label for the mean line (shown in legend).
        alpha: Opacity for individual trajectories.
        linewidth_sample: Line width for individual trajectories.
        linewidth_mean: Line width for the mean line.
        show_mean: Whether to plot the mean line.
    """
    M = states.shape[0]
    for i in range(M):
        ax.step(t_grid, states[i, :, species_idx], where="post",
                color=color, alpha=alpha, linewidth=linewidth_sample)
    if show_mean:
        mean = states[:, :, species_idx].mean(axis=0)
        ax.step(t_grid, mean, where="post",
                color=color, linewidth=linewidth_mean, label=label)


def plot_input_protocol(
    ax: plt.Axes,
    t_grid: np.ndarray,
    schedule_values: np.ndarray,
    color: str = INPUT_COLOR,
    label: str | None = None,
    fill_alpha: float = 0.35,
) -> None:
    """Plot a pulsatile input schedule as a filled step function.

    Args:
        ax: Matplotlib axes to plot on.
        t_grid: (T,) time points.
        schedule_values: (T,) schedule evaluated at t_grid.
        color: Fill and line color.
        label: Optional label for legend.
        fill_alpha: Opacity for the filled region.
    """
    ax.fill_between(t_grid, schedule_values, step="post", alpha=fill_alpha, color=color)
    ax.step(t_grid, schedule_values, where="post", color=color, linewidth=1.5, label=label)


def plot_input_and_response(
    t_grid: np.ndarray,
    input_values: np.ndarray,
    states: np.ndarray,
    species_indices: Sequence[int],
    species_names: Sequence[str],
    input_label: str = "Input",
    title: str = "",
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot input protocol on top, species responses below, with shared x-axis.

    Creates a figure with 1 + len(species_indices) vertically stacked panels.

    Args:
        t_grid: (T,) time points.
        input_values: (T,) input schedule values.
        states: (M, T, n_species) trajectory ensemble.
        species_indices: Which species to plot (one panel each).
        species_names: Display names for each species panel.
        input_label: Y-axis label for the input panel.
        title: Figure title.
        colors: Colors for each species. Defaults to SPECIES_COLORS.
        figsize: Figure size. Defaults to (11, 3 + 2.5 * n_panels).

    Returns:
        The matplotlib Figure.
    """
    n_panels = len(species_indices)
    if colors is None:
        colors = SPECIES_COLORS[:n_panels]
    if figsize is None:
        figsize = (11, 3 + 2.5 * n_panels)

    height_ratios = [1] + [2] * n_panels
    fig, axes = plt.subplots(
        1 + n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if 1 + n_panels == 1:
        axes = [axes]

    # Input panel
    plot_input_protocol(axes[0], t_grid, input_values)
    axes[0].set_ylabel(input_label)
    if title:
        axes[0].set_title(title)

    # Species panels
    for panel_idx, (sp_idx, sp_name, color) in enumerate(
        zip(species_indices, species_names, colors)
    ):
        ax = axes[1 + panel_idx]
        plot_trajectory_ensemble(ax, t_grid, states, species_idx=sp_idx,
                                 color=color, label=f"Mean {sp_name}")
        ax.set_ylabel(sp_name)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    return fig
