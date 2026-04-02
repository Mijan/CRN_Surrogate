"""Trajectory dataclass for simulated time-series data."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crn_surrogate.simulation.interpolation import TimegridUtils


@dataclass
class Trajectory:
    """A simulated trajectory over time.

    Attributes:
        times: (T,) time points (irregular, at event times from SSA).
        states: (T, n_species) molecule counts or concentrations at each time.
    """

    times: torch.Tensor
    states: torch.Tensor

    @property
    def n_steps(self) -> int:
        """Number of recorded time points."""
        return self.states.shape[0]

    @property
    def n_species(self) -> int:
        """Number of species."""
        return self.states.shape[1]

    def mean(self) -> torch.Tensor:
        """Mean state across time steps, shape (n_species,)."""
        return self.states.mean(dim=0)

    def to_grid(self, time_grid: torch.Tensor) -> torch.Tensor:
        """Interpolate onto a regular time grid via zero-order hold.

        Args:
            time_grid: (T_grid,) sorted time points to interpolate onto.

        Returns:
            (T_grid, n_species) tensor of interpolated states.
        """
        return TimegridUtils.interpolate_to_grid(self.times, self.states, time_grid)

    @staticmethod
    def stack_on_grid(
        trajectories: list[Trajectory],
        time_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate multiple trajectories onto a shared grid and stack.

        This is the standard path for building training tensors from SSA output.

        Args:
            trajectories: M independent trajectories (e.g. from
                ``GillespieSSA.simulate_batch``).
            time_grid: (T_grid,) sorted time points.

        Returns:
            (M, T_grid, n_species) tensor.
        """
        return torch.stack([t.to_grid(time_grid) for t in trajectories])
