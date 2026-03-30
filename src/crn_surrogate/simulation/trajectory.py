"""Trajectory dataclass for simulated time-series data."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    """A simulated trajectory over time.

    Attributes:
        times: (T,) time points.
        states: (T, n_species) molecule counts or concentrations at each time.
    """

    times: torch.Tensor
    states: torch.Tensor

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return self.states.shape[0]

    @property
    def n_species(self) -> int:
        """Number of species."""
        return self.states.shape[1]

    def mean(self) -> torch.Tensor:
        """Mean state across time steps, shape (n_species,)."""
        return self.states.mean(dim=0)
