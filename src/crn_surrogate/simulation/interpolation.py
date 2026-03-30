"""Zero-order hold interpolation for SSA event trajectories onto regular grids."""
from __future__ import annotations

import torch


def interpolate_to_grid(
    event_times: torch.Tensor,
    event_states: torch.Tensor,
    time_grid: torch.Tensor,
) -> torch.Tensor:
    """Zero-order hold interpolation of an SSA trajectory onto a regular time grid.

    Each grid point takes the value of the most recent event at or before it.
    Grid points before the first event or beyond the last event are clamped to
    the first or last state respectively.

    Args:
        event_times: (n_events,) sorted event times.
        event_states: (n_events, n_species) states at each event.
        time_grid: (T,) time grid to interpolate onto.

    Returns:
        (T, n_species) states at each grid time.
    """
    indices = torch.searchsorted(event_times, time_grid, right=True) - 1
    indices = indices.clamp(min=0, max=event_states.shape[0] - 1)
    return event_states[indices]
