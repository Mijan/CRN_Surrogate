"""simulation: exact stochastic simulation and trajectory utilities."""

from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.interpolation import TimegridUtils
from crn_surrogate.simulation.trajectory import Trajectory

# Convenience alias for the most common use of TimegridUtils
interpolate_to_grid = TimegridUtils.interpolate_to_grid

__all__ = ["GillespieSSA", "Trajectory", "TimegridUtils", "interpolate_to_grid"]
