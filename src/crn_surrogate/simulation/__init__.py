"""simulation: exact stochastic simulation and trajectory utilities."""

from crn_surrogate.simulation.fast_ssa import FastMassActionSSA
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.timegrid_utils import TimegridUtils
from crn_surrogate.simulation.trajectory import Trajectory

# Convenience alias for the most common use of TimegridUtils
interpolate_to_grid = TimegridUtils.interpolate_to_grid

__all__ = [
    "FastMassActionSSA",
    "GillespieSSA",
    "Trajectory",
    "TimegridUtils",
    "interpolate_to_grid",
]
