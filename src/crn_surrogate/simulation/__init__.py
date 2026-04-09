"""simulation: exact stochastic simulation and trajectory utilities."""

from crn_surrogate.simulation.data_simulator import (
    DataSimulator,
    FastSSASimulator,
    ODESimulator,
    SSASimulator,
)
from crn_surrogate.simulation.fast_ssa import FastMassActionSSA
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.mass_action_ode import MassActionODE
from crn_surrogate.simulation.timegrid_utils import TimegridUtils
from crn_surrogate.simulation.trajectory import Trajectory

# Convenience alias for the most common use of TimegridUtils
interpolate_to_grid = TimegridUtils.interpolate_to_grid

__all__ = [
    "DataSimulator",
    "FastMassActionSSA",
    "FastSSASimulator",
    "GillespieSSA",
    "MassActionODE",
    "ODESimulator",
    "SSASimulator",
    "Trajectory",
    "TimegridUtils",
    "interpolate_to_grid",
]
