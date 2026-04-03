"""Evaluation utilities for trained CRN neural surrogate models.

Provides reusable diagnostics: rollout generation, learned dynamics
visualization, residual analysis, trajectory-level statistics, and
analytical CLE references for known motifs.
"""

from crn_surrogate.evaluation.analytical import (
    birth_death_analytical,
    lotka_volterra_analytical,
)
from crn_surrogate.evaluation.dynamics import DynamicsProfile, DynamicsVisualizer
from crn_surrogate.evaluation.residuals import ResidualAnalyzer, ResidualReport
from crn_surrogate.evaluation.rollout import ModelEvaluator
from crn_surrogate.evaluation.trajectory import TrajectoryComparator

__all__ = [
    "DynamicsProfile",
    "DynamicsVisualizer",
    "ModelEvaluator",
    "ResidualAnalyzer",
    "ResidualReport",
    "TrajectoryComparator",
    "birth_death_analytical",
    "lotka_volterra_analytical",
]
