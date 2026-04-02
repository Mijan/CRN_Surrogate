"""Tests for ViabilityFilter: one test per rejection criterion plus a passing case."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.data.generation.configs import CurationConfig
from crn_surrogate.data.generation.curation import CurationResult, ViabilityFilter

# Trajectory dimensions for all tests
_M = 4  # number of SSA trajectories
_T = 20  # number of timepoints
_S = 2  # number of species


@pytest.fixture
def default_config() -> CurationConfig:
    """CurationConfig with standard thresholds."""
    return CurationConfig()


@pytest.fixture
def viability_filter(default_config: CurationConfig) -> ViabilityFilter:
    """ViabilityFilter with default config."""
    return ViabilityFilter(default_config)


def _good_trajectory() -> torch.Tensor:
    """Create a well-behaved (M, T, S) trajectory with clear dynamics."""
    t = torch.linspace(0.0, 1.0, _T)
    base = 20.0 + 10.0 * torch.sin(t * 6.28).unsqueeze(0).unsqueeze(-1)
    noise = torch.randn(_M, _T, _S) * 2.0
    traj = (base + noise).abs().round()
    return traj


# --- Blowup rejection ---------------------------------------------------


def test_blowup_rejected(viability_filter: ViabilityFilter) -> None:
    """Trajectories with values above blowup_threshold are rejected."""
    traj = _good_trajectory()
    traj[0, 10, 0] = 2e6  # exceeds default 1e6 threshold
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "blowup"


def test_blowup_just_below_threshold_not_rejected(
    viability_filter: ViabilityFilter,
) -> None:
    """Trajectories just below blowup_threshold are not rejected for blowup."""
    traj = _good_trajectory()
    traj[0, 10, 0] = 9.9e5  # just below default 1e6
    result = viability_filter.check(traj)
    # May pass or fail other criteria, but not blowup
    assert result.rejection_reason != "blowup"


# --- Zero stuck rejection -----------------------------------------------


def test_zero_stuck_rejected(viability_filter: ViabilityFilter) -> None:
    """All-zero trajectories are rejected as stuck at zero."""
    traj = torch.zeros(_M, _T, _S)
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "zero_stuck"


# --- Low activity rejection ---------------------------------------------


def test_low_activity_rejected() -> None:
    """Trajectories with no state changes are rejected for low activity."""
    config = CurationConfig(min_reactions_fired=10)
    viability_filter = ViabilityFilter(config)
    # Constant non-zero trajectory → no transitions
    traj = torch.ones(_M, _T, _S) * 5.0
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "low_activity"


# --- Low CV rejection ---------------------------------------------------


def test_low_cv_rejected() -> None:
    """Near-constant trajectories are rejected for insufficient variation."""
    config = CurationConfig(min_coefficient_of_variation=0.1, min_reactions_fired=1)
    viability_filter = ViabilityFilter(config)
    # Trajectory stays at a constant high value (negligible CV), but has transitions
    # so it passes the low_activity check and fails only on CV.
    traj = torch.ones(_M, _T, _S) * 100.0
    # Add a single tiny perturbation to satisfy min_reactions_fired=1
    traj[0, 5, 0] = 101.0
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "low_cv"


# --- Unbounded final state rejection ------------------------------------


def test_unbounded_final_rejected() -> None:
    """Trajectories with large final mean are rejected."""
    config = CurationConfig(max_final_population=50.0)
    viability_filter = ViabilityFilter(config)
    # Build oscillating trajectory to pass CV but have huge final values
    traj = _good_trajectory()
    traj[:, -15:, :] = 200.0  # last 15 timepoints far above threshold
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "unbounded_final"


# --- NaN / Inf rejection ------------------------------------------------


def test_nan_rejected(viability_filter: ViabilityFilter) -> None:
    """Trajectories containing NaN are rejected."""
    traj = _good_trajectory()
    traj[1, 3, 1] = float("nan")
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "nan_or_inf"


def test_inf_rejected(viability_filter: ViabilityFilter) -> None:
    """Trajectories containing Inf are rejected."""
    traj = _good_trajectory()
    traj[0, 0, 0] = float("inf")
    result = viability_filter.check(traj)
    assert not result.viable
    assert result.rejection_reason == "nan_or_inf"


# --- Passing case -------------------------------------------------------


def test_good_trajectory_passes(viability_filter: ViabilityFilter) -> None:
    """A well-behaved trajectory ensemble passes all viability criteria."""
    traj = _good_trajectory()
    result = viability_filter.check(traj)
    assert result.viable
    assert result.rejection_reason == ""


# --- is_viable convenience method --------------------------------------


def test_is_viable_matches_viable_field(viability_filter: ViabilityFilter) -> None:
    """is_viable() returns the same boolean as check().viable."""
    traj = _good_trajectory()
    assert viability_filter.is_viable(traj) == viability_filter.check(traj).viable


# --- CurationResult dataclass -------------------------------------------


def test_curation_result_is_frozen() -> None:
    """CurationResult is a frozen dataclass (immutable)."""
    result = CurationResult(viable=True, rejection_reason="")
    with pytest.raises((AttributeError, TypeError)):
        result.viable = False  # type: ignore[misc]
