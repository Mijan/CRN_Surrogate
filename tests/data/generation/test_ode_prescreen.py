"""Tests for ODEPreScreen."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.configs import ODEPreScreenConfig
from crn_surrogate.data.generation.ode_prescreen import DynamicsType, ODEPreScreen
from crn_surrogate.data.generation.reference_crns import birth_death


def _make_screener() -> ODEPreScreen:
    return ODEPreScreen(ODEPreScreenConfig())


def test_accepts_birth_death() -> None:
    screener = _make_screener()
    crn = birth_death(k_birth=2.0, k_death=0.1)
    initial = torch.tensor([0.0])
    result = screener.check(crn, initial)
    assert result.accepted


def test_rejects_pure_decay() -> None:
    screener = _make_screener()
    # Only degradation: X -> ∅, starting below min_sustained_level so the
    # classifier sees DECAY_TO_ZERO rather than TRANSIENT_PEAK.
    crn = CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(0.5, torch.tensor([1.0])),
                name="decay",
            )
        ]
    )
    initial = torch.tensor([0.1])  # below min_sustained_level (0.5)
    result = screener.check(crn, initial)
    assert not result.accepted


def test_rejects_blowup() -> None:
    screener = _make_screener()
    # Very high production, no degradation → fast blowup
    crn = CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(1e6),
                name="production",
            )
        ]
    )
    initial = torch.tensor([0.0])
    result = screener.check(crn, initial)
    assert not result.accepted
    assert result.dynamics_type == DynamicsType.BLOWUP


def test_result_has_dynamics_type() -> None:
    screener = _make_screener()
    crn = birth_death()
    initial = torch.tensor([5.0])
    result = screener.check(crn, initial)
    assert isinstance(result.dynamics_type, DynamicsType)
