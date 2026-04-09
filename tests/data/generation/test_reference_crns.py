"""Tests for named reference CRN constructors."""

from __future__ import annotations

import torch

from crn_surrogate.data.generation.reference_crns import (
    birth_death,
    lotka_volterra,
    schlogl,
    simple_mapk_cascade,
    toggle_switch,
)


def test_birth_death_structure() -> None:
    crn = birth_death()
    assert crn.n_species == 1
    assert crn.n_reactions == 2


def test_lotka_volterra_structure() -> None:
    crn = lotka_volterra()
    assert crn.n_species == 2
    assert crn.n_reactions == 3


def test_toggle_switch_structure() -> None:
    crn = toggle_switch()
    assert crn.n_species == 2
    assert crn.n_reactions == 4


def test_schlogl_structure() -> None:
    crn = schlogl()
    assert crn.n_species == 1
    assert crn.n_reactions == 4


def test_mapk_cascade_structure() -> None:
    crn = simple_mapk_cascade()
    assert crn.n_species == 7
    assert crn.n_reactions == 6


def test_all_reference_crns_evaluable() -> None:
    crns_and_states = [
        (birth_death(), torch.tensor([5.0])),
        (lotka_volterra(), torch.tensor([50.0, 20.0])),
        (toggle_switch(), torch.tensor([5.0, 5.0])),
        (schlogl(), torch.tensor([50.0])),
        (simple_mapk_cascade(), torch.tensor([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 2.0])),
    ]
    for crn, state in crns_and_states:
        props = crn.evaluate_propensities(state, 0.0)
        assert props.shape == (crn.n_reactions,)
        assert props.isfinite().all()
        assert (props >= 0).all()
