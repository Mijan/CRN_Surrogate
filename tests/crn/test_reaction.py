"""Tests for the Reaction dataclass."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.reaction import Reaction


def _make_prop():
    return lambda state, t: torch.tensor(1.0)


def test_valid_construction():
    prop = _make_prop()
    rxn = Reaction(
        stoichiometry=torch.tensor([1.0, -1.0]), propensity=prop, name="test"
    )
    assert torch.equal(rxn.stoichiometry, torch.tensor([1.0, -1.0]))
    assert rxn.propensity is prop
    assert rxn.name == "test"


def test_rejects_2d_stoichiometry():
    with pytest.raises(ValueError, match="1D"):
        Reaction(
            stoichiometry=torch.zeros(2, 3),
            propensity=_make_prop(),
        )


def test_rejects_non_callable_propensity():
    with pytest.raises(ValueError, match="callable"):
        Reaction(stoichiometry=torch.tensor([1.0]), propensity=42)  # type: ignore[arg-type]


def test_equality_same_propensity_fn():
    prop = _make_prop()
    stoich = torch.tensor([1.0, -1.0])
    rxn1 = Reaction(stoichiometry=stoich, propensity=prop, name="r")
    rxn2 = Reaction(stoichiometry=stoich, propensity=prop, name="r")
    assert rxn1 == rxn2


def test_inequality_different_propensity_fn():
    stoich = torch.tensor([1.0])
    # Functionally identical but different objects
    rxn1 = Reaction(stoichiometry=stoich, propensity=lambda s, t: torch.tensor(1.0))
    rxn2 = Reaction(stoichiometry=stoich, propensity=lambda s, t: torch.tensor(1.0))
    assert rxn1 != rxn2


def test_frozen():
    rxn = Reaction(stoichiometry=torch.tensor([1.0]), propensity=_make_prop())
    with pytest.raises((AttributeError, TypeError)):
        rxn.name = "x"  # type: ignore[misc]
