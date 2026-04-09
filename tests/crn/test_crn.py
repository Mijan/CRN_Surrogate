"""Tests for the CRN class."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction


def _birth_death_reactions() -> list[Reaction]:
    """Return birth-death reactions: ∅ -> X (const), X -> ∅ (mass action)."""
    return [
        Reaction(
            stoichiometry=torch.tensor([1.0]),
            propensity=constant_rate(2.0),
            name="birth",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0]),
            propensity=mass_action(0.5, torch.tensor([1.0])),
            name="death",
        ),
    ]


# ── Construction ──────────────────────────────────────────────────────────────


def test_basic_construction():
    crn = CRN(_birth_death_reactions())
    assert crn.n_species == 1
    assert crn.n_reactions == 2


def test_default_species_names():
    crn = CRN(_birth_death_reactions())
    assert crn.species_names == ("S0",)


def test_custom_species_names():
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=constant_rate(1.0)),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]), propensity=constant_rate(1.0)
        ),
    ]
    crn = CRN(rxns, species_names=["A", "B"])
    assert crn.species_names == ("A", "B")


def test_rejects_empty_reactions():
    with pytest.raises(ValueError):
        CRN([])


def test_rejects_mismatched_stoichiometry():
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=constant_rate(1.0)),
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0, 0.0]), propensity=constant_rate(1.0)
        ),
    ]
    with pytest.raises(ValueError):
        CRN(rxns)


def test_rejects_wrong_species_names_length():
    with pytest.raises(ValueError):
        CRN(_birth_death_reactions(), species_names=["A", "B", "C"])


# ── Stoichiometry matrix ──────────────────────────────────────────────────────


def test_stoichiometry_matrix_shape():
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=constant_rate(1.0)),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]), propensity=constant_rate(1.0)
        ),
        Reaction(stoichiometry=torch.tensor([0.0, 1.0]), propensity=constant_rate(1.0)),
    ]
    crn = CRN(rxns)
    assert crn.stoichiometry_matrix.shape == (3, 2)


def test_stoichiometry_matrix_values():
    crn = CRN(_birth_death_reactions())
    expected = torch.tensor([[1.0], [-1.0]])
    assert torch.allclose(crn.stoichiometry_matrix, expected)


def test_stoichiometry_matrix_cached():
    crn = CRN(_birth_death_reactions())
    m1 = crn.stoichiometry_matrix
    m2 = crn.stoichiometry_matrix
    assert m1 is m2


# ── Propensity evaluation ─────────────────────────────────────────────────────


def test_evaluate_propensities_shape():
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0]), propensity=constant_rate(1.0)),
        Reaction(stoichiometry=torch.tensor([0.0]), propensity=constant_rate(2.0)),
        Reaction(stoichiometry=torch.tensor([-1.0]), propensity=constant_rate(3.0)),
    ]
    crn = CRN(rxns)
    result = crn.evaluate_propensities(torch.tensor([5.0]))
    assert result.shape == (3,)


def test_evaluate_propensities_non_negative():
    # Use a propensity that returns negative to test clamping
    neg_prop = lambda state, t: torch.tensor(-2.0)  # noqa: E731
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0]), propensity=neg_prop),
        Reaction(stoichiometry=torch.tensor([-1.0]), propensity=constant_rate(1.0)),
    ]
    crn = CRN(rxns)
    result = crn.evaluate_propensities(torch.tensor([5.0]))
    assert torch.all(result >= 0.0)


def test_evaluate_propensities_correct_values():
    crn = CRN(_birth_death_reactions())
    state = torch.tensor([8.0])
    result = crn.evaluate_propensities(state)
    assert result[0].item() == pytest.approx(2.0)  # birth: constant rate 2.0
    assert result[1].item() == pytest.approx(4.0)  # death: 0.5 * 8.0


# ── Dependency matrix ─────────────────────────────────────────────────────────


def test_dependency_matrix_shape():
    rxns = [
        Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=constant_rate(1.0)),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]), propensity=constant_rate(1.0)
        ),
        Reaction(stoichiometry=torch.tensor([0.0, 1.0]), propensity=constant_rate(1.0)),
    ]
    crn = CRN(rxns)
    assert crn.dependency_matrix.shape == (3, 2)


def test_dependency_matrix_values():
    crn = CRN(_birth_death_reactions())
    dep = crn.dependency_matrix
    # Birth: constant_rate has no dependencies -> all zeros
    assert torch.all(dep[0] == 0.0)
    # Death: mass_action on species 0 -> dep at species 0 is 1, others 0
    assert dep[1, 0].item() == 1.0


# ── External species ──────────────────────────────────────────────────────────


def test_external_species_empty_by_default():
    crn = CRN(_birth_death_reactions())
    assert crn.external_species == frozenset()


def test_external_species_valid():
    # 3 species: species 2 is external (no net stoichiometric change)
    rxns = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0, 0.0]), propensity=constant_rate(1.0)
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0, 0.0]),
            propensity=mass_action(0.5, torch.tensor([1.0, 0.0, 0.0])),
        ),
    ]
    crn = CRN(rxns, external_species=frozenset({2}))
    assert crn.external_species == frozenset({2})
    assert crn.n_external_species == 1
    assert crn.is_external[2] == True  # noqa: E712
    assert crn.internal_species_mask[2] == False  # noqa: E712


def test_external_species_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        CRN(_birth_death_reactions(), external_species=frozenset({5}))


def test_external_species_rejects_nonzero_stoichiometry():
    # Species 0 is external but has net change +1 in birth reaction
    with pytest.raises(ValueError, match="nonzero"):
        CRN(_birth_death_reactions(), external_species=frozenset({0}))
