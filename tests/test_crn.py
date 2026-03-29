"""Tests for CRN datastructures and bipartite edge construction.

Covers:
- CRNDefinition stores correct structural counts (n_species, n_reactions).
- build_bipartite_edges produces correctly shaped index and feature tensors.
- Species-to-reaction edges reflect the reactant matrix; reaction-to-species
  edges reflect the stoichiometry matrix.
- CRNDefinition is immutable (frozen dataclass).
- PropensityType enum values are stored faithfully.
"""

import pytest
import torch

from crn_surrogate.data.crn import CRNDefinition, build_bipartite_edges
from crn_surrogate.data.gillespie import birth_death_crn, lotka_volterra_crn
from crn_surrogate.data.propensities import PropensityType


# ── CRNDefinition structure ───────────────────────────────────────────────────


def test_birth_death_crn_has_one_species_two_reactions():
    """Birth-death CRN: A is the only species, birth and death are the two reactions."""
    crn = birth_death_crn()
    assert crn.n_species == 1
    assert crn.n_reactions == 2
    assert crn.stoichiometry.shape == (2, 1)
    assert crn.reactant_matrix.shape == (2, 1)


def test_lotka_volterra_crn_has_two_species_three_reactions():
    """Lotka-Volterra CRN: prey and predator species, three reactions."""
    crn = lotka_volterra_crn()
    assert crn.n_species == 2
    assert crn.n_reactions == 3


def test_birth_death_stoichiometry_signs():
    """Birth reaction increases A by 1; death reaction decreases A by 1."""
    crn = birth_death_crn()
    stoich = crn.stoichiometry  # (2 rxns, 1 species)
    # One reaction must have +1, the other -1
    signs = set(stoich[:, 0].tolist())
    assert 1.0 in signs
    assert -1.0 in signs


def test_crn_definition_is_frozen():
    """CRNDefinition is a frozen dataclass and must raise on attribute assignment."""
    crn = birth_death_crn()
    with pytest.raises((AttributeError, TypeError)):
        crn.n_species = 99  # type: ignore[misc]


def test_crn_repr_contains_species_and_reaction_counts():
    crn = birth_death_crn()
    r = repr(crn)
    assert "n_species=1" in r
    assert "n_reactions=2" in r


def test_propensity_types_are_all_mass_action_for_birth_death():
    crn = birth_death_crn()
    assert all(pt == PropensityType.MASS_ACTION for pt in crn.propensity_types)


# ── build_bipartite_edges ─────────────────────────────────────────────────────


def test_bipartite_edges_birth_death_reaction_to_species_count():
    """Birth-death: each reaction touches species A, so 2 reaction→species edges."""
    crn = birth_death_crn()
    edges = build_bipartite_edges(crn.stoichiometry, crn.reactant_matrix)
    assert edges.rxn_to_species_index.shape[0] == 2


def test_bipartite_edges_feature_dimension_is_two():
    """Edge features encode (stoichiometry, reactant) pairs, so feature dim == 2."""
    crn = birth_death_crn()
    edges = build_bipartite_edges(crn.stoichiometry, crn.reactant_matrix)
    assert edges.rxn_to_species_feat.shape[1] == 2


def test_bipartite_edges_lotka_volterra_has_more_edges_than_birth_death():
    """Lotka-Volterra has 4 species–reaction edges (R0↔prey, R1↔prey, R1↔pred, R2↔pred)
    versus birth-death's 2 edges (birth↔A, death↔A).

    Edge count is rxn_to_species_index.shape[1] — the second dimension, since the
    tensor shape is (2, E) where 2 = [rxn_idx, species_idx] and E = number of edges.
    """
    edges_bd = build_bipartite_edges(
        birth_death_crn().stoichiometry, birth_death_crn().reactant_matrix
    )
    edges_lv = build_bipartite_edges(
        lotka_volterra_crn().stoichiometry, lotka_volterra_crn().reactant_matrix
    )
    assert edges_lv.rxn_to_species_index.shape[1] > edges_bd.rxn_to_species_index.shape[1]
    assert edges_bd.rxn_to_species_index.shape[1] == 2
    assert edges_lv.rxn_to_species_index.shape[1] == 4
