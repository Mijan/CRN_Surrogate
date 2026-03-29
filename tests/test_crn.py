"""Tests for CRN datastructures and edge construction."""

from crn_surrogate.data.crn import build_bipartite_edges
from crn_surrogate.data.gillespie import birth_death_crn, lotka_volterra_crn
from crn_surrogate.data.propensities import PropensityType


def test_birth_death_crn_shape():
    crn = birth_death_crn()
    assert crn.n_species == 1
    assert crn.n_reactions == 2
    assert crn.stoichiometry.shape == (2, 1)
    assert crn.reactant_matrix.shape == (2, 1)


def test_lotka_volterra_shape():
    crn = lotka_volterra_crn()
    assert crn.n_species == 2
    assert crn.n_reactions == 3


def test_build_bipartite_edges_birth_death():
    crn = birth_death_crn()
    edges = build_bipartite_edges(crn.stoichiometry, crn.reactant_matrix)
    # Birth: ∅→A has stoich=+1, reactant=0 → edge (rxn=0, species=0)
    # Death: A→∅ has stoich=-1, reactant=1 → edge (rxn=1, species=0)
    assert edges.rxn_to_species_index.shape[0] == 2
    assert edges.rxn_to_species_feat.shape[1] == 2


def test_crn_repr():
    crn = birth_death_crn()
    r = repr(crn)
    assert "n_species=1" in r
    assert "n_reactions=2" in r


def test_propensity_types_stored():
    crn = birth_death_crn()
    assert all(pt == PropensityType.MASS_ACTION for pt in crn.propensity_types)
