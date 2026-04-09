"""Tests for crn_to_tensor_repr, tensor_repr_to_crn, CRNTensorRepr, PropensityType."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.encoder.tensor_repr import (
    CRNTensorRepr,
    PropensityType,
    crn_to_tensor_repr,
    tensor_repr_to_crn,
)

# ── crn_to_tensor_repr ────────────────────────────────────────────────────────


def test_birth_death_shapes(birth_death_repr: CRNTensorRepr) -> None:
    r = birth_death_repr
    assert r.stoichiometry.shape == (2, 1)
    assert r.propensity_type_ids.shape == (2,)
    assert r.propensity_params.shape == (2, 8)
    assert r.dependency_matrix.shape == (2, 1)


def test_birth_death_stoichiometry_values(birth_death_repr: CRNTensorRepr) -> None:
    r = birth_death_repr
    assert r.stoichiometry[0, 0].item() == pytest.approx(1.0)
    assert r.stoichiometry[1, 0].item() == pytest.approx(-1.0)


def test_birth_death_type_ids(birth_death_repr: CRNTensorRepr) -> None:
    r = birth_death_repr
    assert int(r.propensity_type_ids[0].item()) == PropensityType.CONSTANT_RATE.value
    assert int(r.propensity_type_ids[1].item()) == PropensityType.MASS_ACTION.value


def test_dependency_matrix_birth_death(birth_death_repr: CRNTensorRepr) -> None:
    r = birth_death_repr
    # Birth (constant rate): no dependency on species 0
    assert r.dependency_matrix[0, 0].item() == pytest.approx(0.0)
    # Death (mass action on species 0): depends on species 0
    assert r.dependency_matrix[1, 0].item() == pytest.approx(1.0)


def test_two_species_shapes(two_species_repr: CRNTensorRepr) -> None:
    r = two_species_repr
    assert r.stoichiometry.shape == (2, 2)
    assert r.propensity_params.shape == (2, 8)


def test_is_external_default_false(birth_death_repr: CRNTensorRepr) -> None:
    r = birth_death_repr
    assert r.is_external is not None
    assert not r.is_external.any()


def test_species_names_preserved(
    birth_death_crn, birth_death_repr: CRNTensorRepr
) -> None:
    assert birth_death_repr.species_names == birth_death_crn.species_names


# ── CRNTensorRepr properties ─────────────────────────────────────────────────


def test_n_species(birth_death_repr: CRNTensorRepr) -> None:
    assert birth_death_repr.n_species == birth_death_repr.stoichiometry.shape[1]


def test_n_reactions(birth_death_repr: CRNTensorRepr) -> None:
    assert birth_death_repr.n_reactions == birth_death_repr.stoichiometry.shape[0]


def test_bipartite_edges_cached(birth_death_repr: CRNTensorRepr) -> None:
    edges1 = birth_death_repr.bipartite_edges
    edges2 = birth_death_repr.bipartite_edges
    assert edges1 is edges2


def test_to_device_same_device_returns_self(birth_death_repr: CRNTensorRepr) -> None:
    cpu = torch.device("cpu")
    result = birth_death_repr.to(cpu)
    assert result is birth_death_repr


# ── tensor_repr_to_crn round-trip ─────────────────────────────────────────────


def test_round_trip_birth_death(birth_death_crn) -> None:
    repr_ = crn_to_tensor_repr(birth_death_crn)
    reconstructed = tensor_repr_to_crn(repr_)

    assert reconstructed.n_species == birth_death_crn.n_species
    assert reconstructed.n_reactions == birth_death_crn.n_reactions

    state = torch.tensor([5.0])
    for i in range(birth_death_crn.n_reactions):
        orig = birth_death_crn.reactions[i].propensity(state, 0.0)
        recon = reconstructed.reactions[i].propensity(state, 0.0)
        assert orig.item() == pytest.approx(recon.item(), abs=1e-5)


def test_round_trip_two_species(two_species_crn) -> None:
    repr_ = crn_to_tensor_repr(two_species_crn)
    reconstructed = tensor_repr_to_crn(repr_)

    assert reconstructed.n_species == two_species_crn.n_species
    assert reconstructed.n_reactions == two_species_crn.n_reactions

    state = torch.tensor([3.0, 7.0])
    for i in range(two_species_crn.n_reactions):
        orig = two_species_crn.reactions[i].propensity(state, 0.0)
        recon = reconstructed.reactions[i].propensity(state, 0.0)
        assert orig.item() == pytest.approx(recon.item(), abs=1e-5)
