"""Tests for the CRN and Reaction classes, propensity evaluation, and tensor repr round-trip.

Covers:
- Reaction construction and validation (1D stoichiometry, callable propensity).
- CRN construction, properties, stoichiometry_matrix assembly.
- CRN.evaluate_propensities correctness.
- Immutability of Reaction and CRN fields.
- Example CRN factories produce structurally valid CRNs.
- Round-trip: CRN → CRNTensorRepr → CRN preserves kinetics.
"""

import pytest
import torch

from crn_surrogate.crn import CRN, Reaction
from crn_surrogate.crn.examples import (
    birth_death,
    lotka_volterra,
    schlogl,
    simple_mapk_cascade,
    toggle_switch,
)
from crn_surrogate.encoder.tensor_repr import (
    crn_to_tensor_repr,
    tensor_repr_to_crn,
)

# ── Reaction construction ──────────────────────────────────────────────────────


def test_reaction_requires_1d_stoichiometry():
    """Reaction must reject a 2D stoichiometry tensor."""
    with pytest.raises(ValueError, match="1D"):
        Reaction(
            stoichiometry=torch.tensor([[1, 0]]),  # 2D — invalid
            propensity=lambda state, t: torch.tensor(1.0),
        )


def test_reaction_requires_callable_propensity():
    """Reaction must reject a non-callable propensity."""
    with pytest.raises(ValueError, match="callable"):
        Reaction(
            stoichiometry=torch.tensor([1]),
            propensity=42,  # type: ignore[arg-type]
        )


def test_reaction_stores_name_and_stoichiometry():
    """Reaction stores name and stoichiometry exactly as provided."""
    stoich = torch.tensor([1, -1])
    rxn = Reaction(
        stoichiometry=stoich, propensity=lambda s, t: torch.tensor(1.0), name="test"
    )
    assert rxn.name == "test"
    assert torch.equal(rxn.stoichiometry, stoich)


def test_reaction_repr_contains_name_and_stoichiometry():
    """Reaction.__repr__ includes the name and stoichiometry list."""
    rxn = Reaction(
        stoichiometry=torch.tensor([1]),
        propensity=lambda s, t: torch.tensor(1.0),
        name="birth",
    )
    r = repr(rxn)
    assert "birth" in r
    assert "1" in r


# ── CRN construction ───────────────────────────────────────────────────────────


def test_crn_rejects_empty_reaction_list():
    """CRN must raise ValueError when given an empty reaction list."""
    with pytest.raises(ValueError, match="at least one"):
        CRN(reactions=[])


def test_crn_rejects_mismatched_stoichiometry_lengths():
    """All reactions must have the same stoichiometry length."""
    with pytest.raises(ValueError):
        CRN(
            reactions=[
                Reaction(torch.tensor([1]), lambda s, t: torch.tensor(1.0)),
                Reaction(torch.tensor([1, 0]), lambda s, t: torch.tensor(1.0)),
            ]
        )


def test_crn_rejects_wrong_species_names_length():
    """species_names must have exactly n_species entries."""
    with pytest.raises(ValueError):
        CRN(
            reactions=[Reaction(torch.tensor([1, 0]), lambda s, t: torch.tensor(1.0))],
            species_names=["A", "B", "C"],  # too many
        )


def test_crn_default_species_names_are_s0_s1():
    """When species_names is omitted, defaults to S0, S1, ..."""
    crn = CRN(
        reactions=[Reaction(torch.tensor([1, 0]), lambda s, t: torch.tensor(1.0))]
    )
    assert crn.species_names == ("S0", "S1")


def test_crn_n_species_and_n_reactions():
    """CRN properties n_species and n_reactions return correct values."""
    crn = birth_death()
    assert crn.n_species == 1
    assert crn.n_reactions == 2


def test_crn_stoichiometry_matrix_shape():
    """stoichiometry_matrix has shape (n_reactions, n_species)."""
    crn = birth_death()
    assert crn.stoichiometry_matrix.shape == (2, 1)


def test_crn_stoichiometry_matrix_values():
    """birth_death stoichiometry matrix has +1 and -1 entries."""
    crn = birth_death()
    mat = crn.stoichiometry_matrix  # (2, 1)
    values = set(mat[:, 0].tolist())
    assert 1.0 in values
    assert -1.0 in values


def test_crn_stoichiometry_matrix_is_cached():
    """stoichiometry_matrix is the same object on repeated access."""
    crn = birth_death()
    m1 = crn.stoichiometry_matrix
    m2 = crn.stoichiometry_matrix
    assert m1 is m2


def test_crn_reactions_returns_tuple():
    """crn.reactions returns a tuple, which is immutable."""
    crn = birth_death()
    assert isinstance(crn.reactions, tuple)
    with pytest.raises((AttributeError, TypeError)):
        crn.reactions[0] = None  # type: ignore[index]


def test_crn_reaction_access_by_index():
    """CRN.reaction(i) returns the i-th reaction."""
    crn = birth_death()
    assert crn.reaction(0) is crn.reactions[0]
    assert crn.reaction(1) is crn.reactions[1]


# ── evaluate_propensities ─────────────────────────────────────────────────────


def test_evaluate_propensities_shape():
    """evaluate_propensities returns a tensor of shape (n_reactions,)."""
    crn = birth_death(k_birth=1.0, k_death=0.1)
    a = crn.evaluate_propensities(torch.tensor([5.0]))
    assert a.shape == (2,)


def test_evaluate_propensities_birth_death_values():
    """Birth rate = k_birth (zero-order); death rate = k_death * state."""
    k_birth, k_death, state_val = 2.0, 0.5, 4.0
    crn = birth_death(k_birth=k_birth, k_death=k_death)
    state = torch.tensor([state_val])
    a = crn.evaluate_propensities(state)
    assert a[0].item() == pytest.approx(k_birth)
    assert a[1].item() == pytest.approx(k_death * state_val)


def test_evaluate_propensities_clamps_to_nonnegative():
    """Propensities must be non-negative even if the propensity fn returns negative."""
    crn = CRN(reactions=[Reaction(torch.tensor([1]), lambda s, t: torch.tensor(-1.0))])
    a = crn.evaluate_propensities(torch.tensor([5.0]))
    assert a[0].item() == pytest.approx(0.0)


# ── Example CRN factories ─────────────────────────────────────────────────────


def test_birth_death_factory_structure():
    crn = birth_death(k_birth=1.0, k_death=0.1)
    assert crn.n_species == 1
    assert crn.n_reactions == 2
    assert crn.species_names == ("A",)


def test_lotka_volterra_factory_structure():
    crn = lotka_volterra()
    assert crn.n_species == 2
    assert crn.n_reactions == 3
    assert crn.species_names == ("prey", "predator")


def test_schlogl_factory_structure():
    crn = schlogl()
    assert crn.n_species == 1
    assert crn.n_reactions == 4


def test_toggle_switch_factory_structure():
    crn = toggle_switch()
    assert crn.n_species == 2
    assert crn.n_reactions == 4


def test_simple_mapk_cascade_factory_structure():
    crn = simple_mapk_cascade()
    assert crn.n_species == 7
    assert crn.n_reactions == 6


# ── Bipartite edges ───────────────────────────────────────────────────────────


def test_bipartite_edges_birth_death_two_edges():
    """Birth-death: each reaction touches species A, so 2 reaction→species edges."""
    crn = birth_death()
    repr_ = crn_to_tensor_repr(crn)
    edges = repr_.bipartite_edges
    assert edges.rxn_to_species_index.shape[1] == 2


def test_bipartite_edges_feature_dimension_is_three():
    """Edge features encode (net_change, is_stoichiometric, is_dependency), feature dim == 3."""
    crn = birth_death()
    repr_ = crn_to_tensor_repr(crn)
    edges = repr_.bipartite_edges
    assert edges.rxn_to_species_feat.shape[1] == 3


def test_bipartite_edges_lotka_volterra_has_more_edges_than_birth_death():
    """Lotka-Volterra has more edges (4) than birth-death (2)."""
    bd_repr = crn_to_tensor_repr(birth_death())
    lv_repr = crn_to_tensor_repr(lotka_volterra())
    edges_bd = bd_repr.bipartite_edges
    edges_lv = lv_repr.bipartite_edges
    assert edges_bd.rxn_to_species_index.shape[1] == 2
    assert edges_lv.rxn_to_species_index.shape[1] == 4


# ── Tensor repr round-trip ────────────────────────────────────────────────────


def test_tensor_repr_round_trip_birth_death_propensity_values():
    """CRN → tensor_repr → CRN: propensity values are preserved at a test state."""
    original = birth_death(k_birth=2.0, k_death=0.3)
    state = torch.tensor([5.0])

    reconstructed = tensor_repr_to_crn(crn_to_tensor_repr(original))
    a_orig = original.evaluate_propensities(state)
    a_recon = reconstructed.evaluate_propensities(state)

    torch.testing.assert_close(a_orig, a_recon, atol=1e-5, rtol=1e-5)


def test_tensor_repr_round_trip_toggle_switch():
    """Round-trip preserves propensity values for a two-species Hill + mass-action CRN."""
    original = toggle_switch(alpha1=5.0, alpha2=3.0, beta=1.0, hill_n=2.0)
    state = torch.tensor([4.0, 2.0])

    reconstructed = tensor_repr_to_crn(crn_to_tensor_repr(original))
    a_orig = original.evaluate_propensities(state)
    a_recon = reconstructed.evaluate_propensities(state)

    torch.testing.assert_close(a_orig, a_recon, atol=1e-5, rtol=1e-5)


def test_tensor_repr_raises_for_non_serializable_propensity():
    """crn_to_tensor_repr must raise ValueError for custom lambda propensities."""
    crn = CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1]),
                propensity=lambda state, t: torch.tensor(1.0),
            )
        ]
    )
    with pytest.raises(ValueError, match="non-serializable"):
        crn_to_tensor_repr(crn)


def test_crn_repr_contains_species_and_reaction_counts():
    crn = birth_death()
    r = repr(crn)
    assert "n_species=1" in r
    assert "n_reactions=2" in r
