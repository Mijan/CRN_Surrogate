"""Tests for propensity factory functions and their closures."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.propensities import (
    ConstantRateParams,
    HillParams,
    MassActionParams,
    SerializablePropensity,
    constant_rate,
    enzyme_michaelis_menten,
    hill,
    hill_activation_repression,
    hill_repression,
    mass_action,
    substrate_inhibition,
)

# ── Mass action ───────────────────────────────────────────────────────────────


def test_mass_action_first_order():
    prop = mass_action(0.5, torch.tensor([1.0, 0.0]))
    result = prop(torch.tensor([10.0, 20.0]), 0.0)
    assert result.item() == pytest.approx(5.0, abs=1e-5)


def test_mass_action_second_order():
    prop = mass_action(0.1, torch.tensor([1.0, 1.0]))
    result = prop(torch.tensor([10.0, 20.0]), 0.0)
    assert result.item() == pytest.approx(20.0, abs=1e-4)


def test_mass_action_zero_state():
    prop = mass_action(0.5, torch.tensor([1.0, 0.0]))
    result = prop(torch.tensor([0.0, 5.0]), 0.0)
    assert result.item() == pytest.approx(0.0)


def test_mass_action_params():
    prop = mass_action(2.5, torch.tensor([1.0]))
    assert isinstance(prop.params, MassActionParams)
    assert prop.params.rate_constant == pytest.approx(2.5)


def test_mass_action_dependencies():
    prop = mass_action(1.0, torch.tensor([1.0, 0.0, 1.0]))
    assert prop.species_dependencies == {0, 2}


# ── Constant rate ─────────────────────────────────────────────────────────────


def test_constant_rate_value():
    prop = constant_rate(3.0)
    result = prop(torch.tensor([0.0, 100.0, 50.0]), 0.0)
    assert result.item() == pytest.approx(3.0)


def test_constant_rate_ignores_state():
    prop = constant_rate(3.0)
    r1 = prop(torch.tensor([0.0]), 0.0)
    r2 = prop(torch.tensor([999.0]), 5.0)
    assert r1.item() == pytest.approx(r2.item())


def test_constant_rate_no_dependencies():
    prop = constant_rate(1.0)
    assert prop.species_dependencies == frozenset()


# ── Hill activation ───────────────────────────────────────────────────────────


def test_hill_at_half_saturation():
    # At x = K_m the Hill function should be ~v_max / 2
    prop = hill(v_max=10.0, k_m=5.0, hill_coefficient=2.0, species_index=0)
    result = prop(torch.tensor([5.0]), 0.0)
    assert result.item() == pytest.approx(5.0, abs=0.5)


def test_hill_at_zero():
    prop = hill(v_max=10.0, k_m=5.0, hill_coefficient=2.0, species_index=0)
    result = prop(torch.tensor([0.0]), 0.0)
    assert result.item() < 0.5


def test_hill_at_saturation():
    prop = hill(v_max=10.0, k_m=5.0, hill_coefficient=4.0, species_index=0)
    result = prop(torch.tensor([1000.0]), 0.0)
    assert result.item() == pytest.approx(10.0, abs=0.5)


def test_hill_dependencies():
    prop = hill(v_max=1.0, k_m=1.0, hill_coefficient=1.0, species_index=2)
    assert prop.species_dependencies == {2}


# ── Hill repression ───────────────────────────────────────────────────────────


def test_hill_repression_at_zero():
    prop = hill_repression(
        k_max=5.0, k_half=10.0, hill_coefficient=2.0, species_index=0
    )
    result = prop(torch.tensor([0.0]), 0.0)
    assert result.item() == pytest.approx(5.0, abs=0.5)


def test_hill_repression_at_high_concentration():
    prop = hill_repression(
        k_max=5.0, k_half=10.0, hill_coefficient=4.0, species_index=0
    )
    result = prop(torch.tensor([1000.0]), 0.0)
    assert result.item() < 0.1


def test_hill_repression_dependencies():
    prop = hill_repression(k_max=1.0, k_half=1.0, hill_coefficient=1.0, species_index=3)
    assert prop.species_dependencies == {3}


# ── Enzyme Michaelis-Menten ───────────────────────────────────────────────────


def test_enzyme_mm_value():
    prop = enzyme_michaelis_menten(
        k_cat=1.0, k_m=10.0, enzyme_index=0, substrate_index=1
    )
    state = torch.tensor([5.0, 20.0])
    result = prop(state, 0.0)
    expected = 1.0 * 5.0 * 20.0 / (10.0 + 20.0)  # ≈ 3.33
    assert result.item() == pytest.approx(expected, abs=0.1)


def test_enzyme_mm_no_enzyme():
    prop = enzyme_michaelis_menten(
        k_cat=1.0, k_m=10.0, enzyme_index=0, substrate_index=1
    )
    result = prop(torch.tensor([0.0, 20.0]), 0.0)
    assert result.item() == pytest.approx(0.0, abs=1e-4)


def test_enzyme_mm_dependencies():
    prop = enzyme_michaelis_menten(
        k_cat=1.0, k_m=1.0, enzyme_index=1, substrate_index=3
    )
    assert prop.species_dependencies == {1, 3}


# ── Substrate inhibition ──────────────────────────────────────────────────────


def test_substrate_inhibition_low_substrate():
    prop = substrate_inhibition(v_max=10.0, k_m=5.0, k_i=1000.0, species_index=0)
    low = prop(torch.tensor([0.1]), 0.0)
    # At very low substrate, denominator ≈ k_m, so rate ≈ v_max * x / k_m (linear)
    expected = 10.0 * 0.1 / (5.0 + 0.1 + 0.1**2 / 1000.0)
    assert low.item() == pytest.approx(expected, rel=0.01)


def test_substrate_inhibition_high_substrate():
    prop = substrate_inhibition(v_max=10.0, k_m=5.0, k_i=1.0, species_index=0)
    low = prop(torch.tensor([1.0]), 0.0)
    high = prop(torch.tensor([1000.0]), 0.0)
    assert high.item() < low.item()


def test_substrate_inhibition_dependencies():
    prop = substrate_inhibition(v_max=1.0, k_m=1.0, k_i=1.0, species_index=2)
    assert prop.species_dependencies == {2}


# ── SerializablePropensity protocol ──────────────────────────────────────────


def test_serializable_propensity_protocol():
    props = [
        mass_action(1.0, torch.tensor([1.0])),
        constant_rate(1.0),
        hill(1.0, 1.0, 2.0, 0),
        hill_repression(1.0, 1.0, 2.0, 0),
        enzyme_michaelis_menten(1.0, 1.0, 0, 1),
        substrate_inhibition(1.0, 1.0, 1.0, 0),
        hill_activation_repression(1.0, 1.0, 2.0, 0, 1.0, 2.0, 1),
    ]
    for prop in props:
        assert isinstance(prop, SerializablePropensity), (
            f"{type(prop).__name__} does not satisfy SerializablePropensity"
        )


# ── Params tensor round-trip ─────────────────────────────────────────────────


def test_mass_action_params_round_trip():
    original = MassActionParams(rate_constant=3.7)
    t = original.to_tensor()
    reconstructed = MassActionParams.from_tensor(t)
    assert reconstructed.rate_constant == pytest.approx(original.rate_constant)


def test_hill_params_round_trip():
    original = HillParams(v_max=8.0, k_m=3.0, hill_coefficient=2.5, species_index=1)
    t = original.to_tensor()
    reconstructed = HillParams.from_tensor(t)
    assert reconstructed.v_max == pytest.approx(original.v_max)
    assert reconstructed.k_m == pytest.approx(original.k_m)
    assert reconstructed.hill_coefficient == pytest.approx(original.hill_coefficient)
    assert reconstructed.species_index == original.species_index


def test_constant_rate_params_round_trip():
    original = ConstantRateParams(rate=5.2)
    t = original.to_tensor()
    reconstructed = ConstantRateParams.from_tensor(t)
    assert reconstructed.rate == pytest.approx(original.rate)


# ── Reindex ───────────────────────────────────────────────────────────────────


def test_mass_action_reindex():
    prop = mass_action(1.0, torch.tensor([1.0, 0.0, 1.0]))
    new_prop = prop.reindex_species({0: 1, 2: 3}, n_merged=5)
    rs = new_prop.reactant_stoichiometry
    assert rs.shape[0] == 5
    assert rs[1].item() == pytest.approx(1.0)
    assert rs[3].item() == pytest.approx(1.0)
    assert rs[0].item() == pytest.approx(0.0)
    assert rs[2].item() == pytest.approx(0.0)


def test_hill_reindex():
    prop = hill(v_max=5.0, k_m=2.0, hill_coefficient=2.0, species_index=1)
    new_prop = prop.reindex_species({1: 3}, n_merged=5)
    assert new_prop.params.species_index == 3
