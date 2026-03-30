"""Tests for propensity factory functions and parameter dataclasses.

Covers:
- mass_action: zeroth, first, second order; bimolecular; zero state.
- hill: saturation at V_max; half-max at K_m; monotone; species index selection.
- constant_rate: always returns k.
- MassActionParams / HillParams round-trip via to_tensor / from_tensor.
"""

import pytest
import torch

from crn_surrogate.crn.propensities import (
    ConstantRateParams,
    EnzymeMichaelisMentenParams,
    HillParams,
    MassActionParams,
    constant_rate,
    enzyme_michaelis_menten,
    hill,
    mass_action,
)

# ── mass_action ───────────────────────────────────────────────────────────────


def test_mass_action_zeroth_order_rate_equals_k():
    """Zeroth-order (creation) reaction: propensity = k regardless of state."""
    prop = mass_action(rate_constant=3.0, reactant_stoichiometry=torch.tensor([0.0]))
    rate = prop(torch.tensor([999.0]), 0.0)
    assert rate.item() == pytest.approx(3.0)


def test_mass_action_first_order_rate_proportional_to_state():
    """First-order reaction: propensity scales linearly with molecule count."""
    k = 2.0
    prop = mass_action(rate_constant=k, reactant_stoichiometry=torch.tensor([1.0]))
    assert prop(torch.tensor([5.0]), 0.0).item() == pytest.approx(k * 5.0)
    assert prop(torch.tensor([10.0]), 0.0).item() == pytest.approx(k * 10.0)


def test_mass_action_second_order_rate_proportional_to_state_squared():
    """Second-order single-species reaction: propensity scales as X^2."""
    prop = mass_action(rate_constant=1.0, reactant_stoichiometry=torch.tensor([2.0]))
    assert prop(torch.tensor([3.0]), 0.0).item() == pytest.approx(9.0)
    assert prop(torch.tensor([6.0]), 0.0).item() == pytest.approx(36.0)


def test_mass_action_zero_state_gives_zero_for_positive_order():
    """With zero molecules, any first-or-higher-order reaction has zero propensity."""
    prop = mass_action(rate_constant=5.0, reactant_stoichiometry=torch.tensor([1.0]))
    assert prop(torch.tensor([0.0]), 0.0).item() == pytest.approx(0.0)


def test_mass_action_bimolecular_is_product_of_species():
    """Bimolecular reaction: a = k * X_0 * X_1."""
    k = 0.5
    prop = mass_action(rate_constant=k, reactant_stoichiometry=torch.tensor([1.0, 1.0]))
    assert prop(torch.tensor([4.0, 3.0]), 0.0).item() == pytest.approx(k * 4.0 * 3.0)


def test_mass_action_ignores_t():
    """Mass-action propensity is autonomous: same value for different t."""
    prop = mass_action(rate_constant=1.0, reactant_stoichiometry=torch.tensor([1.0]))
    state = torch.tensor([5.0])
    assert prop(state, 0.0).item() == pytest.approx(prop(state, 99.0).item())


def test_mass_action_closure_has_params_property():
    """The returned closure exposes .params for inspection."""
    prop = mass_action(rate_constant=2.5, reactant_stoichiometry=torch.tensor([1.0]))
    assert hasattr(prop, "params")
    assert prop.params.rate_constant == pytest.approx(2.5)


# ── hill ──────────────────────────────────────────────────────────────────────


def test_hill_propensity_saturates_at_vmax_for_large_x():
    """At very high substrate concentration, Hill rate approaches V_max."""
    prop = hill(v_max=10.0, k_m=1.0, hill_coefficient=2.0, species_index=0)
    rate = prop(torch.tensor([1000.0]), 0.0)
    assert rate.item() == pytest.approx(10.0, abs=0.1)


def test_hill_propensity_half_max_rate_at_km():
    """At X = K_m, Hill rate equals V_max / 2 (definition of K_m for n=1)."""
    v_max, k_m = 8.0, 4.0
    prop = hill(v_max=v_max, k_m=k_m, hill_coefficient=1.0, species_index=0)
    rate = prop(torch.tensor([k_m]), 0.0)
    assert rate.item() == pytest.approx(v_max / 2.0, rel=1e-3)


def test_hill_propensity_is_monotone_increasing_in_x():
    """Higher substrate concentration produces higher Hill propensity."""
    prop = hill(v_max=5.0, k_m=2.0, hill_coefficient=2.0, species_index=0)
    rates = [prop(torch.tensor([x]), 0.0).item() for x in [1.0, 2.0, 5.0, 10.0]]
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_hill_propensity_uses_correct_species_index():
    """Hill propensity reads from the species at species_index, not species 0."""
    prop = hill(v_max=10.0, k_m=1.0, hill_coefficient=1.0, species_index=1)
    rate_low = prop(torch.tensor([999.0, 0.5]), 0.0).item()
    rate_high = prop(torch.tensor([0.0, 50.0]), 0.0).item()
    assert rate_low < rate_high


def test_hill_closure_has_params_property():
    """The returned Hill closure exposes .params for inspection."""
    prop = hill(v_max=3.0, k_m=2.0, hill_coefficient=1.5, species_index=0)
    assert hasattr(prop, "params")
    assert prop.params.v_max == pytest.approx(3.0)
    assert prop.params.k_m == pytest.approx(2.0)


# ── constant_rate ──────────────────────────────────────────────────────────────


def test_constant_rate_returns_k_regardless_of_state():
    """Constant propensity returns the same value for any state."""
    prop = constant_rate(k=7.0)
    assert prop(torch.tensor([0.0]), 0.0).item() == pytest.approx(7.0)
    assert prop(torch.tensor([100.0, 50.0]), 0.0).item() == pytest.approx(7.0)


def test_constant_rate_ignores_t():
    """Constant propensity is autonomous."""
    prop = constant_rate(k=3.0)
    state = torch.tensor([10.0])
    assert prop(state, 0.0).item() == pytest.approx(prop(state, 99.0).item())


# ── Parameter serialization ───────────────────────────────────────────────────


def test_mass_action_params_round_trip():
    """MassActionParams.to_tensor / from_tensor preserve the rate constant."""
    params = MassActionParams(rate_constant=1.5)
    t = params.to_tensor(max_params=4)
    assert t[0].item() == pytest.approx(1.5)
    reconstructed = MassActionParams.from_tensor(t)
    assert reconstructed.rate_constant == pytest.approx(1.5)


def test_hill_params_round_trip():
    """HillParams.to_tensor / from_tensor preserve all four parameters."""
    params = HillParams(v_max=5.0, k_m=2.0, hill_coefficient=3.0, species_index=1)
    t = params.to_tensor(max_params=4)
    reconstructed = HillParams.from_tensor(t)
    assert reconstructed.v_max == pytest.approx(5.0)
    assert reconstructed.k_m == pytest.approx(2.0)
    assert reconstructed.hill_coefficient == pytest.approx(3.0)
    assert reconstructed.species_index == 1


# ── constant_rate ──────────────────────────────────────────────────────────────


def test_constant_rate_has_params_property():
    prop = constant_rate(k=5.0)
    assert hasattr(prop, "params")
    assert prop.params.rate == pytest.approx(5.0)


def test_constant_rate_has_empty_species_dependencies():
    prop = constant_rate(k=1.0)
    assert prop.species_dependencies == frozenset()


def test_constant_rate_params_round_trip():
    params = ConstantRateParams(rate=3.7)
    t = params.to_tensor(max_params=4)
    assert t[0].item() == pytest.approx(3.7)
    reconstructed = ConstantRateParams.from_tensor(t)
    assert reconstructed.rate == pytest.approx(3.7)


# ── enzyme_michaelis_menten ────────────────────────────────────────────────────


def test_enzyme_mm_propensity_rate_proportional_to_enzyme():
    prop = enzyme_michaelis_menten(k_cat=1.0, k_m=10.0, enzyme_index=0, substrate_index=1)
    state_low_enzyme  = torch.tensor([1.0, 50.0])
    state_high_enzyme = torch.tensor([10.0, 50.0])
    assert prop(state_low_enzyme, 0.0).item() < prop(state_high_enzyme, 0.0).item()


def test_enzyme_mm_propensity_saturates_at_high_substrate():
    prop = enzyme_michaelis_menten(k_cat=2.0, k_m=1.0, enzyme_index=0, substrate_index=1)
    state_low  = torch.tensor([5.0, 0.5])
    state_high = torch.tensor([5.0, 500.0])
    rate_low  = prop(state_low, 0.0).item()
    rate_high = prop(state_high, 0.0).item()
    # At high substrate the rate approaches k_cat * enzyme
    assert rate_high == pytest.approx(2.0 * 5.0, rel=0.01)
    assert rate_high > rate_low


def test_enzyme_mm_dependencies_contain_both_enzyme_and_substrate():
    prop = enzyme_michaelis_menten(k_cat=1.0, k_m=1.0, enzyme_index=2, substrate_index=4)
    assert prop.species_dependencies == frozenset({2, 4})


def test_enzyme_mm_params_round_trip():
    params = EnzymeMichaelisMentenParams(k_cat=0.5, k_m=8.0, enzyme_index=1, substrate_index=3)
    t = params.to_tensor(max_params=4)
    reconstructed = EnzymeMichaelisMentenParams.from_tensor(t)
    assert reconstructed.k_cat == pytest.approx(0.5)
    assert reconstructed.k_m == pytest.approx(8.0)
    assert reconstructed.enzyme_index == 1
    assert reconstructed.substrate_index == 3


# ── species_dependencies on existing closures ─────────────────────────────────


def test_mass_action_species_dependencies_nonzero_reactants():
    prop = mass_action(rate_constant=1.0, reactant_stoichiometry=torch.tensor([1.0, 0.0, 1.0]))
    assert prop.species_dependencies == frozenset({0, 2})


def test_mass_action_zero_order_has_empty_dependencies():
    prop = mass_action(rate_constant=2.0, reactant_stoichiometry=torch.tensor([0.0]))
    assert prop.species_dependencies == frozenset()


def test_hill_species_dependencies_is_singleton():
    prop = hill(v_max=1.0, k_m=1.0, hill_coefficient=2.0, species_index=3)
    assert prop.species_dependencies == frozenset({3})
