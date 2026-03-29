"""Tests for propensity functions and the make_propensity factory.

Each test verifies a specific mathematical or structural property:
- Mass-action: a(X) = k * prod X_s^R_s
- Hill: a(X) = V_max * X^n / (K_m^n + X^n)
- Factory: returns the correct implementation for each PropensityType
"""

import pytest
import torch

from crn_surrogate.data.propensities import (
    HillPropensity,
    MassActionPropensity,
    PropensityType,
    make_propensity,
)


# ── MassActionPropensity ──────────────────────────────────────────────────────


def test_mass_action_zeroth_order_rate_equals_k():
    """Zeroth-order (creation) reaction: no reactants, propensity = k regardless of state."""
    reactant_stoich = torch.tensor([0.0])
    propensity = MassActionPropensity(reactant_stoich)
    k = 3.0
    params = torch.tensor([k, 0.0])

    rate = propensity.evaluate(torch.tensor([999.0]), params)

    assert rate.item() == pytest.approx(k)


def test_mass_action_first_order_rate_proportional_to_state():
    """First-order (degradation) reaction: propensity scales linearly with molecule count."""
    reactant_stoich = torch.tensor([1.0])
    propensity = MassActionPropensity(reactant_stoich)
    k = 2.0
    params = torch.tensor([k, 0.0])

    rate_5 = propensity.evaluate(torch.tensor([5.0]), params).item()
    rate_10 = propensity.evaluate(torch.tensor([10.0]), params).item()

    assert rate_5 == pytest.approx(k * 5.0)
    assert rate_10 == pytest.approx(k * 10.0)
    assert rate_10 == pytest.approx(2.0 * rate_5)


def test_mass_action_second_order_rate_proportional_to_state_squared():
    """Second-order reaction: propensity scales as X^2."""
    reactant_stoich = torch.tensor([2.0])
    propensity = MassActionPropensity(reactant_stoich)
    k = 1.0
    params = torch.tensor([k, 0.0])

    rate_3 = propensity.evaluate(torch.tensor([3.0]), params).item()
    rate_6 = propensity.evaluate(torch.tensor([6.0]), params).item()

    assert rate_3 == pytest.approx(9.0)
    assert rate_6 == pytest.approx(36.0)


def test_mass_action_zero_state_gives_zero_for_positive_order():
    """With zero molecules, any first-or-higher-order reaction has propensity zero."""
    reactant_stoich = torch.tensor([1.0])
    propensity = MassActionPropensity(reactant_stoich)
    params = torch.tensor([5.0, 0.0])

    rate = propensity.evaluate(torch.tensor([0.0]), params)

    assert rate.item() == pytest.approx(0.0)


def test_mass_action_bimolecular_rate_product_of_species():
    """Bimolecular reaction with two species: a = k * X_0 * X_1."""
    reactant_stoich = torch.tensor([1.0, 1.0])
    propensity = MassActionPropensity(reactant_stoich)
    k = 0.5
    params = torch.tensor([k, 0.0])

    rate = propensity.evaluate(torch.tensor([4.0, 3.0]), params)

    assert rate.item() == pytest.approx(k * 4.0 * 3.0)


# ── HillPropensity ────────────────────────────────────────────────────────────


def test_hill_propensity_saturates_at_vmax_for_large_x():
    """At very high substrate concentration, Hill rate approaches V_max."""
    propensity = HillPropensity()
    v_max, k_m, n, species_idx = 10.0, 1.0, 2.0, 0
    params = torch.tensor([v_max, k_m, n, float(species_idx)])

    rate = propensity.evaluate(torch.tensor([1000.0]), params)

    assert rate.item() == pytest.approx(v_max, abs=0.1)


def test_hill_propensity_half_max_rate_at_km():
    """At X = K_m, Hill rate equals V_max / 2 (definition of K_m)."""
    propensity = HillPropensity()
    v_max, k_m, n, species_idx = 8.0, 4.0, 1.0, 0
    params = torch.tensor([v_max, k_m, n, float(species_idx)])

    rate = propensity.evaluate(torch.tensor([k_m]), params)

    assert rate.item() == pytest.approx(v_max / 2.0, rel=1e-3)


def test_hill_propensity_is_monotone_increasing_in_x():
    """Higher substrate concentration produces higher Hill propensity."""
    propensity = HillPropensity()
    params = torch.tensor([5.0, 2.0, 2.0, 0.0])

    rates = [
        propensity.evaluate(torch.tensor([x]), params).item()
        for x in [1.0, 2.0, 5.0, 10.0]
    ]

    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_hill_propensity_uses_correct_species_index():
    """Hill propensity reads from the species specified by params[3], not species 0."""
    propensity = HillPropensity()
    v_max, k_m, n = 10.0, 1.0, 1.0
    # species 1 drives the reaction; species 0 is irrelevant
    params_s1 = torch.tensor([v_max, k_m, n, 1.0])

    state_low_s1 = torch.tensor([999.0, 0.5])
    state_high_s1 = torch.tensor([0.0, 50.0])

    rate_low = propensity.evaluate(state_low_s1, params_s1).item()
    rate_high = propensity.evaluate(state_high_s1, params_s1).item()

    assert rate_low < rate_high


# ── make_propensity factory ───────────────────────────────────────────────────


def test_make_propensity_mass_action_returns_mass_action_instance():
    """Factory with MASS_ACTION type must return a MassActionPropensity."""
    reactant_stoich = torch.tensor([1.0])
    result = make_propensity(PropensityType.MASS_ACTION, reactant_stoich)
    assert isinstance(result, MassActionPropensity)


def test_make_propensity_hill_returns_hill_instance():
    """Factory with HILL type must return a HillPropensity."""
    reactant_stoich = torch.tensor([1.0])
    result = make_propensity(PropensityType.HILL, reactant_stoich)
    assert isinstance(result, HillPropensity)
