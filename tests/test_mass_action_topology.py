"""Tests for MassActionTopology and named topology factories."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.data.generation.mass_action_topology import (
    MassActionTopology,
    auto_catalysis_topology,
    birth_death_topology,
    enzymatic_catalysis_topology,
    lotka_volterra_topology,
)
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA

# ── Construction validation ───────────────────────────────────────────────────


def test_topology_valid_construction():
    """Valid reactant/product matrices produce a topology."""
    t = MassActionTopology(
        reactant_matrix=torch.tensor([[0.0], [1.0]]),
        product_matrix=torch.tensor([[1.0], [0.0]]),
    )
    assert t.n_species == 1
    assert t.n_reactions == 2


def test_topology_rejects_noop_reaction():
    """Reaction with zero net change raises ValueError."""
    with pytest.raises(ValueError, match="no-ops"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[1.0], [1.0]]),
            product_matrix=torch.tensor([[1.0], [0.0]]),
        )


def test_topology_rejects_inactive_species():
    """Species not participating in any reaction raises ValueError."""
    with pytest.raises(ValueError, match="do not participate"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            product_matrix=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        )


def test_topology_rejects_duplicate_reactions():
    """Duplicate (reactant, product) pairs raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[0.0], [0.0], [1.0]]),
            product_matrix=torch.tensor([[1.0], [1.0], [0.0]]),
        )


def test_topology_rejects_negative_reactant_entries():
    """Negative entries in reactant_matrix raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[-1.0]]),
            product_matrix=torch.tensor([[0.0]]),
        )


def test_topology_rejects_negative_product_entries():
    """Negative entries in product_matrix raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[1.0]]),
            product_matrix=torch.tensor([[-1.0]]),
        )


def test_topology_shape_mismatch():
    """Mismatched reactant/product shapes raise ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[0.0, 0.0]]),
            product_matrix=torch.tensor([[1.0]]),
        )


def test_topology_default_names():
    """Species and reaction names default to S0, S1, ... and R0, R1, ..."""
    t = MassActionTopology(
        reactant_matrix=torch.tensor([[0.0], [1.0]]),
        product_matrix=torch.tensor([[1.0], [0.0]]),
    )
    assert t.species_names == ("S0",)
    assert t.reaction_names == ("R0", "R1")


def test_topology_custom_names_length_mismatch():
    """Wrong length for species_names raises ValueError."""
    with pytest.raises(ValueError, match="species_names length"):
        MassActionTopology(
            reactant_matrix=torch.tensor([[0.0], [1.0]]),
            product_matrix=torch.tensor([[1.0], [0.0]]),
            species_names=("A", "B"),
        )


# ── Named topology factories ──────────────────────────────────────────────────


def test_birth_death_topology():
    """birth_death_topology has 1 species, 2 reactions, production, and degradation."""
    t = birth_death_topology()
    assert t.n_species == 1
    assert t.n_reactions == 2
    assert t.has_production()
    assert t.has_degradation_for_all()


def test_auto_catalysis_topology():
    """auto_catalysis_topology has 1 species and 3 reactions."""
    t = auto_catalysis_topology()
    assert t.n_species == 1
    assert t.n_reactions == 3


def test_lotka_volterra_topology():
    """lotka_volterra_topology has 2 species and 3 reactions."""
    t = lotka_volterra_topology()
    assert t.n_species == 2
    assert t.n_reactions == 3


def test_enzymatic_catalysis_topology():
    """enzymatic_catalysis_topology has 4 species and 5 reactions."""
    t = enzymatic_catalysis_topology()
    assert t.n_species == 4
    assert t.n_reactions == 5


# ── to_crn ────────────────────────────────────────────────────────────────────


def test_to_crn_correct_dimensions():
    """CRN from to_crn() has the expected n_species and n_reactions."""
    t = birth_death_topology()
    crn = t.to_crn([2.0, 0.5])
    assert crn.n_reactions == 2
    assert crn.n_species == 1


def test_to_crn_wrong_n_rates():
    """Passing wrong number of rate constants raises ValueError."""
    t = birth_death_topology()
    with pytest.raises(ValueError, match="rate constants"):
        t.to_crn([1.0])


def test_to_crn_ssa_runs():
    """CRN from to_crn() simulates without error."""
    t = birth_death_topology()
    crn = t.to_crn([2.0, 0.5])
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0]),
        t_max=10.0,
    )
    assert traj.n_steps > 1


def test_to_crn_tensor_repr_roundtrip():
    """CRN from to_crn() converts to CRNTensorRepr without error."""
    t = lotka_volterra_topology()
    crn = t.to_crn([1.0, 0.01, 0.5])
    repr_ = crn_to_tensor_repr(crn)
    assert repr_.n_species == 2
    assert repr_.n_reactions == 3


def test_to_crn_multi_species():
    """to_crn works for a multi-species topology with enzyme kinetics."""
    t = enzymatic_catalysis_topology()
    crn = t.to_crn([1.0, 0.5, 0.2, 2.0, 0.1])
    assert crn.n_species == 4
    assert crn.n_reactions == 5


# ── reaction_orders ───────────────────────────────────────────────────────────


def test_reaction_orders_birth_death():
    """birth: order 0, death: order 1."""
    t = birth_death_topology()
    orders = t.reaction_orders()
    assert orders[0].item() == 0.0
    assert orders[1].item() == 1.0


def test_reaction_orders_lotka_volterra():
    """prey_birth: order 1, predation: order 2, predator_death: order 1."""
    t = lotka_volterra_topology()
    orders = t.reaction_orders()
    assert orders[0].item() == 1.0
    assert orders[1].item() == 2.0
    assert orders[2].item() == 1.0


# ── has_production / has_degradation ─────────────────────────────────────────


def test_has_production_true():
    """Topology with a zero-order reaction has production."""
    t = birth_death_topology()
    assert t.has_production()


def test_has_production_false():
    """Topology with only first-order reactions has no production."""
    # A -> 2A, A -> 0 (no zero-order)
    t = MassActionTopology(
        reactant_matrix=torch.tensor([[1.0], [1.0]]),
        product_matrix=torch.tensor([[2.0], [0.0]]),
    )
    assert not t.has_production()


def test_has_degradation_for_all_true():
    """birth_death has degradation for all species."""
    t = birth_death_topology()
    assert t.has_degradation_for_all()


def test_has_degradation_for_all_false():
    """Topology without degradation for all species returns False."""
    # 0->A, 0->B: no degradation for any species
    t = MassActionTopology(
        reactant_matrix=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        product_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    assert not t.has_degradation_for_all()


# ── summary ───────────────────────────────────────────────────────────────────


def test_summary_readable():
    """summary() contains species count, reaction count, and arrow notation."""
    t = birth_death_topology()
    s = t.summary()
    assert "1 species" in s
    assert "2 reactions" in s
    assert "->" in s


def test_summary_contains_species_names():
    """summary() uses the configured species names."""
    t = birth_death_topology()
    s = t.summary()
    assert "A" in s


# ── net_stoichiometry ─────────────────────────────────────────────────────────


def test_net_stoichiometry_birth_death():
    """Net stoichiometry is product - reactant."""
    t = birth_death_topology()
    net = t.net_stoichiometry
    assert net[0, 0].item() == 1.0  # birth: +1
    assert net[1, 0].item() == -1.0  # death: -1
