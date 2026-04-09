"""Tests for MassActionTopology and named topology factories."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.propensities import ConstantRateParams, MassActionParams
from crn_surrogate.data.generation.mass_action_topology import (
    MassActionTopology,
    birth_death_topology,
    enzymatic_catalysis_topology,
    lotka_volterra_topology,
)

# ── Construction validation ───────────────────────────────────────────────────


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        MassActionTopology(
            reactant_matrix=torch.zeros(2, 3),
            product_matrix=torch.zeros(2, 4),
        )


def test_rejects_negative_entries() -> None:
    with pytest.raises(ValueError):
        MassActionTopology(
            reactant_matrix=torch.tensor([[-1.0], [1.0]]),
            product_matrix=torch.tensor([[1.0], [0.0]]),
        )


def test_rejects_noop_reaction() -> None:
    # Row 0: reactant == product → zero net stoichiometry
    with pytest.raises(ValueError):
        MassActionTopology(
            reactant_matrix=torch.tensor([[1.0], [1.0]]),
            product_matrix=torch.tensor([[1.0], [0.0]]),
        )


def test_rejects_inactive_species() -> None:
    # Species 1 has no net stoichiometry in any reaction
    with pytest.raises(ValueError):
        MassActionTopology(
            reactant_matrix=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            product_matrix=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        )


def test_rejects_duplicate_reaction() -> None:
    with pytest.raises(ValueError):
        MassActionTopology(
            reactant_matrix=torch.tensor([[1.0], [1.0]]),
            product_matrix=torch.tensor([[0.0], [0.0]]),
        )


# ── Properties ────────────────────────────────────────────────────────────────


def test_net_stoichiometry() -> None:
    topo = birth_death_topology()
    net = topo.net_stoichiometry
    assert net.shape == (2, 1)
    assert net[0, 0].item() == pytest.approx(1.0)  # birth: +1
    assert net[1, 0].item() == pytest.approx(-1.0)  # death: -1


def test_n_species_n_reactions() -> None:
    topo = birth_death_topology()
    assert topo.n_species == 1
    assert topo.n_reactions == 2


def test_reaction_orders() -> None:
    topo = birth_death_topology()
    orders = topo.reaction_orders()
    assert orders[0].item() == pytest.approx(0.0)  # zero-order birth
    assert orders[1].item() == pytest.approx(1.0)  # first-order death


# ── to_crn ────────────────────────────────────────────────────────────────────


def test_to_crn_creates_valid_crn() -> None:
    crn = birth_death_topology().to_crn([2.0, 0.5])
    assert crn.n_species == 1
    assert crn.n_reactions == 2


def test_to_crn_propensity_types() -> None:
    crn = birth_death_topology().to_crn([2.0, 0.5])
    assert isinstance(crn.reactions[0].propensity.params, ConstantRateParams)
    assert isinstance(crn.reactions[1].propensity.params, MassActionParams)


def test_to_crn_wrong_rate_count() -> None:
    with pytest.raises(ValueError):
        birth_death_topology().to_crn([1.0, 2.0, 3.0])


# ── Named topologies ──────────────────────────────────────────────────────────


def test_birth_death_topology() -> None:
    topo = birth_death_topology()
    assert topo.n_species == 1
    assert topo.n_reactions == 2
    assert topo.species_names == ("A",)
    assert topo.reaction_names == ("birth", "death")


def test_lotka_volterra_topology() -> None:
    topo = lotka_volterra_topology()
    assert topo.n_species == 2
    assert topo.n_reactions == 3
    assert "prey" in topo.species_names
    assert "predator" in topo.species_names


def test_enzymatic_catalysis_topology() -> None:
    topo = enzymatic_catalysis_topology()
    assert topo.n_species == 4
    assert topo.n_reactions == 5
