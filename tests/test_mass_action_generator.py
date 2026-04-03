"""Tests for RandomTopologySampler and MassActionCRNGenerator."""

from __future__ import annotations

import torch

from crn_surrogate.data.generation.mass_action_generator import (
    MassActionCRNGenerator,
    MassActionGeneratorConfig,
    RandomTopologyConfig,
    RandomTopologySampler,
)
from crn_surrogate.data.generation.mass_action_topology import birth_death_topology
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA

# ── RandomTopologySampler ─────────────────────────────────────────────────────


def test_sampler_returns_valid_topology():
    """sample() returns a topology with dimensions within the configured ranges."""
    torch.manual_seed(0)
    sampler = RandomTopologySampler(RandomTopologyConfig())
    t = sampler.sample()
    assert 1 <= t.n_species <= 3
    assert t.n_reactions >= 2


def test_sampler_production_when_required():
    """All sampled topologies have at least one zero-order reaction."""
    torch.manual_seed(1)
    sampler = RandomTopologySampler(RandomTopologyConfig(require_production=True))
    for _ in range(10):
        t = sampler.sample()
        assert t.has_production(), (
            "Topology missing production despite require_production=True"
        )


def test_sampler_degradation_when_required():
    """All sampled topologies have a degradation path for every species."""
    torch.manual_seed(2)
    sampler = RandomTopologySampler(RandomTopologyConfig(require_degradation=True))
    for _ in range(10):
        t = sampler.sample()
        assert t.has_degradation_for_all(), (
            "Topology missing degradation despite require_degradation=True"
        )


def test_sampler_batch_diverse():
    """sample_batch() produces topologies with varying n_species."""
    torch.manual_seed(3)
    sampler = RandomTopologySampler(RandomTopologyConfig(n_species_range=(1, 3)))
    topologies = sampler.sample_batch(30)
    species_counts = {t.n_species for t in topologies}
    assert len(species_counts) > 1, (
        f"Expected diverse n_species, got only {species_counts}"
    )


def test_sampler_topology_passes_validation():
    """Every sampled topology passes MassActionTopology's __post_init__ validation."""
    torch.manual_seed(4)
    sampler = RandomTopologySampler(RandomTopologyConfig())
    for _ in range(20):
        t = sampler.sample()
        # Reconstruction from matrices should succeed (no validation error)
        from crn_surrogate.data.generation.mass_action_topology import (
            MassActionTopology,
        )

        MassActionTopology(
            reactant_matrix=t.reactant_matrix,
            product_matrix=t.product_matrix,
        )


# ── MassActionCRNGenerator ────────────────────────────────────────────────────


def test_generator_sample_returns_crn():
    """sample() returns a CRN with valid dimensions."""
    torch.manual_seed(5)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    crn = gen.sample()
    assert crn.n_species >= 1
    assert crn.n_reactions >= 2


def test_generator_sample_from_topology():
    """sample_from_topology() uses the given topology's structure."""
    torch.manual_seed(6)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    t = birth_death_topology()
    crn = gen.sample_from_topology(t)
    assert crn.n_species == 1
    assert crn.n_reactions == 2


def test_generator_sample_from_topology_rate_varies():
    """sample_from_topology() assigns different rates each call."""
    torch.manual_seed(7)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    t = birth_death_topology()
    crn_a = gen.sample_from_topology(t)
    crn_b = gen.sample_from_topology(t)
    # Rates are drawn randomly; with high probability they differ
    props_a = crn_a.evaluate_propensities(torch.tensor([5.0]), 0.0)
    props_b = crn_b.evaluate_propensities(torch.tensor([5.0]), 0.0)
    assert not torch.allclose(props_a, props_b), (
        "Expected different rate constants across two independent samples"
    )


def test_generator_ssa_runs():
    """CRNs from sample() can be simulated with GillespieSSA."""
    torch.manual_seed(8)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    crn = gen.sample()
    init = gen.sample_initial_state(crn)
    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=init,
        t_max=10.0,
    )
    assert traj.n_steps > 1
    assert torch.isfinite(traj.states).all()


def test_generator_tensor_repr():
    """CRNs from sample() convert to CRNTensorRepr without error."""
    torch.manual_seed(9)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    crn = gen.sample()
    repr_ = crn_to_tensor_repr(crn)
    assert repr_.n_species == crn.n_species
    assert repr_.n_reactions == crn.n_reactions


def test_generator_sample_initial_state_shape():
    """sample_initial_state returns (n_species,) with non-negative values."""
    torch.manual_seed(10)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    for _ in range(10):
        crn = gen.sample()
        x0 = gen.sample_initial_state(crn)
        assert x0.shape == (crn.n_species,)
        assert (x0 >= 0).all()


def test_generator_batch():
    """sample_batch() returns the requested number of CRNs."""
    torch.manual_seed(11)
    gen = MassActionCRNGenerator(MassActionGeneratorConfig())
    crns = gen.sample_batch(10)
    assert len(crns) >= 5  # allow some failures
    for crn in crns:
        assert crn.n_species >= 1


# ── Duplicate-reaction regression tests ──────────────────────────────────────


def test_sampler_no_duplicate_reactions():
    """RandomTopologySampler produces topologies without duplicate reactions."""
    torch.manual_seed(0)
    sampler = RandomTopologySampler(RandomTopologyConfig())
    for _ in range(50):
        topo = sampler.sample()
        seen: set[tuple] = set()
        for r in range(topo.n_reactions):
            key = (
                tuple(topo.reactant_matrix[r].tolist()),
                tuple(topo.product_matrix[r].tolist()),
            )
            assert key not in seen, f"Duplicate reaction at index {r}: {key}"
            seen.add(key)


def test_no_duplicate_production_reactions():
    """Repair must not create duplicate zero-order production reactions."""
    torch.manual_seed(0)
    sampler = RandomTopologySampler(
        RandomTopologyConfig(
            n_species_range=(1, 1),
            n_reactions_range=(2, 4),
            require_production=True,
            require_degradation=True,
        )
    )
    for _ in range(100):
        topo = sampler.sample()
        seen: set[tuple] = set()
        for r in range(topo.n_reactions):
            key = (
                tuple(topo.reactant_matrix[r].tolist()),
                tuple(topo.product_matrix[r].tolist()),
            )
            assert key not in seen, f"Duplicate at index {r}: {key}"
            seen.add(key)
