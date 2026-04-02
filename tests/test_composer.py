"""Tests for CRNComposer: species merging, reaction reindexing, and error handling."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.propensities import HillRepressionParams
from crn_surrogate.data.generation.composer import CompositionSpec, CRNComposer
from crn_surrogate.data.generation.motifs.birth_death import (
    BirthDeathFactory,
    BirthDeathParams,
)
from crn_surrogate.data.generation.motifs.negative_autoregulation import (
    NegativeAutoregulationFactory,
    NegativeAutoregulationParams,
)


def _bd_params() -> BirthDeathParams:
    """Minimal valid BirthDeath params."""
    return BirthDeathParams(k_prod=5.0, k_deg=0.1)


def _nar_params() -> NegativeAutoregulationParams:
    """Minimal valid NegativeAutoregulation params."""
    return NegativeAutoregulationParams(k_max=10.0, k_half=20.0, k_deg=0.05, n_hill=2.0)


@pytest.fixture
def composer() -> CRNComposer:
    """CRNComposer instance."""
    return CRNComposer()


@pytest.fixture
def bd_factory() -> BirthDeathFactory:
    """BirthDeathFactory instance."""
    return BirthDeathFactory()


@pytest.fixture
def nar_factory() -> NegativeAutoregulationFactory:
    """NegativeAutoregulationFactory instance."""
    return NegativeAutoregulationFactory()


# --- Basic composition tests -------------------------------------------


def test_compose_two_independent_crns(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
) -> None:
    """Composing two BD CRNs with no coupling yields 2 species and 4 reactions."""
    crn_a = bd_factory.create(BirthDeathParams(k_prod=5.0, k_deg=0.1))
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import constant_rate, mass_action
    from crn_surrogate.crn.reaction import Reaction

    # Create a second BD CRN with species named "B"
    reactions_b = [
        Reaction(
            stoichiometry=torch.tensor([1.0]),
            propensity=constant_rate(3.0),
            name="birth_B",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0]),
            propensity=mass_action(0.2, torch.tensor([1.0])),
            name="death_B",
        ),
    ]
    crn_b = CRN(reactions=reactions_b, species_names=["B"])

    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=bd_factory,
        coupling_map={},
    )
    merged = composer.compose(crn_a, crn_b, spec)
    assert merged.n_species == 2
    assert merged.n_reactions == 4
    assert "A" in merged.species_names
    assert "B" in merged.species_names


def test_compose_with_coupling_reduces_species_count(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """Composing BD and NAR with A->A coupling merges into 1 species."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    # Both have species "A" — identify them
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "A"},
    )
    merged = composer.compose(crn_up, crn_down, spec)
    assert merged.n_species == 1
    assert merged.n_reactions == crn_up.n_reactions + crn_down.n_reactions


def test_compose_n_reactions_sum(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """Merged CRN reaction count equals sum of both input CRNs."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "A"},
    )
    merged = composer.compose(crn_up, crn_down, spec)
    assert merged.n_reactions == crn_up.n_reactions + crn_down.n_reactions


# --- Propensity re-indexing --------------------------------------------


def test_hill_repression_index_remapped(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """After composition, Hill repression in NAR references the merged species index 0."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "A"},
    )
    merged = composer.compose(crn_up, crn_down, spec)

    # Find the hill-repression reaction in the merged CRN
    hill_rxns = [
        rxn
        for rxn in merged.reactions
        if hasattr(rxn.propensity, "params")
        and isinstance(rxn.propensity.params, HillRepressionParams)
    ]
    assert len(hill_rxns) >= 1
    for rxn in hill_rxns:
        # After merging "A" -> index 0, the species_index should still be 0
        assert rxn.propensity.params.species_index == 0


# --- Stoichiometry expansion -------------------------------------------


def test_merged_stoichiometry_correct_shape(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """Merged stoichiometry has shape (n_up + n_down, n_merged_species)."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "A"},
    )
    merged = composer.compose(crn_up, crn_down, spec)
    expected_shape = (merged.n_reactions, merged.n_species)
    assert merged.stoichiometry_matrix.shape == torch.Size(expected_shape)


def test_upstream_stoichiometry_preserved(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """Upstream stoichiometry values are preserved in the merged CRN."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "A"},
    )
    merged = composer.compose(crn_up, crn_down, spec)
    # First two reactions come from upstream (birth/death)
    assert merged.stoichiometry_matrix[0, 0].item() == pytest.approx(1.0)
    assert merged.stoichiometry_matrix[1, 0].item() == pytest.approx(-1.0)


# --- Error handling ----------------------------------------------------


def test_invalid_upstream_species_raises(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """ValueError raised when coupling_map references unknown upstream species."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"NONEXISTENT": "A"},
    )
    with pytest.raises(ValueError, match="Upstream species"):
        composer.compose(crn_up, crn_down, spec)


def test_reindex_propensity_raises_for_missing_method(
    composer: CRNComposer,
) -> None:
    """_reindex_propensity raises TypeError when propensity lacks reindex_species."""

    class _NonReindexablePropensity:
        def __call__(self, state: object, t: float) -> object:
            return 0.0

    bad_propensity = _NonReindexablePropensity()
    with pytest.raises(TypeError, match="reindex_species"):
        composer._reindex_propensity(bad_propensity, index_map={0: 1}, n_merged=2)


def test_composed_propensities_reference_correct_merged_indices(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
) -> None:
    """After composition with remapping, downstream propensities use merged species indices."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import MassActionParams, constant_rate, mass_action
    from crn_surrogate.crn.reaction import Reaction

    # upstream: species "X" at index 0
    crn_up = bd_factory.create(BirthDeathParams(k_prod=5.0, k_deg=0.1))

    # downstream: species "Y" at index 0 in its own indexing; no coupling
    reactions_down = [
        Reaction(
            stoichiometry=torch.tensor([1.0]),
            propensity=constant_rate(2.0),
            name="Y_birth",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0]),
            propensity=mass_action(0.5, torch.tensor([1.0])),
            name="Y_death",
        ),
    ]
    crn_down = CRN(reactions=reactions_down, species_names=["Y"])

    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=bd_factory,
        coupling_map={},
    )
    merged = composer.compose(crn_up, crn_down, spec)

    # In merged CRN: X=0, Y=1
    # The downstream mass-action propensity originally referenced species 0 (Y in downstream).
    # After reindexing it should reference species 1 (Y in merged).
    downstream_death = merged.reactions[-1]
    assert isinstance(downstream_death.propensity.params, MassActionParams)
    # species_dependencies should include index 1 (Y in merged), not 0
    assert 1 in downstream_death.propensity.species_dependencies
    assert 0 not in downstream_death.propensity.species_dependencies


def test_invalid_downstream_species_raises(
    composer: CRNComposer,
    bd_factory: BirthDeathFactory,
    nar_factory: NegativeAutoregulationFactory,
) -> None:
    """ValueError raised when coupling_map references unknown downstream species."""
    crn_up = bd_factory.create(_bd_params())
    crn_down = nar_factory.create(_nar_params())
    spec = CompositionSpec(
        upstream_factory=bd_factory,
        downstream_factory=nar_factory,
        coupling_map={"A": "NONEXISTENT"},
    )
    with pytest.raises(ValueError, match="Downstream species"):
        composer.compose(crn_up, crn_down, spec)
