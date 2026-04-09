"""Tests for MassActionCRNGenerator."""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.mass_action_generator import (
    MassActionCRNGenerator,
    MassActionGeneratorConfig,
    RandomTopologyConfig,
)


def _make_gen(
    n_species_range: tuple[int, int] = (1, 3),
    n_reactions_range: tuple[int, int] = (2, 6),
) -> MassActionCRNGenerator:
    cfg = MassActionGeneratorConfig(
        topology=RandomTopologyConfig(
            n_species_range=n_species_range,
            n_reactions_range=n_reactions_range,
        )
    )
    return MassActionCRNGenerator(cfg)


def test_sample_returns_crn() -> None:
    gen = _make_gen()
    crn = gen.sample()
    assert isinstance(crn, CRN)


def test_sample_respects_species_range() -> None:
    gen = _make_gen(n_species_range=(1, 3))
    torch.manual_seed(0)
    for _ in range(10):
        crn = gen.sample()
        assert 1 <= crn.n_species <= 3


def test_sample_respects_reactions_range() -> None:
    gen = _make_gen(n_reactions_range=(2, 6))
    torch.manual_seed(1)
    for _ in range(10):
        crn = gen.sample()
        assert 2 <= crn.n_reactions <= 10  # repair steps may add extra reactions


def test_sample_initial_state_shape() -> None:
    gen = _make_gen()
    crn = gen.sample()
    state = gen.sample_initial_state(crn)
    assert state.shape == (crn.n_species,)


def test_sample_initial_state_positive() -> None:
    gen = _make_gen()
    torch.manual_seed(2)
    for _ in range(10):
        crn = gen.sample()
        state = gen.sample_initial_state(crn)
        assert (state >= 0).all()


def test_sample_diversity() -> None:
    gen = _make_gen()
    torch.manual_seed(42)
    pairs = {(crn.n_species, crn.n_reactions) for crn in gen.sample_batch(20)}
    assert len(pairs) > 1


def test_sample_propensities_evaluate() -> None:
    gen = _make_gen()
    torch.manual_seed(3)
    crn = gen.sample()
    state = gen.sample_initial_state(crn)
    props = crn.evaluate_propensities(state, 0.0)
    assert props.isfinite().all()
    assert (props >= 0).all()
