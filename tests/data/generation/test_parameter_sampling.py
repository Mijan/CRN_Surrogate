"""Tests for ParameterSampler."""

from __future__ import annotations

from crn_surrogate.data.generation.configs import SamplingConfig
from crn_surrogate.data.generation.motifs.base import extract_parameter_ranges
from crn_surrogate.data.generation.motifs.birth_death import (
    BirthDeathFactory,
    BirthDeathParams,
)
from crn_surrogate.data.generation.parameter_sampling import ParameterSampler


def _make_sampler(seed: int = 42) -> ParameterSampler:
    return ParameterSampler(SamplingConfig(random_seed=seed))


def test_sample_returns_correct_count() -> None:
    sampler = _make_sampler()
    factory = BirthDeathFactory()
    params = sampler.sample(factory, 10)
    assert len(params) == 10


def test_sample_returns_correct_type() -> None:
    sampler = _make_sampler()
    factory = BirthDeathFactory()
    params = sampler.sample(factory, 5)
    for p in params:
        assert isinstance(p, BirthDeathParams)


def test_sample_values_in_range() -> None:
    sampler = _make_sampler()
    factory = BirthDeathFactory()
    ranges = extract_parameter_ranges(BirthDeathParams)
    params_list = sampler.sample(factory, 100)
    for params in params_list:
        for name, r in ranges.items():
            val = getattr(params, name)
            assert r.low <= val <= r.high, (
                f"{name}={val} out of range [{r.low}, {r.high}]"
            )


def test_sample_initial_states_positive() -> None:
    sampler = _make_sampler()
    factory = BirthDeathFactory()
    states = sampler.sample_initial_states(factory, 20)
    for state in states:
        for val in state.values():
            assert val >= 0


def test_reproducibility() -> None:
    factory = BirthDeathFactory()
    sampler1 = _make_sampler(seed=7)
    sampler2 = _make_sampler(seed=7)
    p1 = sampler1.sample(factory, 5)
    p2 = sampler2.sample(factory, 5)
    for a, b in zip(p1, p2):
        assert a == b
