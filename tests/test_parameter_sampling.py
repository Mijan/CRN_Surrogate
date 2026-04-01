"""Tests for kinetic parameter sampling: ranges, constraints, and reproducibility."""

from __future__ import annotations

import dataclasses
import math

import pytest

from crn_surrogate.data.generation.configs import SamplingConfig
from crn_surrogate.data.generation.motifs.auto_catalysis import (
    AutoCatalysisFactory,
    AutoCatalysisParams,
)
from crn_surrogate.data.generation.motifs.base import extract_parameter_ranges
from crn_surrogate.data.generation.motifs.birth_death import (
    BirthDeathFactory,
    BirthDeathParams,
)
from crn_surrogate.data.generation.motifs.repressilator import RepressilatorFactory
from crn_surrogate.data.generation.motifs.toggle_switch import ToggleSwitchFactory
from crn_surrogate.data.generation.parameter_sampling import ParameterSampler


@pytest.fixture
def sampler() -> ParameterSampler:
    """ParameterSampler with fixed seed 0."""
    return ParameterSampler(SamplingConfig(random_seed=0))


# --- Returned type ----------------------------------------------------


def test_sample_returns_typed_params(sampler: ParameterSampler) -> None:
    """sample() returns instances of the factory's params_type."""
    factory = BirthDeathFactory()
    param_list = sampler.sample(factory, n_samples=10)
    assert len(param_list) == 10
    for params in param_list:
        assert isinstance(params, BirthDeathParams)


def test_sample_initial_states_returns_dicts(sampler: ParameterSampler) -> None:
    """sample_initial_states() returns a list of dicts with int values."""
    factory = BirthDeathFactory()
    states = sampler.sample_initial_states(factory, n_samples=10)
    assert len(states) == 10
    for state in states:
        assert isinstance(state, dict)
        assert set(state.keys()) == set(factory.species_names)
        for v in state.values():
            assert isinstance(v, int)


# --- Rate parameter bounds --------------------------------------------


def test_rate_params_within_bounds(sampler: ParameterSampler) -> None:
    """Sampled rate params fall within [lo, hi] for BirthDeath."""
    factory = BirthDeathFactory()
    param_list = sampler.sample(factory, n_samples=200)
    ranges = extract_parameter_ranges(factory.params_type)
    for params in param_list:
        params_dict = dataclasses.asdict(params)
        for name, r in ranges.items():
            assert r.low <= params_dict[name] <= r.high, (
                f"{name}={params_dict[name]} not in [{r.low}, {r.high}]"
            )


def test_rate_params_within_bounds_repressilator(sampler: ParameterSampler) -> None:
    """Sampled rate params fall within bounds for Repressilator."""
    factory = RepressilatorFactory()
    param_list = sampler.sample(factory, n_samples=50)
    ranges = extract_parameter_ranges(factory.params_type)
    for params in param_list:
        params_dict = dataclasses.asdict(params)
        for name, r in ranges.items():
            assert r.low <= params_dict[name] <= r.high


# --- Hill coefficient bounds ------------------------------------------


def test_hill_coefficients_within_bounds(sampler: ParameterSampler) -> None:
    """Sampled Hill coefficients (log_uniform=False) fall within [lo, hi] for ToggleSwitch."""
    factory = ToggleSwitchFactory()
    param_list = sampler.sample(factory, n_samples=100)
    ranges = extract_parameter_ranges(factory.params_type)
    hill_ranges = {name: r for name, r in ranges.items() if not r.log_uniform}
    assert len(hill_ranges) > 0, "ToggleSwitch must have non-log-uniform parameters"
    for params in param_list:
        params_dict = dataclasses.asdict(params)
        for name, r in hill_ranges.items():
            assert r.low <= params_dict[name] <= r.high, (
                f"Hill coeff {name}={params_dict[name]} not in [{r.low}, {r.high}]"
            )


# --- AutoCatalysis constraint -----------------------------------------


def test_autocatalysis_constraint_always_satisfied(sampler: ParameterSampler) -> None:
    """All sampled AutoCatalysis parameter sets satisfy k_deg > k_cat."""
    factory = AutoCatalysisFactory()
    param_list = sampler.sample(factory, n_samples=100)
    for params in param_list:
        assert isinstance(params, AutoCatalysisParams)
        assert params.k_deg > params.k_cat, (
            f"Constraint violated: k_deg={params.k_deg}, k_cat={params.k_cat}"
        )


def test_validate_params_used_for_rejection_sampling() -> None:
    """ParameterSampler uses validate_params for rejection, not a separate check."""
    factory = AutoCatalysisFactory()
    sampler = ParameterSampler(SamplingConfig(random_seed=0))
    param_list = sampler.sample(factory, n_samples=50)
    # If validate_params is called, all results must satisfy the constraint
    for params in param_list:
        try:
            factory.validate_params(params)
        except ValueError as e:
            pytest.fail(f"validate_params raised for a sampled params: {e}")


# --- Reproducibility --------------------------------------------------


def test_sampling_reproducible_with_fixed_seed() -> None:
    """Two samplers with the same seed produce identical parameter lists."""
    factory = BirthDeathFactory()
    sampler_a = ParameterSampler(SamplingConfig(random_seed=42))
    sampler_b = ParameterSampler(SamplingConfig(random_seed=42))
    list_a = sampler_a.sample(factory, n_samples=50)
    list_b = sampler_b.sample(factory, n_samples=50)
    for pa, pb in zip(list_a, list_b):
        assert pa.k_prod == pytest.approx(pb.k_prod)
        assert pa.k_deg == pytest.approx(pb.k_deg)


def test_different_seeds_produce_different_samples() -> None:
    """Two samplers with different seeds produce different parameter lists."""
    factory = BirthDeathFactory()
    sampler_a = ParameterSampler(SamplingConfig(random_seed=1))
    sampler_b = ParameterSampler(SamplingConfig(random_seed=2))

    list_a = sampler_a.sample(factory, n_samples=10)
    list_b = sampler_b.sample(factory, n_samples=10)
    differs = any(abs(pa.k_prod - pb.k_prod) > 1e-9 for pa, pb in zip(list_a, list_b))
    assert differs, "Expected different samples for different seeds"


# --- Log-uniform distribution check ----------------------------------


def test_log_uniform_roughly_symmetric(sampler: ParameterSampler) -> None:
    """Log-uniform samples: roughly half fall below and above geometric mean."""
    factory = BirthDeathFactory()
    param_list = sampler.sample(factory, n_samples=500)
    r = extract_parameter_ranges(factory.params_type)["k_prod"]
    geo_mean = math.exp((math.log(r.low) + math.log(r.high)) / 2.0)
    below = sum(1 for p in param_list if p.k_prod < geo_mean)
    above = sum(1 for p in param_list if p.k_prod >= geo_mean)
    ratio = below / (below + above)
    assert 0.35 <= ratio <= 0.65, f"Log-uniform split {ratio:.2f} far from 0.5"


# --- Initial state bounds ---------------------------------------------


def test_initial_states_within_bounds(sampler: ParameterSampler) -> None:
    """Sampled initial states fall within declared ranges."""
    factory = BirthDeathFactory()
    states = sampler.sample_initial_states(factory, n_samples=100)
    ranges = factory.initial_state_ranges()
    for state in states:
        for name, r in ranges.items():
            assert r.low <= state[name] <= r.high, (
                f"{name}={state[name]} not in [{r.low}, {r.high}]"
            )
