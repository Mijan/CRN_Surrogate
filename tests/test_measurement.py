"""Tests for the measurement model subpackage."""

from __future__ import annotations

import math

import pytest
import torch

from crn_surrogate.measurement.config import (
    MeasurementConfig,
    NoiseConfig,
    NoiseMode,
    NoiseSharing,
)
from crn_surrogate.measurement.direct import DirectObservation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_direct(
    init_eps: float = 0.02,
    mode: NoiseMode = NoiseMode.LEARNED,
    sharing: NoiseSharing = NoiseSharing.SHARED,
    n_species: int | None = None,
    min_variance: float = 1e-2,
) -> DirectObservation:
    cfg = NoiseConfig(mode=mode, sharing=sharing, init_value=init_eps)
    return DirectObservation(
        noise_config=cfg, min_variance=min_variance, n_species=n_species
    )


def _zero_process_var(x: torch.Tensor) -> torch.Tensor:
    """Zero process variance — isolates the pure observation noise in log_likelihood."""
    return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# DirectObservation tests
# ---------------------------------------------------------------------------


def test_direct_observation_predict_is_identity():
    """predict() returns input unchanged."""
    model = _make_direct()
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(model.predict(x), x)


def test_direct_observation_variance_scales_quadratically():
    """Observation variance is (eps * x)^2; ratio at x=100k vs x=100 is 1e6."""
    model = _make_direct(init_eps=0.1)
    eps = model.eps.item()

    var_low = (eps * 100.0) ** 2
    var_high = (eps * 100_000.0) ** 2

    assert pytest.approx(var_high / var_low, rel=1e-5) == 1e6


def test_direct_observation_eps_always_positive():
    """eps > 0 even when raw parameter is very negative."""
    model = _make_direct()
    with torch.no_grad():
        model._raw_eps.fill_(-100.0)
    assert model.eps.item() > 0.0


def test_direct_observation_init_eps_accuracy():
    """softplus(raw_init) matches init_value at construction."""
    for init_eps in [0.01, 0.02, 0.05, 0.1]:
        model = _make_direct(init_eps=init_eps)
        assert abs(model.eps.item() - init_eps) < 1e-5


def test_direct_observation_init_eps_validation():
    """init_value <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="init_value"):
        _make_direct(init_eps=0.0)
    with pytest.raises(ValueError, match="init_value"):
        _make_direct(init_eps=-0.01)


def test_direct_observation_min_variance_validation():
    """min_variance <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="min_variance"):
        _make_direct(min_variance=0.0)
    with pytest.raises(ValueError, match="min_variance"):
        _make_direct(min_variance=-1.0)


def test_direct_observation_eps_is_learnable():
    """Gradient flows through eps to raw_eps when mode=LEARNED."""
    model = _make_direct(mode=NoiseMode.LEARNED)
    x = torch.tensor([[50.0, 100.0]])
    y = torch.tensor([[51.0, 99.0]])
    log_lik = model.log_likelihood(y, x, _zero_process_var(x))
    loss = -log_lik.sum()
    loss.backward()
    assert model._raw_eps.grad is not None
    assert model._raw_eps.grad.abs().sum().item() > 0.0


def test_direct_observation_eps_is_fixed():
    """No gradient on eps when mode=FIXED."""
    model = _make_direct(mode=NoiseMode.FIXED)
    # With FIXED mode, _raw_eps is a buffer, not a Parameter
    assert not isinstance(model._raw_eps, torch.nn.Parameter)
    # Buffers do not have requires_grad by default
    assert not model._raw_eps.requires_grad


def test_direct_observation_per_species_noise():
    """PER_SPECIES creates a vector of length n_species."""
    model = _make_direct(sharing=NoiseSharing.PER_SPECIES, n_species=3)
    assert model.eps.shape == (3,)


def test_direct_observation_per_species_requires_n_species():
    """PER_SPECIES with n_species=None raises ValueError."""
    with pytest.raises(ValueError, match="n_species"):
        _make_direct(sharing=NoiseSharing.PER_SPECIES, n_species=None)


def test_direct_observation_sample_shape():
    """sample() returns same shape as input."""
    model = _make_direct()
    x = torch.ones(4, 3)
    assert model.sample(x).shape == x.shape


def test_direct_observation_sample_adds_noise():
    """sample() != predict() (stochastic)."""
    model = _make_direct(init_eps=0.1)
    x = torch.full((100, 1), 1000.0)
    samples = model.sample(x)
    # Very unlikely all samples equal the input with eps=0.1 and x=1000
    assert not torch.equal(samples, x)


def test_log_likelihood_correct_zero_process_variance():
    """Hand-computed Gaussian log-likelihood matches log_likelihood with zero process variance.

    x=100, y=102, eps=0.1 -> obs_var = (0.1*100)^2 = 100
    log p = -0.5 * ((102-100)^2 / 100 + log(100))
    """
    model = _make_direct(init_eps=0.1)
    # Pin eps to exactly 0.1
    with torch.no_grad():
        model._raw_eps.fill_(math.log(math.expm1(0.1)))

    x = torch.tensor([[100.0]])
    y = torch.tensor([[102.0]])
    log_lik = model.log_likelihood(y, x, _zero_process_var(x))

    eps = model.eps.item()
    variance = (eps * 100.0) ** 2
    expected = -0.5 * ((102.0 - 100.0) ** 2 / variance + math.log(variance))
    assert pytest.approx(log_lik.item(), rel=1e-5) == expected


def test_log_likelihood_combines_process_and_obs_variance():
    """log_likelihood matches manual total-variance formula."""
    model = _make_direct(init_eps=0.1)
    x = torch.tensor([[100.0]])
    y = torch.tensor([[150.0]])  # large residual: 50
    process_var = torch.tensor([[400.0]])  # 20^2

    result = model.log_likelihood(y, x, process_var)

    eps = model.eps.item()
    obs_var = (eps * 100.0) ** 2
    total_var = obs_var + 400.0
    expected = -0.5 * ((150.0 - 100.0) ** 2 / total_var + math.log(total_var))
    assert pytest.approx(result.item(), rel=1e-5) == expected


def test_log_likelihood_reduces_high_count_nll():
    """At high molecule counts, obs noise prevents NLL explosion vs. process-only NLL.

    Without a measurement model, the process variance at X=100k is ~300^2=90000.
    A 5% drift error (residual=5000) gives residual^2/var = 5000^2/90000 ~ 278,
    causing catastrophic NLL. With the DirectObservation measurement model (eps=0.02),
    obs_var = (0.02*100000)^2 = 4e6. The combined NLL uses total_var ~ 4e6,
    giving residual^2/total ~ 6.1 — dramatically smaller.
    """
    model = _make_direct(init_eps=0.02)
    x = torch.tensor([[100_000.0]])
    y = torch.tensor([[105_000.0]])  # 5% drift error, residual=5000
    process_var = torch.tensor([[90_000.0]])  # 300^2

    # NLL using process variance only (no obs noise)
    process_only_var = process_var.clamp(min=1e-2)
    nll_process_only = 0.5 * ((y - x) ** 2 / process_only_var + process_only_var.log())

    nll_with_obs = -model.log_likelihood(y, x, process_var)

    # obs_var dominates (4e6 >> 90000), so the combined NLL is far smaller
    assert nll_with_obs.item() < nll_process_only.item()


def test_from_config():
    """from_config constructs correctly from MeasurementConfig."""
    cfg = MeasurementConfig(
        noise=NoiseConfig(
            mode=NoiseMode.LEARNED, sharing=NoiseSharing.SHARED, init_value=0.05
        ),
        min_variance=1e-2,
    )
    model = DirectObservation.from_config(cfg)
    assert pytest.approx(model.eps.item(), rel=1e-5) == 0.05
    assert model._min_variance == 1e-2
