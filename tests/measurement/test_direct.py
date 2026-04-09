"""Tests for DirectObservation measurement model."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.measurement.config import (
    MeasurementConfig,
    NoiseConfig,
    NoiseMode,
    NoiseSharing,
)
from crn_surrogate.measurement.direct import DirectObservation


def _make_model(
    mode: NoiseMode = NoiseMode.LEARNED,
    sharing: NoiseSharing = NoiseSharing.SHARED,
    init_value: float = 0.02,
    min_variance: float = 1e-2,
    n_species: int | None = None,
) -> DirectObservation:
    noise = NoiseConfig(mode=mode, sharing=sharing, init_value=init_value)
    return DirectObservation(
        noise_config=noise, min_variance=min_variance, n_species=n_species
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_from_config_shared() -> None:
    model = DirectObservation.from_config(MeasurementConfig())
    assert model.eps.numel() == 1


def test_from_config_per_species() -> None:
    cfg = MeasurementConfig(noise=NoiseConfig(sharing=NoiseSharing.PER_SPECIES))
    model = DirectObservation.from_config(cfg, n_species=3)
    assert model.eps.shape == (3,)


def test_per_species_requires_n_species() -> None:
    cfg = MeasurementConfig(noise=NoiseConfig(sharing=NoiseSharing.PER_SPECIES))
    with pytest.raises(ValueError):
        DirectObservation.from_config(cfg)


def test_rejects_nonpositive_min_variance() -> None:
    with pytest.raises(ValueError):
        _make_model(min_variance=0.0)
    with pytest.raises(ValueError):
        _make_model(min_variance=-1.0)


def test_rejects_nonpositive_init_value() -> None:
    with pytest.raises(ValueError):
        _make_model(init_value=0.0)


# ── eps property ──────────────────────────────────────────────────────────────


def test_eps_positive() -> None:
    model = _make_model()
    assert (model.eps > 0).all()


def test_eps_initial_value() -> None:
    model = _make_model(init_value=0.02)
    assert model.eps.item() == pytest.approx(0.02, abs=1e-4)


def test_eps_learned_has_grad() -> None:
    model = _make_model(mode=NoiseMode.LEARNED)
    assert model._raw_eps.requires_grad is True


def test_eps_fixed_no_grad() -> None:
    model = _make_model(mode=NoiseMode.FIXED)
    param_ids = {id(p) for p in model.parameters()}
    assert id(model._raw_eps) not in param_ids


# ── predict ───────────────────────────────────────────────────────────────────


def test_predict_identity() -> None:
    model = _make_model()
    x = torch.tensor([1.0, 5.0, 10.0])
    assert torch.equal(model.predict(x), x)


# ── sample ────────────────────────────────────────────────────────────────────


def test_sample_shape() -> None:
    model = _make_model()
    x = torch.tensor([10.0, 50.0, 200.0])
    out = model.sample(x)
    assert out.shape == x.shape


def test_sample_adds_noise() -> None:
    model = _make_model()
    x = torch.tensor([100.0, 50.0, 200.0])
    out = model.sample(x)
    assert not torch.equal(out, x)


def test_sample_noise_proportional_to_state() -> None:
    model = _make_model(init_value=0.1)
    n = 1000
    x_low = torch.tensor([10.0]).expand(n)
    x_high = torch.tensor([1000.0]).expand(n)

    samples_low = torch.stack([model.sample(x_low[i : i + 1]) for i in range(n)])
    samples_high = torch.stack([model.sample(x_high[i : i + 1]) for i in range(n)])

    std_low = samples_low.std().item()
    std_high = samples_high.std().item()
    assert std_high > 10 * std_low


# ── log_likelihood ────────────────────────────────────────────────────────────


def test_perfect_prediction_high_ll() -> None:
    model = _make_model()
    x = torch.tensor([10.0])
    process_var = torch.tensor([0.01])
    ll = model.log_likelihood(x, x, process_var)
    # Perfect prediction: residual=0, ll = -0.5 * log(v_total)
    # Should be a finite negative number close to max
    assert ll.isfinite().all()
    assert ll.item() > -10.0


def test_bad_prediction_low_ll() -> None:
    model = _make_model()
    y = torch.tensor([100.0])
    x = torch.tensor([10.0])
    process_var = torch.tensor([0.01])

    ll_good = model.log_likelihood(y, y, process_var)
    ll_bad = model.log_likelihood(y, x, process_var)
    assert ll_good.item() > ll_bad.item()


def test_log_likelihood_shape() -> None:
    model = _make_model()
    y = torch.randn(4, 3)
    x = torch.randn(4, 3)
    pv = torch.ones(4, 3) * 0.1
    ll = model.log_likelihood(y, x, pv)
    assert ll.shape == (4, 3)


def test_min_variance_prevents_explosion() -> None:
    model = _make_model(min_variance=0.01)
    x_pred = torch.tensor([0.0])
    y_obs = torch.tensor([0.0])
    process_var = torch.tensor([0.0])
    ll = model.log_likelihood(y_obs, x_pred, process_var)
    assert ll.isfinite().all()


def test_higher_process_variance_lowers_ll_penalty() -> None:
    model = _make_model()
    y = torch.tensor([15.0])  # residual = 5.0
    x = torch.tensor([10.0])

    ll_tight = model.log_likelihood(y, x, torch.tensor([0.01]))
    ll_wide = model.log_likelihood(y, x, torch.tensor([1.0]))
    # Wider variance means the Gaussian is flatter; the residual penalty is smaller.
    assert ll_wide.item() > ll_tight.item()


def test_gradient_flows_through_eps() -> None:
    model = _make_model(mode=NoiseMode.LEARNED)
    y = torch.tensor([10.0])
    x = torch.tensor([9.0])
    pv = torch.tensor([0.1])
    ll = model.log_likelihood(y, x, pv)
    ll.sum().backward()
    assert model._raw_eps.grad is not None


def test_gradient_does_not_flow_when_fixed() -> None:
    model = _make_model(mode=NoiseMode.FIXED)
    y = torch.tensor([10.0])
    x = torch.tensor([9.0], requires_grad=True)  # ensure backward has a leaf
    pv = torch.tensor([0.1])
    ll = model.log_likelihood(y, x, pv)
    ll.sum().backward()
    # Buffer: _raw_eps has no grad
    assert not hasattr(model._raw_eps, "grad") or model._raw_eps.grad is None
