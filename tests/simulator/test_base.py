"""Tests for predict_transition on the SurrogateModel / StochasticSurrogate hierarchy."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE
from tests.simulator.conftest import make_fake_context

_N_SPECIES = 3
_N_NOISE_CHANNELS = 2
_BATCH = 10
_DT = 0.1

_CONFIG = SDEConfig(
    d_model=16,
    d_hidden=32,
    n_noise_channels=_N_NOISE_CHANNELS,
    n_hidden_layers=1,
)


@pytest.fixture
def drift_model() -> NeuralDrift:
    return NeuralDrift(_CONFIG, n_species=_N_SPECIES)


@pytest.fixture
def sde_model() -> NeuralSDE:
    return NeuralSDE(_CONFIG, n_species=_N_SPECIES)


@pytest.fixture
def ctx():
    return make_fake_context(d_model=16, n_species=_N_SPECIES, n_reactions=4)


@pytest.fixture
def batch_inputs(ctx):
    """Return (t, x, context_vector) expanded to batch size _BATCH."""
    x = torch.rand(_BATCH, _N_SPECIES) * 5 + 1
    t = torch.zeros(_BATCH)
    ctx_vec = ctx.context_vector.unsqueeze(0).expand(_BATCH, -1)
    return t, x, ctx_vec


# ── NeuralDrift (deterministic) ───────────────────────────────────────────────


def test_drift_predict_transition_shapes(drift_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    mu, variance = drift_model.predict_transition(t, x, ctx_vec, dt=_DT)
    assert mu.shape == (_BATCH, _N_SPECIES)
    assert variance.shape == (_BATCH, _N_SPECIES)


def test_drift_predict_transition_zero_variance(drift_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    _, variance = drift_model.predict_transition(t, x, ctx_vec, dt=_DT)
    assert variance.abs().max().item() == pytest.approx(0.0, abs=1e-9)


# ── NeuralSDE (stochastic) ────────────────────────────────────────────────────


def test_sde_predict_transition_shapes(sde_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    mu, variance = sde_model.predict_transition(t, x, ctx_vec, dt=_DT)
    assert mu.shape == (_BATCH, _N_SPECIES)
    assert variance.shape == (_BATCH, _N_SPECIES)


def test_sde_predict_transition_positive_variance(sde_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    _, variance = sde_model.predict_transition(t, x, ctx_vec, dt=_DT)
    assert (variance > 0).all()


def test_sde_predict_transition_mu_matches_manual(sde_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    mu, _ = sde_model.predict_transition(t, x, ctx_vec, dt=_DT)
    drift = sde_model.drift_from_context(t, x, ctx_vec)
    mu_manual = x + drift * _DT
    torch.testing.assert_close(mu, mu_manual, atol=1e-5, rtol=0.0)


def test_sde_predict_transition_variance_matches_manual(
    sde_model, batch_inputs
) -> None:
    t, x, ctx_vec = batch_inputs
    _, variance = sde_model.predict_transition(t, x, ctx_vec, dt=_DT)
    G = sde_model.diffusion_from_context(t, x, ctx_vec)
    variance_manual = (G**2).sum(dim=-1) * _DT
    torch.testing.assert_close(variance, variance_manual, atol=1e-5, rtol=0.0)


def test_predict_transition_gradients_flow(sde_model, batch_inputs) -> None:
    t, x, ctx_vec = batch_inputs
    mu, variance = sde_model.predict_transition(t, x, ctx_vec, dt=_DT)
    (mu.sum() + variance.sum()).backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in sde_model.parameters()
    )
    assert has_grad
