"""Tests for NeuralDrift and NeuralSDE type hierarchy and forward methods."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.simulator.base import StochasticSurrogate, SurrogateModel
from crn_surrogate.simulator.neural_sde import NeuralDrift, NeuralSDE

from .conftest import make_fake_context

N_SPECIES = 2
T = torch.tensor(0.0)


# ── Type hierarchy ────────────────────────────────────────────────────────────


def test_neural_drift_is_surrogate_model(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    assert isinstance(model, SurrogateModel)


def test_neural_drift_is_not_stochastic(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    assert not isinstance(model, StochasticSurrogate)


def test_neural_sde_is_stochastic_surrogate(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    assert isinstance(model, StochasticSurrogate)


def test_neural_sde_is_surrogate_model(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    assert isinstance(model, SurrogateModel)


# ── NeuralDrift ───────────────────────────────────────────────────────────────


def test_drift_output_shape_unbatched(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    out = model.drift(T, state, ctx)
    assert out.shape == (N_SPECIES,)


def test_drift_output_shape_batched(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    B = 5
    state = torch.randn(B, N_SPECIES)
    out = model.drift(T, state, ctx)
    assert out.shape == (B, N_SPECIES)


def test_drift_from_context_matches_drift(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES).eval()
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    with torch.no_grad():
        out_drift = model.drift(T, state, ctx)
        out_from_ctx = model.drift_from_context(T, state, ctx.context_vector)
    assert torch.allclose(out_drift, out_from_ctx)


def test_n_species_property(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    assert model.n_species == N_SPECIES


def test_no_diffusion_parameters(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    has_diff_param = any("diff" in name for name, _ in model.named_parameters())
    assert not has_diff_param


# ── NeuralSDE ─────────────────────────────────────────────────────────────────


def test_diffusion_output_shape_unbatched(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    out = model.diffusion(T, state, ctx)
    assert out.shape == (N_SPECIES, small_sde_config.n_noise_channels)


def test_diffusion_output_shape_batched(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    B = 5
    state = torch.randn(B, N_SPECIES)
    out = model.diffusion(T, state, ctx)
    assert out.shape == (B, N_SPECIES, small_sde_config.n_noise_channels)


def test_diffusion_non_negative(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    out = model.diffusion(T, state, ctx)
    assert (out >= 0).all()


def test_diffusion_from_context_matches_diffusion(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES).eval()
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    with torch.no_grad():
        out_diff = model.diffusion(T, state, ctx)
        out_from_ctx = model.diffusion_from_context(T, state, ctx.context_vector)
    assert torch.allclose(out_diff, out_from_ctx)


def test_sde_has_more_params_than_drift(small_sde_config: SDEConfig) -> None:
    drift = NeuralDrift(small_sde_config, N_SPECIES)
    sde = NeuralSDE(small_sde_config, N_SPECIES)
    n_drift = sum(p.numel() for p in drift.parameters())
    n_sde = sum(p.numel() for p in sde.parameters())
    assert n_sde > n_drift


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_drift_gradients_flow(small_sde_config: SDEConfig) -> None:
    model = NeuralDrift(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    out = model.drift(T, state, ctx)
    out.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad


def test_diffusion_gradients_flow(small_sde_config: SDEConfig) -> None:
    model = NeuralSDE(small_sde_config, N_SPECIES)
    ctx = make_fake_context(small_sde_config.d_model)
    state = torch.randn(N_SPECIES)
    out = model.diffusion(T, state, ctx)
    out.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )
    assert has_grad
