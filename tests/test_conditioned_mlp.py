"""Tests for ConditionedMLP and FiLMLayer.

Covers:
- ConditionedMLP: output shapes (unbatched, batched, extra dims).
- ConditionedMLP: context actually influences output.
- ConditionedMLP: gradient flow to all parameters.
- ConditionedMLP: correct behaviour for n_hidden_layers = 1, 2, 3.
- ConditionedMLP: ValueError for n_hidden_layers = 0.
- ConditionedMLP: TypeError when arguments are passed positionally.
- FiLMLayer: output shape matches input shape.
- FiLMLayer: different contexts produce different outputs.
- NeuralSDE: different contexts produce different drift / diffusion.
- NeuralSDE: end-to-end gradient from drift output to encoder parameters.
- SDEConfig: n_hidden_layers validation.
"""

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulator.conditioned_mlp import ConditionedMLP
from crn_surrogate.simulator.film import FiLMLayer
from crn_surrogate.simulator.neural_sde import NeuralSDE

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mlp(n_hidden_layers: int = 2) -> ConditionedMLP:
    return ConditionedMLP(
        d_in=4, d_hidden=8, d_out=3, d_context=6, n_hidden_layers=n_hidden_layers
    )


def _make_context(d_model: int = 16):
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
    crn_repr = crn_to_tensor_repr(birth_death())
    return encoder(crn_repr)


# ── ConditionedMLP: output shapes ─────────────────────────────────────────────


def test_conditioned_mlp_unbatched_output_shape():
    """Unbatched input (d_in,) must produce output (d_out,)."""
    mlp = _make_mlp()
    out = mlp(torch.randn(4), torch.randn(6))
    assert out.shape == (3,)


def test_conditioned_mlp_batched_output_shape():
    """Batched input (B, d_in) must produce output (B, d_out)."""
    mlp = _make_mlp()
    out = mlp(torch.randn(5, 4), torch.randn(5, 6))
    assert out.shape == (5, 3)


def test_conditioned_mlp_extra_leading_dims_output_shape():
    """Extra leading dimensions are preserved in the output."""
    mlp = _make_mlp()
    out = mlp(torch.randn(2, 7, 4), torch.randn(2, 7, 6))
    assert out.shape == (2, 7, 3)


# ── ConditionedMLP: context sensitivity ───────────────────────────────────────


def test_conditioned_mlp_context_influences_output():
    """Same input x with two different context vectors must produce different outputs."""
    mlp = _make_mlp()
    x = torch.randn(4)
    out_a = mlp(x, torch.randn(6))
    out_b = mlp(x, torch.randn(6))
    assert not torch.allclose(out_a, out_b), (
        "ConditionedMLP produced identical output for different context vectors"
    )


# ── ConditionedMLP: gradient flow ─────────────────────────────────────────────


def test_conditioned_mlp_gradients_flow_to_all_parameters():
    """A scalar loss on the output must produce non-None gradients on every parameter."""
    mlp = _make_mlp()
    out = mlp(torch.randn(4), torch.randn(6))
    out.sum().backward()
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ── ConditionedMLP: depth variants ────────────────────────────────────────────


@pytest.mark.parametrize("n_hidden", [1, 2, 3])
def test_conditioned_mlp_n_hidden_layers_variants(n_hidden: int):
    """ConditionedMLP must work for n_hidden_layers = 1, 2, and 3."""
    mlp = _make_mlp(n_hidden_layers=n_hidden)
    out = mlp(torch.randn(4), torch.randn(6))
    assert out.shape == (3,)


def test_conditioned_mlp_n_hidden_layers_zero_raises():
    """n_hidden_layers=0 must raise ValueError."""
    with pytest.raises(ValueError, match="n_hidden_layers"):
        ConditionedMLP(d_in=4, d_hidden=8, d_out=3, d_context=6, n_hidden_layers=0)


# ── ConditionedMLP: keyword-only enforcement ──────────────────────────────────


def test_conditioned_mlp_positional_args_raise_type_error():
    """All constructor arguments are keyword-only; positional usage must raise TypeError."""
    with pytest.raises(TypeError):
        ConditionedMLP(4, 8, 3, 6)  # type: ignore[misc]


# ── FiLMLayer ─────────────────────────────────────────────────────────────────


def test_film_layer_output_shape_matches_input():
    """FiLMLayer must return the same shape as the input features."""
    film = FiLMLayer(d_context=6, d_features=8)
    x = torch.randn(5, 8)
    out = film(x, torch.randn(5, 6))
    assert out.shape == x.shape


def test_film_layer_different_contexts_produce_different_outputs():
    """Different context vectors must produce different modulated outputs."""
    film = FiLMLayer(d_context=6, d_features=8)
    x = torch.randn(8)
    out_a = film(x, torch.randn(6))
    out_b = film(x, torch.randn(6))
    assert not torch.allclose(out_a, out_b), (
        "FiLMLayer produced identical output for different context vectors"
    )


# ── NeuralSDE: context sensitivity ────────────────────────────────────────


def test_sde_different_contexts_produce_different_drift():
    """Two different CRN contexts must produce different drift vectors for the same state."""
    sde = NeuralSDE(SDEConfig(d_model=16, d_hidden=32, n_noise_channels=2), n_species=1)
    ctx_a = _make_context()
    ctx_b = _make_context()
    state = torch.tensor([5.0])
    t = torch.tensor(0.0)

    drift_a = sde.drift(t, state, ctx_a)
    drift_b = sde.drift(t, state, ctx_b)
    assert not torch.allclose(drift_a, drift_b), (
        "SDE drift is insensitive to CRN context"
    )


def test_sde_different_contexts_produce_different_diffusion():
    """Two different CRN contexts must produce different diffusion matrices."""
    sde = NeuralSDE(SDEConfig(d_model=16, d_hidden=32, n_noise_channels=2), n_species=1)
    ctx_a = _make_context()
    ctx_b = _make_context()
    state = torch.tensor([5.0])
    t = torch.tensor(0.0)

    diff_a = sde.diffusion(t, state, ctx_a)
    diff_b = sde.diffusion(t, state, ctx_b)
    assert not torch.allclose(diff_a, diff_b), (
        "SDE diffusion is insensitive to CRN context"
    )


# ── NeuralSDE: end-to-end gradient ────────────────────────────────────────


def test_sde_drift_gradients_flow_to_encoder_parameters():
    """A loss on drift output must propagate gradients back through the encoder."""
    d_model = 16
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
    sde = NeuralSDE(
        SDEConfig(d_model=d_model, d_hidden=32, n_noise_channels=2), n_species=1
    )

    crn_repr = crn_to_tensor_repr(birth_death())
    ctx = encoder(crn_repr)

    init = torch.tensor([5.0])
    drift = sde.drift(torch.tensor(0.0), init, ctx)
    drift.sum().backward()

    assert any(p.grad is not None for p in encoder.parameters()), (
        "No gradients reached encoder parameters from SDE drift"
    )


# ── SDEConfig: n_hidden_layers validation ─────────────────────────────────────


def test_sde_config_n_hidden_layers_zero_raises():
    """SDEConfig with n_hidden_layers=0 must raise ValueError."""
    with pytest.raises(ValueError, match="n_hidden_layers"):
        SDEConfig(n_hidden_layers=0)


def test_sde_config_n_hidden_layers_default_is_two():
    """Default n_hidden_layers is 2."""
    config = SDEConfig()
    assert config.n_hidden_layers == 2
