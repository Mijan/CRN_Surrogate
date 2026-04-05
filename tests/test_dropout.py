"""Tests for dropout regularization in encoder and SDE MLP.

Covers:
- context_dropout is active in training mode, inactive in eval mode.
- mlp_dropout in ConditionedMLP is active in training, inactive in eval.
- Both drift and diffusion networks in CRNNeuralSDE respect mlp_dropout.
"""

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulator.conditioned_mlp import ConditionedMLP
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE

# ── Encoder context dropout ────────────────────────────────────────────────────


def test_encoder_context_dropout_active_in_train_mode():
    """With high dropout, consecutive training-mode forward passes differ."""
    config = EncoderConfig(d_model=32, n_layers=2, context_dropout=0.9)
    encoder = BipartiteGNNEncoder(config)
    crn_repr = crn_to_tensor_repr(birth_death())
    encoder.train()
    torch.manual_seed(0)
    ctx1 = encoder(crn_repr).context_vector
    torch.manual_seed(1)
    ctx2 = encoder(crn_repr).context_vector

    # With p=0.9 dropout over a 64-element vector, outputs almost certainly differ.
    assert not torch.equal(ctx1, ctx2)


def test_encoder_context_dropout_inactive_in_eval_mode():
    """In eval mode, the context vector is identical across forward passes."""
    config = EncoderConfig(d_model=32, n_layers=2, context_dropout=0.5)
    encoder = BipartiteGNNEncoder(config)
    crn_repr = crn_to_tensor_repr(birth_death())

    encoder.eval()
    ctx1 = encoder(crn_repr).context_vector
    ctx2 = encoder(crn_repr).context_vector

    assert torch.equal(ctx1, ctx2)


def test_encoder_zero_context_dropout_is_deterministic():
    """Zero dropout produces identical outputs in both train and eval modes."""
    config = EncoderConfig(d_model=32, n_layers=2, context_dropout=0.0)
    encoder = BipartiteGNNEncoder(config)
    crn_repr = crn_to_tensor_repr(birth_death())

    encoder.train()
    ctx1 = encoder(crn_repr).context_vector
    ctx2 = encoder(crn_repr).context_vector

    assert torch.equal(ctx1, ctx2)


# ── ConditionedMLP dropout ─────────────────────────────────────────────────────


def test_conditioned_mlp_dropout_active_in_train_mode():
    """With high MLP dropout, consecutive training-mode forward passes differ."""
    mlp = ConditionedMLP(
        d_in=4, d_hidden=32, d_out=4, d_context=8, n_hidden_layers=2, dropout=0.9
    )
    x = torch.ones(4)
    ctx = torch.ones(8)

    mlp.train()
    torch.manual_seed(0)
    out1 = mlp(x, ctx)
    torch.manual_seed(1)
    out2 = mlp(x, ctx)

    assert not torch.equal(out1, out2)


def test_conditioned_mlp_dropout_inactive_in_eval_mode():
    """In eval mode, MLP output is identical across forward passes."""
    mlp = ConditionedMLP(
        d_in=4, d_hidden=32, d_out=4, d_context=8, n_hidden_layers=2, dropout=0.5
    )
    x = torch.ones(4)
    ctx = torch.ones(8)

    mlp.eval()
    out1 = mlp(x, ctx)
    out2 = mlp(x, ctx)

    assert torch.equal(out1, out2)


# ── CRNNeuralSDE mlp_dropout ──────────────────────────────────────────────────


@pytest.fixture()
def sde_with_dropout() -> CRNNeuralSDE:
    config = SDEConfig(
        d_model=16,
        d_hidden=32,
        n_noise_channels=2,
        n_hidden_layers=2,
        mlp_dropout=0.9,
    )
    return CRNNeuralSDE(config, n_species=2)


def test_sde_drift_dropout_active_in_train_mode(sde_with_dropout):
    """Drift output differs across forward passes in training mode."""
    crn_repr = crn_to_tensor_repr(birth_death())
    config = EncoderConfig(d_model=16, n_layers=1)
    encoder = BipartiteGNNEncoder(config)
    crn_context = encoder(crn_repr)

    # Pad context_vector to match d_model=16 → 2*16=32
    # birth_death has 1 species; pad state to 2 for this SDE
    state = torch.tensor([5.0, 0.0])

    sde_with_dropout.train()
    torch.manual_seed(0)
    drift1 = sde_with_dropout.drift(torch.tensor(0.0), state, crn_context)
    torch.manual_seed(1)
    drift2 = sde_with_dropout.drift(torch.tensor(0.0), state, crn_context)

    assert not torch.equal(drift1, drift2)


def test_sde_drift_dropout_inactive_in_eval_mode(sde_with_dropout):
    """Drift output is identical across forward passes in eval mode."""
    crn_repr = crn_to_tensor_repr(birth_death())
    config = EncoderConfig(d_model=16, n_layers=1)
    encoder = BipartiteGNNEncoder(config)
    crn_context = encoder(crn_repr)

    state = torch.tensor([5.0, 0.0])

    sde_with_dropout.eval()
    drift1 = sde_with_dropout.drift(torch.tensor(0.0), state, crn_context)
    drift2 = sde_with_dropout.drift(torch.tensor(0.0), state, crn_context)

    assert torch.equal(drift1, drift2)
