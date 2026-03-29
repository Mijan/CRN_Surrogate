"""Tests for the bipartite GNN encoder.

Covers:
- Output tensor shapes for 1-species and 2-species CRNs.
- context_vector dimension is always 2 * d_model (species + reaction mean pools).
- Different initial states produce different encodings (encoder is not ignoring state).
- Gradients flow through all encoder parameters.
- Encoder respects dropout=0 (deterministic output in eval mode).
"""

import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.data.gillespie import birth_death_crn, lotka_volterra_crn
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder


# ── Output shapes ─────────────────────────────────────────────────────────────


def test_encoder_birth_death_output_shapes():
    """Birth-death (1 species, 2 reactions): verify all three output tensor shapes."""
    d_model = 16
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=2))
    ctx = encoder(birth_death_crn(), torch.tensor([5.0]))

    assert ctx.species_embeddings.shape == (1, d_model)
    assert ctx.reaction_embeddings.shape == (2, d_model)
    assert ctx.context_vector.shape == (2 * d_model,)


def test_encoder_lotka_volterra_output_shapes():
    """Lotka-Volterra (2 species, 3 reactions): verify all three output tensor shapes."""
    d_model = 32
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=2))
    ctx = encoder(lotka_volterra_crn(), torch.tensor([50.0, 20.0]))

    assert ctx.species_embeddings.shape == (2, d_model)
    assert ctx.reaction_embeddings.shape == (3, d_model)
    assert ctx.context_vector.shape == (2 * d_model,)


def test_encoder_context_vector_dimension_is_twice_d_model():
    """context_vector concatenates the mean-pooled species and reaction embeddings,
    so its length must always equal 2 * d_model regardless of CRN size."""
    for d_model in (8, 24, 64):
        encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
        ctx = encoder(birth_death_crn(), torch.tensor([1.0]))
        assert ctx.context_vector.shape[0] == 2 * d_model, (
            f"d_model={d_model}: expected context dim {2*d_model}, "
            f"got {ctx.context_vector.shape[0]}"
        )


# ── State sensitivity ─────────────────────────────────────────────────────────


def test_encoder_different_initial_states_produce_different_context_vectors():
    """The encoder receives the initial state as input, so different states must
    produce distinguishably different context vectors."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2))
    crn = birth_death_crn()

    ctx_low = encoder(crn, torch.tensor([0.0]))
    ctx_high = encoder(crn, torch.tensor([100.0]))

    assert not torch.allclose(ctx_low.context_vector, ctx_high.context_vector), (
        "Encoder produced identical context for X=0 and X=100 — state input is not used"
    )


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_encoder_gradients_flow_through_all_parameters():
    """A scalar loss on the context vector must produce non-None gradients on every
    encoder parameter (verifies no dead computation graph branches)."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2))
    ctx = encoder(birth_death_crn(), torch.tensor([5.0]))
    ctx.context_vector.sum().backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ── Determinism ───────────────────────────────────────────────────────────────


def test_encoder_deterministic_in_eval_mode_with_no_dropout():
    """With dropout=0.0, two forward passes in eval mode must return identical outputs."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2, dropout=0.0))
    encoder.eval()
    crn = birth_death_crn()
    init = torch.tensor([5.0])

    ctx1 = encoder(crn, init)
    ctx2 = encoder(crn, init)

    assert torch.allclose(ctx1.context_vector, ctx2.context_vector)
