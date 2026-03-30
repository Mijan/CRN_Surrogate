"""Tests for the bipartite GNN encoder.

Covers:
- Output tensor shapes for 1-species and 2-species CRNs.
- context_vector dimension is always 2 * d_model.
- Different initial states produce different encodings.
- Gradients flow through all encoder parameters.
- Encoder is deterministic in eval mode with dropout=0.
- EdgeFeature enum / EDGE_FEAT_DIM single source of truth.
- SumMessagePassingLayer output shapes.
- AttentiveMessagePassingLayer: output shapes, attention weights sum to 1,
  gradient flow, single-edge attention weight == 1.
- BipartiteGNNEncoder attention vs. sum A/B test.
"""

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.crn.examples import birth_death, lotka_volterra
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.graph_utils import (
    EDGE_FEAT_DIM,
    EdgeFeature,
    build_bipartite_edges,
)
from crn_surrogate.encoder.message_passing import (
    AttentiveMessagePassingLayer,
    SumMessagePassingLayer,
)
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

# ── Output shapes ─────────────────────────────────────────────────────────────


def test_encoder_birth_death_output_shapes():
    """Birth-death (1 species, 2 reactions): verify all three output tensor shapes."""
    d_model = 16
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=2))
    crn_repr = crn_to_tensor_repr(birth_death())
    ctx = encoder(crn_repr, torch.tensor([5.0]))

    assert ctx.species_embeddings.shape == (1, d_model)
    assert ctx.reaction_embeddings.shape == (2, d_model)
    assert ctx.context_vector.shape == (2 * d_model,)


def test_encoder_lotka_volterra_output_shapes():
    """Lotka-Volterra (2 species, 3 reactions): verify all three output tensor shapes."""
    d_model = 32
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=2))
    crn_repr = crn_to_tensor_repr(lotka_volterra())
    ctx = encoder(crn_repr, torch.tensor([50.0, 20.0]))

    assert ctx.species_embeddings.shape == (2, d_model)
    assert ctx.reaction_embeddings.shape == (3, d_model)
    assert ctx.context_vector.shape == (2 * d_model,)


def test_encoder_context_vector_dimension_is_twice_d_model():
    """context_vector always equals 2 * d_model regardless of CRN size."""
    for d_model in (8, 24, 64):
        encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
        crn_repr = crn_to_tensor_repr(birth_death())
        ctx = encoder(crn_repr, torch.tensor([1.0]))
        assert ctx.context_vector.shape[0] == 2 * d_model, (
            f"d_model={d_model}: expected context dim {2 * d_model}, "
            f"got {ctx.context_vector.shape[0]}"
        )


# ── State sensitivity ─────────────────────────────────────────────────────────


def test_encoder_different_initial_states_produce_different_context_vectors():
    """Different initial states must produce distinguishably different context vectors."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2))
    crn_repr = crn_to_tensor_repr(birth_death())

    ctx_low = encoder(crn_repr, torch.tensor([0.0]))
    ctx_high = encoder(crn_repr, torch.tensor([100.0]))

    assert not torch.allclose(ctx_low.context_vector, ctx_high.context_vector), (
        "Encoder produced identical context for X=0 and X=100 — state input is not used"
    )


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_encoder_gradients_flow_through_all_parameters():
    """A scalar loss on the context vector must produce non-None gradients on every
    encoder parameter (verifies no dead computation graph branches)."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2))
    crn_repr = crn_to_tensor_repr(birth_death())
    ctx = encoder(crn_repr, torch.tensor([5.0]))
    ctx.context_vector.sum().backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ── Determinism ───────────────────────────────────────────────────────────────


def test_encoder_deterministic_in_eval_mode_with_no_dropout():
    """With dropout=0.0, two forward passes in eval mode must return identical outputs."""
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2, dropout=0.0))
    encoder.eval()
    crn_repr = crn_to_tensor_repr(birth_death())
    init = torch.tensor([5.0])

    ctx1 = encoder(crn_repr, init)
    ctx2 = encoder(crn_repr, init)

    assert torch.allclose(ctx1.context_vector, ctx2.context_vector)


# ── EdgeFeature single source of truth ────────────────────────────────────────


def test_edge_feat_dim_matches_len_edge_feature_enum():
    """EDGE_FEAT_DIM must equal the number of entries in EdgeFeature."""
    assert EDGE_FEAT_DIM == len(EdgeFeature)


def test_edge_feature_enum_has_expected_channels():
    """EdgeFeature must define NET_CHANGE, IS_STOICHIOMETRIC, IS_DEPENDENCY."""
    assert EdgeFeature.NET_CHANGE == 0
    assert EdgeFeature.IS_STOICHIOMETRIC == 1
    assert EdgeFeature.IS_DEPENDENCY == 2


def test_build_bipartite_edges_feat_dim_matches_edge_feat_dim():
    """build_bipartite_edges must produce feature vectors of width EDGE_FEAT_DIM."""
    crn = birth_death()
    edges = build_bipartite_edges(crn.stoichiometry_matrix, crn.dependency_matrix)
    assert edges.rxn_to_species_feat.shape[1] == EDGE_FEAT_DIM
    assert edges.edge_feat_dim == EDGE_FEAT_DIM


# ── SumMessagePassingLayer ────────────────────────────────────────────────────


def test_sum_layer_output_shapes_birth_death():
    """SumMessagePassingLayer returns tensors of the same shape as its inputs."""
    d_model = 16
    layer = SumMessagePassingLayer(d_model)
    crn = birth_death()
    edges = build_bipartite_edges(crn.stoichiometry_matrix, crn.dependency_matrix)

    h_species = torch.randn(crn.n_species, d_model)
    h_reactions = torch.randn(crn.n_reactions, d_model)

    h_s_out, h_r_out = layer(h_species, h_reactions, edges)
    assert h_s_out.shape == (crn.n_species, d_model)
    assert h_r_out.shape == (crn.n_reactions, d_model)


# ── AttentiveMessagePassingLayer ──────────────────────────────────────────────


def test_attentive_layer_output_shapes_birth_death():
    """AttentiveMessagePassingLayer returns tensors of the same shape as inputs."""
    d_model = 16
    layer = AttentiveMessagePassingLayer(d_model)
    crn = birth_death()
    edges = build_bipartite_edges(crn.stoichiometry_matrix, crn.dependency_matrix)

    h_species = torch.randn(crn.n_species, d_model)
    h_reactions = torch.randn(crn.n_reactions, d_model)

    h_s_out, h_r_out = layer(h_species, h_reactions, edges)
    assert h_s_out.shape == (crn.n_species, d_model)
    assert h_r_out.shape == (crn.n_reactions, d_model)


def test_attentive_layer_single_edge_weight_is_one():
    """A species receiving exactly one message must have attention weight 1.0."""
    d_model = 16
    layer = AttentiveMessagePassingLayer(d_model)

    # Minimal graph: 1 reaction → 1 species
    h_species = torch.randn(1, d_model)
    h_reactions = torch.randn(1, d_model)

    from crn_surrogate.encoder.graph_utils import BipartiteEdges

    edges = BipartiteEdges(
        rxn_to_species_index=torch.tensor([[0], [0]]),
        rxn_to_species_feat=torch.zeros(1, EDGE_FEAT_DIM),
        species_to_rxn_index=torch.tensor([[0], [0]]),
        species_to_rxn_feat=torch.zeros(1, EDGE_FEAT_DIM),
    )

    # Compute attention weights via the static helper directly
    raw_scores = torch.tensor([2.5])
    weights = AttentiveMessagePassingLayer._scatter_softmax(raw_scores, torch.tensor([0]), 1)
    assert weights.item() == pytest.approx(1.0)


def test_attentive_layer_weights_sum_to_one_per_receiving_node():
    """Attention weights must sum to 1.0 per receiving node."""
    d_model = 16
    n_reactions = 4
    n_species = 2

    # Build a manual multi-edge graph: reactions 0,1,2 → species 0; reaction 3 → species 1
    rxn_idx = torch.tensor([0, 1, 2, 3])
    spe_idx = torch.tensor([0, 0, 0, 1])
    raw_scores = torch.randn(4)

    weights = AttentiveMessagePassingLayer._scatter_softmax(raw_scores, spe_idx, n_species)

    # Weights for species 0 (edges 0,1,2) must sum to 1
    assert weights[:3].sum().item() == pytest.approx(1.0, abs=1e-5)
    # Weights for species 1 (edge 3) must equal 1
    assert weights[3].item() == pytest.approx(1.0, abs=1e-5)


def test_attentive_layer_gradients_flow():
    """Gradients must flow through AttentiveMessagePassingLayer to all parameters."""
    d_model = 16
    layer = AttentiveMessagePassingLayer(d_model)
    crn = birth_death()
    edges = build_bipartite_edges(crn.stoichiometry_matrix, crn.dependency_matrix)

    h_species = torch.randn(crn.n_species, d_model, requires_grad=True)
    h_reactions = torch.randn(crn.n_reactions, d_model, requires_grad=True)

    h_s_out, h_r_out = layer(h_species, h_reactions, edges)
    (h_s_out.sum() + h_r_out.sum()).backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ── Attention vs. sum A/B test ────────────────────────────────────────────────


def test_encoder_attention_and_sum_produce_different_outputs():
    """With identical weights (same seed), attention and sum encoders produce different outputs."""
    crn_repr = crn_to_tensor_repr(lotka_volterra())
    init_state = torch.tensor([50.0, 20.0])

    torch.manual_seed(0)
    enc_sum = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2, use_attention=False))
    ctx_sum = enc_sum(crn_repr, init_state)

    torch.manual_seed(0)
    enc_att = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=2, use_attention=True))
    ctx_att = enc_att(crn_repr, init_state)

    assert not torch.allclose(ctx_sum.context_vector, ctx_att.context_vector), (
        "Sum and attentive encoders produced identical outputs — attention has no effect"
    )


def test_encoder_with_attention_gradients_flow():
    """Gradients must reach all parameters of the attentive encoder."""
    encoder = BipartiteGNNEncoder(
        EncoderConfig(d_model=16, n_layers=2, use_attention=True)
    )
    crn_repr = crn_to_tensor_repr(birth_death())
    ctx = encoder(crn_repr, torch.tensor([5.0]))
    ctx.context_vector.sum().backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
