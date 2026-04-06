"""Tests for batched GNN encoder and end-to-end training equivalence.

Covers:
- merge_bipartite_edges produces correct offset indices and isolated subgraphs.
- forward_batch is mathematically equivalent to B individual forward() calls.
- forward_batch handles a single-item list (edge case).
- Full training batch loss matches between batched and sequential encoding.
"""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.reference_crns import birth_death, lotka_volterra
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.graph_utils import merge_bipartite_edges
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.training.trainer import Trainer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_encoder(d_model: int = 16) -> BipartiteGNNEncoder:
    return BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=2))


def _make_3species_crn() -> CRN:
    return CRN(
        [
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
                propensity=constant_rate(k=1.0),
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 1.0, 0.0]),
                propensity=mass_action(0.5, torch.tensor([1.0, 0.0, 0.0])),
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, -1.0, 1.0]),
                propensity=mass_action(0.3, torch.tensor([0.0, 1.0, 0.0])),
            ),
            Reaction(
                stoichiometry=torch.tensor([0.0, 0.0, -1.0]),
                propensity=mass_action(0.4, torch.tensor([0.0, 0.0, 1.0])),
            ),
        ]
    )


def _sample_crn_reprs() -> list:
    """Three CRNs with different species/reaction counts on CPU."""
    return [
        crn_to_tensor_repr(birth_death()),  # 1 species, 2 reactions
        crn_to_tensor_repr(_make_3species_crn()),  # 3 species, 4 reactions
        crn_to_tensor_repr(lotka_volterra()),  # 2 species, 3 reactions
    ]


def _make_trainer(tmp_path, n_species: int = 3) -> tuple[Trainer, CRNTrajectoryDataset]:
    enc_cfg = EncoderConfig(d_model=16, n_layers=1)
    sde_cfg = SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4)
    model_config = ModelConfig(encoder=enc_cfg, sde=sde_cfg)
    encoder = BipartiteGNNEncoder(enc_cfg)
    sde = CRNNeuralSDE(sde_cfg, n_species=n_species)
    config = TrainingConfig(
        max_epochs=1,
        batch_size=4,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    trainer = Trainer(encoder, sde, model_config, config)

    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, 8)
    crn_1s = birth_death()
    crn_3s = _make_3species_crn()

    def _item(crn, init):
        crn_repr = crn_to_tensor_repr(crn)
        trajs = Trajectory.stack_on_grid(
            ssa.simulate_batch(
                stoichiometry=crn.stoichiometry_matrix,
                propensity_fn=crn.evaluate_propensities,
                initial_state=init.clone(),
                t_max=5.0,
                n_trajectories=4,
            ),
            time_grid,
        )
        return TrajectoryItem(
            crn_repr=crn_repr, initial_state=init, trajectories=trajs, times=time_grid
        )

    items = [_item(crn_1s, torch.tensor([5.0])) for _ in range(2)]
    items += [_item(crn_3s, torch.tensor([5.0, 2.0, 1.0])) for _ in range(2)]
    dataset = CRNTrajectoryDataset(items)
    return trainer, dataset


# ── Task 1: merge_bipartite_edges ─────────────────────────────────────────────


def test_merge_bipartite_edges_preserves_isolation():
    """No edge in the merged graph should connect species from one CRN to
    reactions from a different CRN."""
    reprs = _sample_crn_reprs()
    n_species = [r.n_species for r in reprs]
    n_reactions = [r.n_reactions for r in reprs]

    # Pre-build edges on CPU
    edges_list = [r.bipartite_edges for r in reprs]
    merged = merge_bipartite_edges(edges_list, n_species, n_reactions)

    # Compute per-CRN ranges
    spe_starts = [sum(n_species[:i]) for i in range(len(reprs))]
    rxn_starts = [sum(n_reactions[:i]) for i in range(len(reprs))]

    r2s = merged.rxn_to_species_index  # (2, E)
    for i in range(len(reprs)):
        # For every edge whose reaction belongs to CRN i, its species must also be in CRN i
        rxn_mask = (r2s[0] >= rxn_starts[i]) & (r2s[0] < rxn_starts[i] + n_reactions[i])
        species_in_edge = r2s[1][rxn_mask]
        assert (species_in_edge >= spe_starts[i]).all()
        assert (species_in_edge < spe_starts[i] + n_species[i]).all()


def test_merge_bipartite_edges_total_edge_count():
    """Total edges in merged graph equals the sum of individual edge counts."""
    reprs = _sample_crn_reprs()
    n_species = [r.n_species for r in reprs]
    n_reactions = [r.n_reactions for r in reprs]
    edges_list = [r.bipartite_edges for r in reprs]

    total_r2s = sum(e.rxn_to_species_index.shape[1] for e in edges_list)
    total_s2r = sum(e.species_to_rxn_index.shape[1] for e in edges_list)

    merged = merge_bipartite_edges(edges_list, n_species, n_reactions)
    assert merged.rxn_to_species_index.shape[1] == total_r2s
    assert merged.species_to_rxn_index.shape[1] == total_s2r


def test_merge_bipartite_edges_features_unchanged():
    """Edge features must be concatenated without modification."""
    reprs = _sample_crn_reprs()
    n_species = [r.n_species for r in reprs]
    n_reactions = [r.n_reactions for r in reprs]
    edges_list = [r.bipartite_edges for r in reprs]

    merged = merge_bipartite_edges(edges_list, n_species, n_reactions)
    expected_feats = torch.cat([e.rxn_to_species_feat for e in edges_list], dim=0)
    assert torch.equal(merged.rxn_to_species_feat, expected_feats)


# ── Task 4: forward_batch equivalence ────────────────────────────────────────


def test_forward_batch_matches_individual():
    """forward_batch must produce identical results to B individual forward() calls."""
    torch.manual_seed(0)
    encoder = _make_encoder()
    encoder.eval()

    reprs = _sample_crn_reprs()

    individual = [encoder(r) for r in reprs]
    individual_vectors = torch.stack([c.context_vector for c in individual])

    batched = encoder.forward_batch(reprs)
    batched_vectors = torch.stack([c.context_vector for c in batched])

    assert torch.allclose(individual_vectors, batched_vectors, atol=1e-5), (
        f"Max diff: {(individual_vectors - batched_vectors).abs().max().item():.2e}"
    )

    for i in range(len(reprs)):
        assert torch.allclose(
            individual[i].species_embeddings,
            batched[i].species_embeddings,
            atol=1e-5,
        ), f"Species embeddings differ for item {i}"
        assert torch.allclose(
            individual[i].reaction_embeddings,
            batched[i].reaction_embeddings,
            atol=1e-5,
        ), f"Reaction embeddings differ for item {i}"


def test_forward_batch_single_item():
    """forward_batch with a single item must match individual forward()."""
    torch.manual_seed(1)
    encoder = _make_encoder()
    encoder.eval()

    reprs = _sample_crn_reprs()
    single = encoder(reprs[0])
    batched = encoder.forward_batch([reprs[0]])
    assert torch.allclose(single.context_vector, batched[0].context_vector, atol=1e-6)


def test_forward_batch_output_count_matches_input():
    """forward_batch must return exactly as many CRNContexts as inputs."""
    encoder = _make_encoder()
    encoder.eval()
    reprs = _sample_crn_reprs()
    result = encoder.forward_batch(reprs)
    assert len(result) == len(reprs)


def test_forward_batch_context_vector_shape():
    """Each context vector must be (2 * d_model,)."""
    d_model = 16
    encoder = _make_encoder(d_model=d_model)
    encoder.eval()
    reprs = _sample_crn_reprs()
    result = encoder.forward_batch(reprs)
    for i, ctx in enumerate(result):
        assert ctx.context_vector.shape == (2 * d_model,), (
            f"Item {i}: expected ({2 * d_model},), got {ctx.context_vector.shape}"
        )


# ── Task 6: end-to-end equivalence ───────────────────────────────────────────


def test_batched_encoder_training_loss_matches(tmp_path):
    """Full batch loss must match between batched and sequential encoding."""
    torch.manual_seed(42)
    trainer, dataset = _make_trainer(tmp_path)
    trainer._encoder.eval()
    trainer._sde.eval()

    collator = CRNCollator(n_species_sde=trainer._sde.n_species)
    batch = collator(list(dataset))
    batch = trainer._batch_to_device(batch)

    # Batched path (uses _prepare_batch → forward_batch)
    batched_loss = trainer._compute_batch_loss(batch, epoch=1)

    # Sequential path (uses _prepare_item → individual forward() calls)
    B = batch["stoichiometry"].shape[0]
    items_seq = [trainer._prepare_item(batch, idx) for idx in range(B)]
    sequential_loss = trainer._compute_batch_nll_batched(items_seq)

    assert torch.allclose(batched_loss, sequential_loss, atol=1e-4), (
        f"Batched {batched_loss.item():.6f} != sequential {sequential_loss.item():.6f}"
    )
