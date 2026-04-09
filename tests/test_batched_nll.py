"""Tests for batched drift/diffusion and the batched NLL training path.

Covers:
- drift_from_context produces identical output to drift() for single-item input.
- diffusion_from_context produces identical output to diffusion() for single-item input.
- drift_from_context handles (N, d_context) batched context vectors.
- diffusion_from_context handles (N, d_context) batched context vectors.
- _compute_batch_nll_batched matches the sequential per-item NLL to float tolerance.
- Masking is exercised by mixing 1-species and 3-species items in one batch.
"""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulation.gillespie import GillespieSSA
from crn_surrogate.simulation.trajectory import Trajectory
from crn_surrogate.simulator.neural_sde import NeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.trainer import Trainer

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_sde_and_encoder(d_model: int = 16, n_species: int = 3):
    enc_cfg = EncoderConfig(d_model=d_model, n_layers=1)
    sde_cfg = SDEConfig(d_model=d_model, d_hidden=32, n_noise_channels=4)
    encoder = BipartiteGNNEncoder(enc_cfg)
    sde = NeuralSDE(sde_cfg, n_species=n_species)
    return encoder, sde, enc_cfg, sde_cfg


def _birth_death_context(encoder, d_model: int = 16):
    crn = birth_death()
    crn_repr = crn_to_tensor_repr(crn)
    return encoder(crn_repr)


def _make_3species_crn():
    """3-species linear chain: A produced, A->B, B->C, C degrades."""
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


def _make_mixed_batch(
    encoder, sde, n_1s: int = 2, n_3s: int = 2, M: int = 4, T: int = 8
):
    """Dataset with n_1s 1-species items and n_3s 3-species items; SDE has n_species=3."""
    ssa = GillespieSSA()
    time_grid = torch.linspace(0.0, 5.0, T)

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
                n_trajectories=M,
            ),
            time_grid,
        )
        return TrajectoryItem(
            crn_repr=crn_repr, initial_state=init, trajectories=trajs, times=time_grid
        )

    items = [_item(crn_1s, torch.tensor([5.0])) for _ in range(n_1s)]
    items += [_item(crn_3s, torch.tensor([5.0, 2.0, 1.0])) for _ in range(n_3s)]
    return CRNTrajectoryDataset(items)


def _make_trainer(encoder, sde, enc_cfg, sde_cfg, tmp_path):
    model_config = ModelConfig(encoder=enc_cfg, sde=sde_cfg)
    config = TrainingConfig(
        max_epochs=1,
        batch_size=4,
        n_sde_samples=2,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        scheduler_type=SchedulerType.COSINE,
    )
    return Trainer(
        encoder,
        sde,
        model_config,
        config,
        simulator=EulerMaruyamaSolver(SolverConfig()),
    )


# ── drift_from_context / diffusion_from_context ───────────────────────────────


def test_drift_from_context_matches_drift():
    """drift_from_context must produce the same result as drift() for single-item input."""
    torch.manual_seed(0)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)

    N = 5
    t = torch.zeros(N)
    x = torch.randn(N, sde.n_species)

    out_ctx = sde.drift(t, x, ctx)
    out_vec = sde.drift_from_context(t, x, ctx.context_vector)

    assert torch.allclose(out_ctx, out_vec)


def test_diffusion_from_context_matches_diffusion():
    """diffusion_from_context must produce the same result as diffusion() for single-item input."""
    torch.manual_seed(0)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)

    N = 5
    t = torch.zeros(N)
    x = torch.randn(N, sde.n_species)

    out_ctx = sde.diffusion(t, x, ctx)
    out_vec = sde.diffusion_from_context(t, x, ctx.context_vector)

    assert torch.allclose(out_ctx, out_vec)


def test_drift_from_context_batched_shape():
    """drift_from_context with (N, d_context) context must return (N, n_species)."""
    torch.manual_seed(1)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)

    N = 100
    t = torch.zeros(N)
    x = torch.randn(N, sde.n_species)
    ctx_expanded = ctx.context_vector.unsqueeze(0).expand(N, -1)

    out = sde.drift_from_context(t, x, ctx_expanded)
    assert out.shape == (N, sde.n_species)


def test_diffusion_from_context_batched_shape():
    """diffusion_from_context with (N, d_context) context must return (N, n_species, n_noise)."""
    torch.manual_seed(2)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)
    n_noise = sde._config.n_noise_channels

    N = 100
    t = torch.zeros(N)
    x = torch.randn(N, sde.n_species)
    ctx_expanded = ctx.context_vector.unsqueeze(0).expand(N, -1)

    out = sde.diffusion_from_context(t, x, ctx_expanded)
    assert out.shape == (N, sde.n_species, n_noise)


def test_drift_from_context_1d_state_matches_drift():
    """drift_from_context with 1D state delegates correctly (same as drift)."""
    torch.manual_seed(3)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)
    state = torch.randn(sde.n_species)

    out_ctx = sde.drift(torch.tensor(0.0), state, ctx)
    out_vec = sde.drift_from_context(torch.tensor(0.0), state, ctx.context_vector)
    assert torch.allclose(out_ctx, out_vec)
    assert out_vec.shape == (sde.n_species,)


def test_diffusion_from_context_1d_state_matches_diffusion():
    """diffusion_from_context with 1D state delegates correctly (same as diffusion)."""
    torch.manual_seed(4)
    encoder, sde, _, _ = _make_sde_and_encoder()
    ctx = _birth_death_context(encoder)
    n_noise = sde._config.n_noise_channels
    state = torch.randn(sde.n_species)

    out_ctx = sde.diffusion(torch.tensor(0.0), state, ctx)
    out_vec = sde.diffusion_from_context(torch.tensor(0.0), state, ctx.context_vector)
    assert torch.allclose(out_ctx, out_vec)
    assert out_vec.shape == (sde.n_species, n_noise)


# ── Batched NLL equivalence ───────────────────────────────────────────────────


def test_batched_nll_matches_sequential(tmp_path):
    """_compute_batch_nll_batched must match sequential per-item NLL to float tolerance.

    Uses a mixed 1-species and 3-species batch to exercise the masking logic.
    """
    torch.manual_seed(42)
    encoder, sde, enc_cfg, sde_cfg = _make_sde_and_encoder(d_model=16, n_species=3)
    trainer = _make_trainer(encoder, sde, enc_cfg, sde_cfg, tmp_path)

    dataset = _make_mixed_batch(encoder, sde, n_1s=2, n_3s=2)
    collator = CRNCollator()
    batch = collator(list(dataset))
    batch = trainer._batch_to_device(batch)
    B = batch["stoichiometry"].shape[0]

    # Sequential: sum per-item NLL / B
    items = [trainer._prepare_item(batch, idx) for idx in range(B)]
    sequential_total = torch.zeros(1, device=batch["stoichiometry"].device)
    for item in items:
        sequential_total = sequential_total + trainer._nll_loss.compute(
            sde=trainer._model,
            crn_context=item.context,
            true_trajectory=item.true_trajs_padded,
            times=item.times,
            dt=trainer._train_config.dt,
            mask=item.species_mask,
        )
    sequential_loss = sequential_total / B

    # Batched
    batched_loss = trainer._compute_batch_nll_batched(items)

    assert torch.allclose(sequential_loss, batched_loss, atol=1e-4, rtol=1e-4), (
        f"Sequential {sequential_loss.item():.6f} != batched {batched_loss.item():.6f}"
    )


def test_batched_nll_is_finite(tmp_path):
    """_compute_batch_nll_batched must return a finite scalar."""
    torch.manual_seed(5)
    encoder, sde, enc_cfg, sde_cfg = _make_sde_and_encoder(d_model=16, n_species=3)
    trainer = _make_trainer(encoder, sde, enc_cfg, sde_cfg, tmp_path)

    dataset = _make_mixed_batch(encoder, sde)
    collator = CRNCollator()
    batch = collator(list(dataset))
    batch = trainer._batch_to_device(batch)
    B = batch["stoichiometry"].shape[0]

    items = [trainer._prepare_item(batch, idx) for idx in range(B)]
    loss = trainer._compute_batch_nll_batched(items)

    assert loss.shape == ()
    assert loss.item() == loss.item()  # not NaN
    assert loss.item() < float("inf")


def test_batched_nll_gradients_flow(tmp_path):
    """Gradients must flow from _compute_batch_nll_batched back to SDE parameters."""
    torch.manual_seed(6)
    encoder, sde, enc_cfg, sde_cfg = _make_sde_and_encoder(d_model=16, n_species=3)
    trainer = _make_trainer(encoder, sde, enc_cfg, sde_cfg, tmp_path)

    dataset = _make_mixed_batch(encoder, sde)
    collator = CRNCollator()
    batch = collator(list(dataset))
    batch = trainer._batch_to_device(batch)
    B = batch["stoichiometry"].shape[0]

    items = [trainer._prepare_item(batch, idx) for idx in range(B)]
    loss = trainer._compute_batch_nll_batched(items)
    loss.backward()

    assert any(p.grad is not None for p in sde.parameters()), (
        "No gradients reached SDE parameters from batched NLL"
    )
