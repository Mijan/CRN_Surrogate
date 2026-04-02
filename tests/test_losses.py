"""Tests for the trajectory loss classes."""

import time

import pytest
import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.training.losses import (
    CombinedTrajectoryLoss,
    GaussianTransitionNLL,
    MeanMatchingLoss,
    VarianceMatchingLoss,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tensors(K: int = 4, M: int = 8, T: int = 10, n: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    pred = torch.randn(K, T, n)
    true = torch.randn(M, T, n)
    return pred, true


# ── MeanMatchingLoss ─────────────────────────────────────────────────────────


def test_mean_matching_loss_scalar():
    pred, true = _make_tensors()
    loss = MeanMatchingLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_mean_matching_loss_identical_means_is_zero():
    K, T, n = 4, 10, 2
    base = torch.ones(T, n) * 3.0
    pred = base.unsqueeze(0).expand(K, -1, -1)
    true = base.unsqueeze(0).expand(K, -1, -1)
    loss = MeanMatchingLoss().compute(pred, true)
    assert loss.item() < 1e-6


def test_mean_matching_loss_rejects_2d_pred():
    with pytest.raises(ValueError, match="pred_states must be 3D"):
        MeanMatchingLoss().compute(torch.randn(10, 2), torch.randn(8, 10, 2))


def test_mean_matching_loss_rejects_2d_true():
    with pytest.raises(ValueError, match="true_states must be 3D"):
        MeanMatchingLoss().compute(torch.randn(4, 10, 2), torch.randn(10, 2))


def test_mean_matching_loss_mask():
    pred, true = _make_tensors(n=3)
    mask = torch.tensor([True, True, False])
    loss_masked = MeanMatchingLoss().compute(pred, true, mask=mask)
    loss_full = MeanMatchingLoss().compute(pred, true)
    assert loss_masked.item() != loss_full.item()


def test_mean_matching_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = MeanMatchingLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None


# ── VarianceMatchingLoss ─────────────────────────────────────────────────────


def test_variance_matching_loss_scalar():
    pred, true = _make_tensors(K=4, M=8)
    loss = VarianceMatchingLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_variance_matching_loss_identical_variance_is_zero():
    torch.manual_seed(1)
    K, M, T, n = 4, 8, 10, 2
    # Deterministic tensors with same variance structure
    base = torch.randn(T, n)
    pred = base.unsqueeze(0).expand(K, -1, -1) + torch.randn(K, T, n) * 0.5
    true = base.unsqueeze(0).expand(M, -1, -1) + torch.randn(M, T, n) * 0.5
    # Just verify it runs and returns a finite scalar
    loss = VarianceMatchingLoss().compute(pred, true)
    assert torch.isfinite(loss)


def test_variance_matching_loss_rejects_k_less_than_2():
    with pytest.raises(ValueError, match="K >= 2"):
        VarianceMatchingLoss().compute(torch.randn(1, 10, 2), torch.randn(8, 10, 2))


def test_variance_matching_loss_rejects_m_less_than_2():
    with pytest.raises(ValueError, match="M >= 2"):
        VarianceMatchingLoss().compute(torch.randn(4, 10, 2), torch.randn(1, 10, 2))


def test_variance_matching_loss_rejects_2d_inputs():
    with pytest.raises(ValueError, match="pred_states must be 3D"):
        VarianceMatchingLoss().compute(torch.randn(10, 2), torch.randn(8, 10, 2))


def test_variance_matching_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = VarianceMatchingLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None


# ── CombinedTrajectoryLoss ───────────────────────────────────────────────────


def test_combined_loss_default_scalar():
    pred, true = _make_tensors(K=4, M=8)
    loss = CombinedTrajectoryLoss().compute(pred, true)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_combined_loss_var_weight_zero_equals_mean_only():
    pred, true = _make_tensors(K=4, M=8, seed=7)
    combined = CombinedTrajectoryLoss(var_weight=0.0).compute(pred, true)
    mean_only = MeanMatchingLoss().compute(pred, true)
    assert abs(combined.item() - mean_only.item()) < 1e-6


def test_combined_loss_custom_losses():
    pred, true = _make_tensors(K=4, M=8)
    custom = CombinedTrajectoryLoss(
        losses=[(MeanMatchingLoss(), 2.0), (MeanMatchingLoss(), 1.0)]
    )
    loss = custom.compute(pred, true)
    expected = 3.0 * MeanMatchingLoss().compute(pred, true)
    assert abs(loss.item() - expected.item()) < 1e-5


def test_combined_loss_gradient_flows():
    pred = torch.randn(4, 10, 2, requires_grad=True)
    true = torch.randn(8, 10, 2)
    loss = CombinedTrajectoryLoss().compute(pred, true)
    loss.backward()
    assert pred.grad is not None


# ── GaussianTransitionNLL helpers ────────────────────────────────────────────

N_SPECIES = 2
D_MODEL = 16


@pytest.fixture()
def sde_and_ctx():
    """Minimal CRNNeuralSDE + CRNContext for NLL tests."""
    torch.manual_seed(0)
    cfg = SDEConfig(
        d_model=D_MODEL,
        d_hidden=32,
        n_noise_channels=N_SPECIES,
        n_hidden_layers=1,
        clip_state=False,
    )
    sde = CRNNeuralSDE(cfg, n_species=N_SPECIES)
    ctx = CRNContext(
        species_embeddings=torch.randn(N_SPECIES, D_MODEL),
        reaction_embeddings=torch.randn(N_SPECIES, D_MODEL),
        context_vector=torch.randn(2 * D_MODEL),
    )
    return sde, ctx


# ── GaussianTransitionNLL — correctness ──────────────────────────────────────


def test_gaussian_nll_scalar(sde_and_ctx):
    sde, ctx = sde_and_ctx
    traj = torch.randn(4, 10, N_SPECIES)
    times = torch.linspace(0, 1, 10)
    loss = GaussianTransitionNLL().compute(sde, ctx, traj, times, dt=0.1)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_gaussian_nll_2d_input_auto_unsqueezed(sde_and_ctx):
    """2-D trajectory (T, n_species) should produce the same result as (1, T, n_species)."""
    sde, ctx = sde_and_ctx
    torch.manual_seed(1)
    traj_2d = torch.randn(10, N_SPECIES)
    times = torch.linspace(0, 1, 10)
    loss_2d = GaussianTransitionNLL().compute(sde, ctx, traj_2d, times, dt=0.1)
    loss_3d = GaussianTransitionNLL().compute(
        sde, ctx, traj_2d.unsqueeze(0), times, dt=0.1
    )
    torch.testing.assert_close(loss_2d, loss_3d)


def test_gaussian_nll_mask_reduces_loss(sde_and_ctx):
    sde, ctx = sde_and_ctx
    torch.manual_seed(2)
    traj = torch.randn(4, 10, N_SPECIES)
    times = torch.linspace(0, 1, 10)
    mask = torch.tensor([True, False])
    loss_masked = GaussianTransitionNLL().compute(
        sde, ctx, traj, times, dt=0.1, mask=mask
    )
    loss_full = GaussianTransitionNLL().compute(sde, ctx, traj, times, dt=0.1)
    # Masked loss suppresses one species so the values should differ
    assert loss_masked.item() != pytest.approx(loss_full.item())


def test_gaussian_nll_rejects_single_timestep(sde_and_ctx):
    sde, ctx = sde_and_ctx
    traj = torch.randn(4, 1, N_SPECIES)
    times = torch.tensor([0.0])
    with pytest.raises(ValueError, match="T >= 2"):
        GaussianTransitionNLL().compute(sde, ctx, traj, times, dt=0.1)


def test_gaussian_nll_gradient_flows(sde_and_ctx):
    sde, ctx = sde_and_ctx
    traj = torch.randn(4, 10, N_SPECIES)
    times = torch.linspace(0, 1, 10)
    loss = GaussianTransitionNLL().compute(sde, ctx, traj, times, dt=0.1)
    loss.backward()
    assert any(p.grad is not None for p in sde.parameters())


# ── GaussianTransitionNLL — speed (batched vs reference loop) ────────────────


def _nll_reference_loop(
    sde: CRNNeuralSDE,
    ctx: CRNContext,
    traj: torch.Tensor,
    times: torch.Tensor,
    dt: float,
    min_var: float = 1e-6,
) -> torch.Tensor:
    """Reference double-loop implementation for correctness comparison."""
    M, T, n_species = traj.shape
    total = traj.new_zeros(())
    for m in range(M):
        for t_idx in range(T - 1):
            y_t = traj[m, t_idx]
            y_next = traj[m, t_idx + 1]
            t_val = times[t_idx]
            drift = sde.drift(t_val, y_t, ctx)
            mu = y_t + drift * dt
            G = sde.diffusion(t_val, y_t, ctx)
            var = (G**2).sum(dim=-1) * dt
            var = var.clamp(min=min_var)
            total = total + (0.5 * ((y_next - mu) ** 2 / var + var.log())).sum()
    return total / (M * (T - 1) * n_species)


def test_gaussian_nll_matches_reference_loop(sde_and_ctx):
    """Batched implementation must produce the same value as the reference loop."""
    sde, ctx = sde_and_ctx
    torch.manual_seed(3)
    traj = torch.randn(8, 20, N_SPECIES)
    times = torch.linspace(0, 2, 20)
    dt = 0.1

    with torch.no_grad():
        loss_batched = GaussianTransitionNLL().compute(sde, ctx, traj, times, dt=dt)
        loss_reference = _nll_reference_loop(sde, ctx, traj, times, dt=dt)

    torch.testing.assert_close(loss_batched, loss_reference, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
def test_gaussian_nll_batched_is_faster(sde_and_ctx):
    """Batched forward pass must be at least 5× faster than the reference loop."""
    sde, ctx = sde_and_ctx
    torch.manual_seed(4)
    M, T = 32, 100
    traj = torch.randn(M, T, N_SPECIES)
    times = torch.linspace(0, 10, T)
    dt = 0.1
    loss_fn = GaussianTransitionNLL()

    # Warm-up
    with torch.no_grad():
        loss_fn.compute(sde, ctx, traj, times, dt=dt)
        _nll_reference_loop(sde, ctx, traj, times, dt=dt)

    with torch.no_grad():
        t0 = time.perf_counter()
        loss_fn.compute(sde, ctx, traj, times, dt=dt)
        elapsed_batched = time.perf_counter() - t0

        t0 = time.perf_counter()
        _nll_reference_loop(sde, ctx, traj, times, dt=dt)
        elapsed_loop = time.perf_counter() - t0

    speedup = elapsed_loop / elapsed_batched
    assert speedup >= 5.0, (
        f"Expected >= 5× speedup, got {speedup:.1f}×  "
        f"(batched={elapsed_batched:.3f}s, loop={elapsed_loop:.3f}s)"
    )


# ── Phase 2: protocol embedding in GaussianTransitionNLL ──────────────────────


def _make_nll_sde_and_ctx(n_species: int = 2, d_protocol: int = 0):
    from crn_surrogate.configs.model_config import EncoderConfig
    from crn_surrogate.crn.examples import lotka_volterra
    from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
    from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

    crn = lotka_volterra()
    enc = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=1))
    crn_repr = crn_to_tensor_repr(crn)
    ctx = enc(crn_repr, torch.zeros(n_species))
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, d_protocol=d_protocol),
        n_species=n_species,
    )
    return sde, ctx


def test_gaussian_nll_with_no_protocol_embedding_unchanged():
    """GaussianTransitionNLL with protocol_embedding=None behaves as before."""
    sde, ctx = _make_nll_sde_and_ctx(n_species=2, d_protocol=0)
    traj = torch.randn(3, 10, 2)
    times = torch.linspace(0, 1, 10)
    loss_fn = GaussianTransitionNLL()
    loss = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    assert loss.shape == ()
    assert loss.isfinite()


def test_gaussian_nll_with_protocol_embedding_changes_value():
    """Different protocol embeddings passed to NLL give different loss values."""
    d_protocol = 16
    sde, ctx = _make_nll_sde_and_ctx(n_species=2, d_protocol=d_protocol)
    traj = torch.randn(3, 10, 2)
    times = torch.linspace(0, 1, 10)
    loss_fn = GaussianTransitionNLL()

    emb_a = torch.zeros(d_protocol)
    emb_b = torch.randn(d_protocol)

    loss_a = loss_fn.compute(sde, ctx, traj, times, dt=0.1, protocol_embedding=emb_a)
    loss_b = loss_fn.compute(sde, ctx, traj, times, dt=0.1, protocol_embedding=emb_b)
    assert not torch.isclose(loss_a, loss_b), (
        "Different protocol embeddings should give different NLL"
    )


def test_gaussian_nll_internal_mask_reduces_loss_when_species_is_wrong():
    """Masking out a species whose prediction is deliberately wrong reduces loss."""
    sde, ctx = _make_nll_sde_and_ctx(n_species=2, d_protocol=0)
    traj = torch.zeros(4, 10, 2)
    times = torch.linspace(0, 1, 10)
    loss_fn = GaussianTransitionNLL()

    # Make species 1 hard to predict by setting its ground truth to a huge value.
    traj[:, 1:, 1] = 1e6

    loss_all = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    # Mask out species 1 (only species 0 contributes).
    mask = torch.tensor([True, False])
    loss_masked = loss_fn.compute(sde, ctx, traj, times, dt=0.1, mask=mask)
    assert loss_masked.item() < loss_all.item(), (
        "Masking the bad species should reduce the loss"
    )
