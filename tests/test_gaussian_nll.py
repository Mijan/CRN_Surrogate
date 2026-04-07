"""Tests for GaussianTransitionNLL.

Covers:
- Shape test: output is a scalar for various input shapes.
- Min variance test: zero diffusion output does not produce NaN.
- Gradient direction (variance): gradient pushes σ² up when too small, down when too large.
- Gradient direction (mean): gradient pushes drift toward the correct value.
- Analytical test: NLL is lower at true parameters than at perturbed parameters.
- Scheduled sampling: _effective_mode returns correct mode across epoch ranges.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.configs.training_config import TrainingConfig, TrainingMode
from crn_surrogate.data.generation.reference_crns import birth_death
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.training.losses import GaussianTransitionNLL
from crn_surrogate.training.trainer import Trainer

# ── Fixtures ──────────────────────────────────────────────────────────────────


class ConstantDriftDiffusion(nn.Module):
    """Minimal SDE with learnable constant drift and diffusion.

    drift(t, y) = mu_param  (same vector for all y, t)
    diffusion(t, y) = diag(sigma_param)  (diagonal, all y, t)
    """

    def __init__(self, n_species: int, n_noise: int) -> None:
        super().__init__()
        self._mu = nn.Parameter(torch.zeros(n_species))
        self._log_sigma = nn.Parameter(torch.zeros(n_species))
        self._n_noise = n_noise
        self._n_species = n_species

    def drift(
        self, t: torch.Tensor, state: torch.Tensor, crn_context: CRNContext
    ) -> torch.Tensor:
        return self._mu

    def diffusion(
        self, t: torch.Tensor, state: torch.Tensor, crn_context: CRNContext
    ) -> torch.Tensor:
        # Return (n_species, n_noise) with first column = exp(log_sigma), rest zero
        G = torch.zeros(self._n_species, self._n_noise, device=self._log_sigma.device)
        G[:, 0] = self._log_sigma.exp()
        return G


def _make_context() -> CRNContext:
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=1))
    crn_repr = crn_to_tensor_repr(birth_death())
    with torch.no_grad():
        return encoder(crn_repr)


def _make_trajectory(T: int = 10, n_species: int = 1) -> torch.Tensor:
    """(T, n_species) simple ramp trajectory."""
    return torch.linspace(1.0, 2.0, T).unsqueeze(-1).expand(T, n_species).clone()


# ── Shape tests ───────────────────────────────────────────────────────────────


def test_gaussian_nll_scalar_output_single_trajectory():
    """Output is a scalar for a single (T, n_species) trajectory."""
    loss_fn = GaussianTransitionNLL()
    sde = ConstantDriftDiffusion(n_species=1, n_noise=2)
    ctx = _make_context()
    traj = _make_trajectory(T=5, n_species=1)
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    assert result.shape == ()


def test_gaussian_nll_scalar_output_multiple_trajectories():
    """Output is a scalar for (M, T, n_species) multiple trajectories."""
    loss_fn = GaussianTransitionNLL()
    sde = ConstantDriftDiffusion(n_species=1, n_noise=2)
    ctx = _make_context()
    traj = _make_trajectory(T=5, n_species=1).unsqueeze(0).expand(3, -1, -1).clone()
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    assert result.shape == ()


def test_gaussian_nll_scalar_output_with_mask():
    """Output is a scalar when a species mask is applied."""
    loss_fn = GaussianTransitionNLL()
    sde = ConstantDriftDiffusion(n_species=2, n_noise=2)
    ctx = _make_context()
    traj = torch.ones(5, 2)
    times = torch.linspace(0.0, 1.0, 5)
    mask = torch.tensor([True, False])

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1, mask=mask)
    assert result.shape == ()


# ── Min variance test ─────────────────────────────────────────────────────────


def test_gaussian_nll_no_nan_when_diffusion_is_zero():
    """min_variance floor prevents NaN when G_θ outputs zeros."""
    loss_fn = GaussianTransitionNLL(min_variance=1e-6)

    class ZeroDiffusionSDE(nn.Module):
        def drift(self, t, state, ctx):
            return torch.zeros_like(state)

        def diffusion(self, t, state, ctx):
            return torch.zeros(state.shape[0], 2)

    sde = ZeroDiffusionSDE()
    ctx = _make_context()
    traj = _make_trajectory(T=5, n_species=1)
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    assert torch.isfinite(result), f"Expected finite NLL, got {result}"


# ── Gradient direction tests ──────────────────────────────────────────────────


def test_gaussian_nll_gradient_increases_variance_when_too_small():
    """When predicted σ² is too small, gradient should push it up (log_sigma up)."""
    # Use low min_variance so the small sigma is not clamped; this test is checking
    # gradient direction, not the min_variance default.
    loss_fn = GaussianTransitionNLL(min_variance=1e-6)
    # Use small sigma: exp(log_sigma) = exp(-3) ≈ 0.05
    sde = ConstantDriftDiffusion(n_species=1, n_noise=1)
    with torch.no_grad():
        sde._log_sigma.fill_(-3.0)

    ctx = _make_context()
    # Trajectory with variance ~1; sigma=0.05 is far too small
    traj = torch.tensor([[0.0], [1.0], [0.5], [1.5], [1.0]])
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    result.backward()

    # Gradient of log_sigma should be negative → increasing log_sigma reduces loss
    assert sde._log_sigma.grad is not None
    assert sde._log_sigma.grad.item() < 0, (
        "Expected negative gradient on log_sigma (loss decreases as σ grows)"
    )


def test_gaussian_nll_gradient_decreases_variance_when_too_large():
    """When predicted σ² is too large, gradient should push it down (log_sigma down)."""
    loss_fn = GaussianTransitionNLL()
    # Use very large sigma: exp(log_sigma) = exp(5) ≈ 150
    sde = ConstantDriftDiffusion(n_species=1, n_noise=1)
    with torch.no_grad():
        sde._log_sigma.fill_(5.0)

    ctx = _make_context()
    # Trajectory with small transitions ~0.1; sigma=150 is way too large
    traj = torch.tensor([[1.0], [1.1], [1.0], [1.1], [1.0]])
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    result.backward()

    assert sde._log_sigma.grad is not None
    assert sde._log_sigma.grad.item() > 0, (
        "Expected positive gradient on log_sigma (loss decreases as σ shrinks)"
    )


def test_gaussian_nll_gradient_corrects_biased_drift():
    """When drift is systematically too high, gradient should push it down."""
    loss_fn = GaussianTransitionNLL()
    sde = ConstantDriftDiffusion(n_species=1, n_noise=1)
    # mu is 1.0 (too high); true transitions are ~0.0
    with torch.no_grad():
        sde._mu.fill_(1.0)
        sde._log_sigma.fill_(0.0)  # sigma=1, reasonable

    ctx = _make_context()
    # Flat trajectory — true drift should be ~0
    traj = torch.ones(5, 1)
    times = torch.linspace(0.0, 1.0, 5)

    result = loss_fn.compute(sde, ctx, traj, times, dt=0.1)
    result.backward()

    assert sde._mu.grad is not None
    # mu=1 is too high; gradient should be positive (loss decreases as mu goes down)
    assert sde._mu.grad.item() > 0, (
        "Expected positive gradient on mu when drift is too high (mu > 0, true drift ≈ 0)"
    )


# ── Analytical test ───────────────────────────────────────────────────────────


def test_gaussian_nll_lower_at_true_parameters_than_perturbed():
    """NLL at the true drift/diffusion parameters must be lower than at perturbed ones."""
    loss_fn = GaussianTransitionNLL()
    n_species = 1
    dt = 0.1
    T = 20
    times = torch.linspace(0.0, (T - 1) * dt, T)

    # True parameters: drift=0.1, sigma=0.5
    true_mu = torch.tensor([0.1])
    true_sigma = torch.tensor([0.5])

    # Generate a trajectory consistent with these parameters
    torch.manual_seed(42)
    traj = torch.zeros(T, n_species)
    traj[0] = 1.0
    for t in range(T - 1):
        noise = true_sigma * torch.randn(n_species) * (dt**0.5)
        traj[t + 1] = traj[t] + true_mu * dt + noise

    class FixedSDE(nn.Module):
        def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
            super().__init__()
            self._mu = mu
            self._sigma = sigma

        def drift(self, t, state, ctx):
            return self._mu

        def diffusion(self, t, state, ctx):
            G = torch.zeros(n_species, 1)
            G[:, 0] = self._sigma
            return G

    ctx = _make_context()
    sde_true = FixedSDE(true_mu, true_sigma)
    sde_perturbed = FixedSDE(true_mu * 3.0, true_sigma * 5.0)

    with torch.no_grad():
        nll_true = loss_fn.compute(sde_true, ctx, traj, times, dt=dt)
        nll_perturbed = loss_fn.compute(sde_perturbed, ctx, traj, times, dt=dt)

    assert nll_true.item() < nll_perturbed.item(), (
        f"NLL at true params ({nll_true:.4f}) should be lower than at "
        f"perturbed params ({nll_perturbed:.4f})"
    )


# ── Scheduled sampling tests ──────────────────────────────────────────────────


def _make_trainer(training_mode: TrainingMode) -> Trainer:
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=16, n_layers=1))
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=16, n_noise_channels=2), n_species=1
    )
    from crn_surrogate.configs.model_config import ModelConfig

    model_config = ModelConfig(
        encoder=EncoderConfig(d_model=16, n_layers=1),
        sde=SDEConfig(d_model=16, d_hidden=16, n_noise_channels=2),
    )
    train_config = TrainingConfig(
        training_mode=training_mode,
        scheduled_sampling_start_epoch=10,
        scheduled_sampling_end_epoch=20,
        max_epochs=30,
        use_wandb=False,
    )
    return Trainer(encoder, sde, model_config, train_config)


def test_effective_mode_teacher_forcing_always_returns_teacher_forcing():
    """TEACHER_FORCING mode always returns TEACHER_FORCING regardless of epoch."""
    trainer = _make_trainer(TrainingMode.TEACHER_FORCING)
    for epoch in [1, 10, 50, 100]:
        assert trainer._effective_mode(epoch) == TrainingMode.TEACHER_FORCING


def test_effective_mode_full_rollout_always_returns_full_rollout():
    """FULL_ROLLOUT mode always returns FULL_ROLLOUT regardless of epoch."""
    trainer = _make_trainer(TrainingMode.FULL_ROLLOUT)
    for epoch in [1, 10, 50, 100]:
        assert trainer._effective_mode(epoch) == TrainingMode.FULL_ROLLOUT


def test_effective_mode_scheduled_sampling_before_start_is_teacher_forcing():
    """SCHEDULED_SAMPLING before start_epoch must return TEACHER_FORCING."""
    trainer = _make_trainer(TrainingMode.SCHEDULED_SAMPLING)
    # start_epoch=10; epoch 9 should always be teacher forcing
    for _ in range(20):
        assert trainer._effective_mode(9) == TrainingMode.TEACHER_FORCING


def test_effective_mode_scheduled_sampling_after_end_is_full_rollout():
    """SCHEDULED_SAMPLING after end_epoch must return FULL_ROLLOUT."""
    trainer = _make_trainer(TrainingMode.SCHEDULED_SAMPLING)
    # end_epoch=20; epoch 20+ should always be full rollout
    for _ in range(20):
        assert trainer._effective_mode(20) == TrainingMode.FULL_ROLLOUT


def test_effective_mode_scheduled_sampling_between_returns_both():
    """SCHEDULED_SAMPLING between start and end epoch must return both modes."""
    trainer = _make_trainer(TrainingMode.SCHEDULED_SAMPLING)
    # epoch 15 is halfway between 10 and 20
    results = {trainer._effective_mode(15) for _ in range(100)}
    assert TrainingMode.TEACHER_FORCING in results, (
        "Expected some TEACHER_FORCING samples"
    )
    assert TrainingMode.FULL_ROLLOUT in results, "Expected some FULL_ROLLOUT samples"
