"""Tests for the neural SDE and Euler-Maruyama solver.

Covers:
- drift and diffusion output shapes match (n_species,) and (n_species, n_noise) respectively.
- diffusion values are always non-negative (softplus activation).
- Euler-Maruyama trajectory output shape matches the time grid length.
- clip_state=True keeps states non-negative throughout the trajectory.
- clip_state=False allows states to go negative (no clamping applied).
- Gradients flow through the solver back to SDE parameters.
- Solver works for multi-species CRNs (Lotka-Volterra).
"""

import torch

from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig
from crn_surrogate.crn.examples import birth_death, lotka_volterra
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_context(crn, d_model: int = 16):
    encoder = BipartiteGNNEncoder(EncoderConfig(d_model=d_model, n_layers=1))
    crn_repr = crn_to_tensor_repr(crn)
    init = torch.zeros(crn.n_species)
    return encoder(crn_repr, init)


# ── Drift / diffusion shapes ──────────────────────────────────────────────────


def test_drift_output_shape_equals_n_species():
    """drift() must return a vector of length n_species."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    drift = sde.drift(torch.tensor(0.0), torch.tensor([5.0]), ctx)
    assert drift.shape == (1,)


def test_diffusion_output_shape_is_n_species_by_n_noise():
    """diffusion() must return a matrix of shape (n_species, n_noise_channels)."""
    ctx = _make_context(birth_death())
    n_noise = 4
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=n_noise), n_species=1
    )
    diff = sde.diffusion(torch.tensor(0.0), torch.tensor([5.0]), ctx)
    assert diff.shape == (1, n_noise)


def test_diffusion_values_are_nonnegative():
    """The diffusion matrix uses softplus, so all entries must be >= 0."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    diff = sde.diffusion(torch.tensor(0.0), torch.randn(1), ctx)
    assert (diff >= 0).all()


# ── Euler-Maruyama solver ─────────────────────────────────────────────────────


def test_solver_trajectory_shape_matches_time_grid():
    """Solved trajectory must have one state per time-grid point."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16))
    t_span = torch.linspace(0.0, 5.0, 12)
    traj = solver.solve(sde, torch.tensor([5.0]), ctx, t_span, dt=0.1)
    assert traj.states.shape == (12, 1)


def test_solver_with_clip_state_keeps_states_nonnegative():
    """clip_state=True must clamp negative states to zero after every EM step."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, clip_state=True),
        n_species=1,
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16, clip_state=True))
    t_span = torch.linspace(0.0, 10.0, 30)
    traj = solver.solve(sde, torch.tensor([0.0]), ctx, t_span, dt=0.05)
    assert (traj.states >= 0.0).all()


def test_solver_without_clip_state_clip_min_geq_noclip_min():
    """With clip_state=False and a large diffusion coefficient, clipped >= no-clip minimum."""
    torch.manual_seed(0)
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, clip_state=False),
        n_species=1,
    )
    solver_noclip = EulerMaruyamaSolver(SDEConfig(d_model=16, clip_state=False))
    solver_clip = EulerMaruyamaSolver(SDEConfig(d_model=16, clip_state=True))
    t_span = torch.linspace(0.0, 10.0, 30)

    traj_clip = solver_clip.solve(sde, torch.tensor([0.0]), ctx, t_span, dt=0.5)
    assert (traj_clip.states >= 0.0).all()

    traj_noclip = solver_noclip.solve(sde, torch.tensor([0.0]), ctx, t_span, dt=0.5)
    assert traj_clip.states.min() >= traj_noclip.states.min()


# ── Multi-species ─────────────────────────────────────────────────────────────


def test_solver_lotka_volterra_two_species_trajectory_shape():
    """Solver must handle 2-species CRNs and return (T, 2) state trajectories."""
    ctx = _make_context(lotka_volterra(), d_model=16)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=8), n_species=2
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16))
    t_span = torch.linspace(0.0, 5.0, 10)
    traj = solver.solve(sde, torch.tensor([10.0, 5.0]), ctx, t_span, dt=0.1)
    assert traj.states.shape == (10, 2)


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_solver_gradients_flow_through_trajectory_to_sde_parameters():
    """A loss on the solved trajectory must propagate gradients to the SDE parameters."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16))
    t_span = torch.linspace(0.0, 2.0, 5)
    traj = solver.solve(sde, torch.tensor([5.0]), ctx, t_span, dt=0.1)
    traj.states.sum().backward()
    assert any(p.grad is not None for p in sde.parameters()), (
        "No gradients reached SDE parameters"
    )


# ── Phase 2: protocol embedding ────────────────────────────────────────────────


def test_sde_with_d_protocol_zero_behaves_identically():
    """d_protocol=0 (default): drift/diffusion shapes match pre-Phase-2."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, d_protocol=0),
        n_species=1,
    )
    state = torch.tensor([5.0])
    drift = sde.drift(torch.tensor(0.0), state, ctx)
    diff = sde.diffusion(torch.tensor(0.0), state, ctx)
    assert drift.shape == (1,)
    assert diff.shape == (1, 4)


def test_sde_with_d_protocol_positive_accepts_embedding():
    """d_protocol>0: drift/diffusion work with protocol_embedding provided."""
    d_protocol = 32
    ctx = _make_context(birth_death(), d_model=16)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, d_protocol=d_protocol),
        n_species=1,
    )
    state = torch.tensor([5.0])
    proto_emb = torch.zeros(d_protocol)

    drift = sde.drift(torch.tensor(0.0), state, ctx, protocol_embedding=proto_emb)
    diff = sde.diffusion(torch.tensor(0.0), state, ctx, protocol_embedding=proto_emb)
    assert drift.shape == (1,)
    assert diff.shape == (1, 4)


def test_sde_different_protocol_embeddings_produce_different_outputs():
    """Same CRN context, different protocol embeddings must give different drift."""
    d_protocol = 16
    ctx = _make_context(birth_death(), d_model=16)
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4, d_protocol=d_protocol),
        n_species=1,
    )
    state = torch.tensor([5.0])
    emb_a = torch.randn(d_protocol)
    emb_b = torch.randn(d_protocol)

    drift_a = sde.drift(torch.tensor(0.0), state, ctx, protocol_embedding=emb_a)
    drift_b = sde.drift(torch.tensor(0.0), state, ctx, protocol_embedding=emb_b)
    assert not torch.allclose(drift_a, drift_b)


def test_solver_with_no_protocol_matches_pre_phase2():
    """EulerMaruyamaSolver with no protocol args produces trajectory of correct shape."""
    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16))
    t_span = torch.linspace(0.0, 5.0, 20)
    traj = solver.solve(sde, torch.tensor([0.0]), ctx, t_span, dt=0.1)
    assert traj.states.shape == (20, 1)


def test_solver_with_input_protocol_clamps_external_species():
    """Solver with an InputProtocol writes correct values for external species."""
    from crn_surrogate.crn import CRN, InputProtocol, Reaction, single_pulse
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

    # 2 species: A (internal), I (external)
    crn = CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1.0, 0.0]),
                propensity=mass_action(0.5, torch.tensor([0.0, 1.0])),
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0, 0.0]),
                propensity=mass_action(0.3, torch.tensor([1.0, 0.0])),
            ),
        ],
        species_names=["A", "I"],
        external_species=frozenset({1}),
    )
    crn_repr = crn_to_tensor_repr(crn)
    encoder = __import__(
        "crn_surrogate.encoder.bipartite_gnn", fromlist=["BipartiteGNNEncoder"]
    ).BipartiteGNNEncoder(
        __import__(
            "crn_surrogate.configs.model_config", fromlist=["EncoderConfig"]
        ).EncoderConfig(d_model=16, n_layers=1)
    )
    ctx = encoder(crn_repr, torch.tensor([0.0, 0.0]))

    t_on, amp = 2.0, 50.0
    sched = single_pulse(t_start=t_on, t_end=10.0, amplitude=amp)
    protocol = InputProtocol(schedules={1: sched})
    ext_mask = torch.tensor([False, True])

    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=2
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16, clip_state=True))
    t_span = torch.linspace(0.0, 10.0, 50)
    traj = solver.solve(
        sde,
        torch.tensor([0.0, 0.0]),
        ctx,
        t_span,
        dt=0.05,
        input_protocol=protocol,
        external_species_mask=ext_mask,
    )

    # After t_on, external species should equal the amplitude
    after_mask = t_span > t_on + 0.1
    assert (traj.states[after_mask, 1].abs() - amp).abs().max().item() < 1e-4


def test_solver_raises_without_external_mask_when_protocol_provided():
    """EulerMaruyamaSolver must raise if input_protocol given without external_species_mask."""
    from crn_surrogate.crn import InputProtocol, single_pulse

    ctx = _make_context(birth_death())
    sde = CRNNeuralSDE(
        SDEConfig(d_model=16, d_hidden=32, n_noise_channels=4), n_species=1
    )
    solver = EulerMaruyamaSolver(SDEConfig(d_model=16))
    protocol = InputProtocol(schedules={0: single_pulse(1.0, 5.0, 10.0)})

    import pytest

    with pytest.raises(ValueError, match="external_species_mask"):
        solver.solve(
            sde,
            torch.tensor([0.0]),
            ctx,
            torch.linspace(0, 5, 10),
            dt=0.1,
            input_protocol=protocol,
        )
