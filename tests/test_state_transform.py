"""Tests for state space transforms."""

import torch

from crn_surrogate.simulator.state_transform import (
    Log1pTransform,
    StateTransform,
    get_state_transform,
)


class TestStateTransform:
    def test_identity_roundtrip(self):
        t = StateTransform()
        x = torch.tensor([0.0, 1.0, 10.0, 100.0])
        assert torch.allclose(t.inverse(t.forward(x)), x)

    def test_log1p_roundtrip(self):
        t = Log1pTransform()
        x = torch.tensor([0.0, 1.0, 10.0, 100.0])
        assert torch.allclose(t.inverse(t.forward(x)), x, atol=1e-5)

    def test_log1p_zero(self):
        t = Log1pTransform()
        assert t.forward(torch.tensor([0.0])).item() == 0.0
        assert t.inverse(torch.tensor([0.0])).item() == 0.0

    def test_log1p_negative_clamped(self):
        t = Log1pTransform()
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0])
        z = t.forward(x)
        assert (z >= 0).all()

    def test_log1p_trajectory(self):
        t = Log1pTransform()
        traj = torch.rand(8, 50, 3) * 100  # (M, T, n_species)
        roundtrip = t.inverse_trajectory(t.transform_trajectory(traj))
        assert torch.allclose(roundtrip, traj, atol=1e-4)

    def test_factory(self):
        assert isinstance(get_state_transform(False), StateTransform)
        assert isinstance(get_state_transform(True), Log1pTransform)

    def test_log1p_gradient_flows(self):
        t = Log1pTransform()
        x = torch.tensor([1.0, 10.0, 100.0], requires_grad=True)
        z = t.forward(x)
        z.sum().backward()
        assert x.grad is not None
        # Gradient of log1p(x) = 1/(1+x), so larger x gets smaller gradient
        # This is exactly the rebalancing we want
        assert x.grad[0] > x.grad[1] > x.grad[2]
