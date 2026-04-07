"""Tests for ODE pre-screening."""

import torch

from crn_surrogate.data.generation.configs import ODEPreScreenConfig
from crn_surrogate.data.generation.ode_prescreen import DynamicsType, ODEPreScreen
from crn_surrogate.data.generation.reference_crns import birth_death, lotka_volterra


class TestODEPreScreen:
    def setup_method(self):
        self.prescreen = ODEPreScreen(ODEPreScreenConfig(t_max=20.0))

    def test_birth_death_sustained(self):
        """Birth-death with balanced rates should be accepted."""
        crn = birth_death(k_birth=5.0, k_death=0.5)
        init = torch.tensor([10.0])
        result = self.prescreen.check(crn, init)
        assert result.accepted
        assert result.dynamics_type != DynamicsType.DECAY_TO_ZERO

    def test_pure_degradation_rejected(self):
        """CRN with only degradation should be rejected."""
        crn = birth_death(k_birth=0.001, k_death=5.0)
        init = torch.tensor([10.0])
        result = self.prescreen.check(crn, init)
        assert not result.accepted
        assert result.dynamics_type == DynamicsType.DECAY_TO_ZERO

    def test_lotka_volterra_accepted(self):
        """Lotka-Volterra should be accepted (oscillatory/sustained)."""
        crn = lotka_volterra(k_prey_birth=1.0, k_predation=0.01, k_predator_death=0.5)
        init = torch.tensor([50.0, 50.0])
        result = self.prescreen.check(crn, init)
        assert result.accepted

    def test_blowup_rejected(self):
        """CRN with strong production and weak degradation should be rejected as blowup."""
        crn = birth_death(k_birth=100.0, k_death=0.001)
        init = torch.tensor([10.0])
        result = self.prescreen.check(crn, init)
        assert not result.accepted
