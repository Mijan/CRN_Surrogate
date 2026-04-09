"""Shared fixtures for simulator tests."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.encoder.bipartite_gnn import CRNContext


@pytest.fixture
def small_sde_config() -> SDEConfig:
    """Minimal SDEConfig for fast tests."""
    return SDEConfig(
        d_model=16,
        d_hidden=32,
        n_noise_channels=4,
        n_hidden_layers=1,
        clip_state=True,
        d_protocol=0,
    )


@pytest.fixture
def solver_config() -> SolverConfig:
    return SolverConfig(clip_state=True)


@pytest.fixture
def solver_config_no_clip() -> SolverConfig:
    return SolverConfig(clip_state=False)


def make_fake_context(
    d_model: int, n_species: int = 2, n_reactions: int = 3
) -> CRNContext:
    """Build a CRNContext with random embeddings for testing."""
    return CRNContext(
        species_embeddings=torch.randn(n_species, d_model),
        reaction_embeddings=torch.randn(n_reactions, d_model),
        context_vector=torch.randn(2 * d_model),
    )
