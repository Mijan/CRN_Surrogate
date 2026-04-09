"""Shared fixtures for encoder tests."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import EncoderConfig
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr, crn_to_tensor_repr


@pytest.fixture
def birth_death_crn() -> CRN:
    """Birth-death: ∅ -> X (const rate 2.0), X -> ∅ (mass action k=0.5)."""
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1.0]),
                propensity=constant_rate(2.0),
                name="birth",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1.0]),
                propensity=mass_action(0.5, torch.tensor([1.0])),
                name="death",
            ),
        ]
    )


@pytest.fixture
def two_species_crn() -> CRN:
    """A -> B (mass action k=0.5), B -> A (mass action k=0.3)."""
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([-1.0, 1.0]),
                propensity=mass_action(0.5, torch.tensor([1.0, 0.0])),
                name="a_to_b",
            ),
            Reaction(
                stoichiometry=torch.tensor([1.0, -1.0]),
                propensity=mass_action(0.3, torch.tensor([0.0, 1.0])),
                name="b_to_a",
            ),
        ]
    )


@pytest.fixture
def birth_death_repr(birth_death_crn: CRN) -> CRNTensorRepr:
    return crn_to_tensor_repr(birth_death_crn)


@pytest.fixture
def two_species_repr(two_species_crn: CRN) -> CRNTensorRepr:
    return crn_to_tensor_repr(two_species_crn)


@pytest.fixture
def small_encoder_config() -> EncoderConfig:
    """Small config for fast tests."""
    return EncoderConfig(d_model=32, n_layers=2, use_attention=False)


@pytest.fixture
def small_attention_config() -> EncoderConfig:
    """Small config with attention for testing the attentive path."""
    return EncoderConfig(d_model=32, n_layers=2, use_attention=True)
