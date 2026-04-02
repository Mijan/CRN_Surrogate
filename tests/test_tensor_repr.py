"""Tests for CRNTensorRepr, including the .to(device) method."""

import torch

from crn_surrogate.crn.examples import birth_death
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr


def test_tensor_repr_to_same_device_returns_self():
    """to() on same device returns the same object (no copy)."""
    crn = birth_death(k_birth=1.0, k_death=0.5)
    repr_cpu = crn_to_tensor_repr(crn)
    same = repr_cpu.to(torch.device("cpu"))
    assert same is repr_cpu


def test_tensor_repr_to_moves_all_tensors():
    """to() produces a new repr with all tensors on the target device."""
    crn = birth_death(k_birth=1.0, k_death=0.5)
    repr_cpu = crn_to_tensor_repr(crn)

    result = repr_cpu.to(torch.device("cpu"))
    assert result.stoichiometry.device == torch.device("cpu")
    assert result.n_species == repr_cpu.n_species
    assert result.n_reactions == repr_cpu.n_reactions
    edges = result.bipartite_edges
    assert edges.rxn_to_species_index.device == torch.device("cpu")


@torch.no_grad()
def test_tensor_repr_to_preserves_values():
    """to() does not alter tensor values, only device."""
    crn = birth_death(k_birth=1.0, k_death=0.5)
    repr_cpu = crn_to_tensor_repr(crn)
    result = repr_cpu.to(torch.device("cpu"))
    torch.testing.assert_close(result.stoichiometry, repr_cpu.stoichiometry)
    torch.testing.assert_close(result.propensity_params, repr_cpu.propensity_params)
