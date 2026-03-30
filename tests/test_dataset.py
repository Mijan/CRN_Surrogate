"""Tests for CRNTrajectoryDataset, TrajectoryItem, and CRNCollator.

Covers:
- TrajectoryItem holds the expected tensor shapes.
- CRNTrajectoryDataset supports len() and indexing.
- CRNCollator produces correctly shaped, padded output for homogeneous
  and heterogeneous batches, and sets masks correctly.
"""

import pytest
import torch

from crn_surrogate.crn.examples import birth_death, lotka_volterra
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_item(crn, M: int = 4, T: int = 10) -> TrajectoryItem:
    """Build a TrajectoryItem with random trajectory data for the given CRN."""
    n = crn.n_species
    return TrajectoryItem(
        crn_repr=crn_to_tensor_repr(crn),
        initial_state=torch.zeros(n),
        trajectories=torch.rand(M, T, n),
        times=torch.linspace(0.0, 5.0, T),
    )


# ── TrajectoryItem ────────────────────────────────────────────────────────────


def test_trajectory_item_trajectory_tensor_is_3d():
    """TrajectoryItem.trajectories must be (M, T, n_species)."""
    item = _make_item(birth_death(), M=4, T=10)
    assert item.trajectories.ndim == 3
    assert item.trajectories.shape == (4, 10, 1)


def test_trajectory_item_times_tensor_is_1d_with_length_t():
    """TrajectoryItem.times must be a 1-D tensor of length T."""
    item = _make_item(birth_death(), T=15)
    assert item.times.ndim == 1
    assert item.times.shape[0] == 15


def test_trajectory_item_crn_repr_has_correct_dimensions():
    """TrajectoryItem.crn_repr has the expected n_species and n_reactions."""
    item = _make_item(birth_death())
    assert item.crn_repr.n_species == 1
    assert item.crn_repr.n_reactions == 2


# ── CRNTrajectoryDataset ──────────────────────────────────────────────────────


def test_dataset_length_matches_number_of_items():
    items = [_make_item(birth_death()) for _ in range(7)]
    dataset = CRNTrajectoryDataset(items)
    assert len(dataset) == 7


def test_dataset_getitem_returns_same_trajectory_item():
    item = _make_item(birth_death())
    dataset = CRNTrajectoryDataset([item])
    assert dataset[0] is item


# ── CRNCollator — homogeneous batch ──────────────────────────────────────────


def test_collator_stoichiometry_shape_for_uniform_batch():
    """Batch of B identical CRNs yields stoichiometry (B, n_rxn, n_species)."""
    batch = [_make_item(birth_death()) for _ in range(3)]
    result = CRNCollator()(batch)
    assert result["stoichiometry"].shape == (3, 2, 1)


def test_collator_trajectories_shape_is_4d():
    """Collated trajectories must be 4-D: (B, M, T, max_species)."""
    batch = [_make_item(birth_death(), M=4, T=10) for _ in range(2)]
    result = CRNCollator()(batch)
    assert result["trajectories"].ndim == 4
    assert result["trajectories"].shape == (2, 4, 10, 1)


def test_collator_times_shape():
    """Collated times are (B, T)."""
    batch = [_make_item(birth_death(), T=8) for _ in range(3)]
    result = CRNCollator()(batch)
    assert result["times"].shape == (3, 8)


def test_collator_species_mask_all_true_for_uniform_batch():
    """With a uniform batch, all species are valid and species_mask is all True."""
    batch = [_make_item(birth_death()) for _ in range(4)]
    result = CRNCollator()(batch)
    assert result["species_mask"].all()


def test_collator_reaction_mask_all_true_for_uniform_batch():
    """With a uniform batch, all reactions are valid and reaction_mask is all True."""
    batch = [_make_item(birth_death()) for _ in range(4)]
    result = CRNCollator()(batch)
    assert result["reaction_mask"].all()


# ── CRNCollator — heterogeneous batch (padding) ───────────────────────────────


def test_collator_pads_stoichiometry_to_max_species_and_reactions():
    """Mixing birth-death (1 species, 2 rxns) and Lotka-Volterra (2 species, 3 rxns)
    yields stoichiometry padded to (B=2, max_rxn=3, max_species=2)."""
    batch = [_make_item(birth_death()), _make_item(lotka_volterra())]
    result = CRNCollator()(batch)
    assert result["stoichiometry"].shape == (2, 3, 2)


def test_collator_species_mask_marks_padded_species_as_false():
    """Birth-death has 1 species; when batched with Lotka-Volterra (2 species),
    the second species column must be masked False for the birth-death item."""
    batch = [_make_item(birth_death()), _make_item(lotka_volterra())]
    result = CRNCollator()(batch)
    species_mask = result["species_mask"]  # (B=2, max_species=2)
    assert species_mask[0, 0].item() is True
    assert species_mask[0, 1].item() is False
    assert species_mask[1, 0].item() is True
    assert species_mask[1, 1].item() is True


def test_collator_reaction_mask_marks_padded_reactions_as_false():
    """Birth-death has 2 reactions; when batched with Lotka-Volterra (3 reactions),
    the third reaction row must be masked False for the birth-death item."""
    batch = [_make_item(birth_death()), _make_item(lotka_volterra())]
    result = CRNCollator()(batch)
    reaction_mask = result["reaction_mask"]  # (B=2, max_rxn=3)
    assert reaction_mask[0, 0].item() is True
    assert reaction_mask[0, 1].item() is True
    assert reaction_mask[0, 2].item() is False  # padded reaction


def test_collator_initial_states_padded_with_zeros():
    """Padded species in initial_states must be zero."""
    batch = [_make_item(birth_death()), _make_item(lotka_volterra())]
    result = CRNCollator()(batch)
    init = result["initial_states"]  # (B=2, max_species=2)
    assert init[0, 1].item() == pytest.approx(0.0)
