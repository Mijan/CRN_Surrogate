"""Tests for TrajectoryItem, CRNTrajectoryDataset, and CRNCollator."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset, TrajectoryItem
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr


def _birth_death_repr():
    crn = CRN(
        reactions=[
            Reaction(torch.tensor([1.0]), constant_rate(2.0), name="birth"),
            Reaction(
                torch.tensor([-1.0]),
                mass_action(0.5, torch.tensor([1.0])),
                name="death",
            ),
        ]
    )
    return crn_to_tensor_repr(crn)


def _two_species_repr():
    crn = CRN(
        reactions=[
            Reaction(
                torch.tensor([-1.0, 1.0]),
                mass_action(0.5, torch.tensor([1.0, 0.0])),
                name="a_to_b",
            ),
            Reaction(
                torch.tensor([1.0, -1.0]),
                mass_action(0.3, torch.tensor([0.0, 1.0])),
                name="b_to_a",
            ),
        ]
    )
    return crn_to_tensor_repr(crn)


def _make_item(crn_repr, M: int = 3, T: int = 20) -> TrajectoryItem:
    ns = crn_repr.n_species
    return TrajectoryItem(
        crn_repr=crn_repr,
        initial_state=torch.ones(ns),
        trajectories=torch.rand(M, T, ns),
        times=torch.linspace(0.0, 10.0, T),
    )


# ── TrajectoryItem ────────────────────────────────────────────────────────────


def test_construction() -> None:
    repr_ = _birth_death_repr()
    item = _make_item(repr_)
    assert item.crn_repr is repr_
    assert item.initial_state.shape == (1,)
    assert item.trajectories.shape == (3, 20, 1)


def test_cluster_id_mutable() -> None:
    item = _make_item(_birth_death_repr())
    assert item.cluster_id == -1
    item.cluster_id = 5
    assert item.cluster_id == 5


# ── CRNTrajectoryDataset ──────────────────────────────────────────────────────


def test_len() -> None:
    items = [_make_item(_birth_death_repr()) for _ in range(5)]
    ds = CRNTrajectoryDataset(items)
    assert len(ds) == 5


def test_getitem() -> None:
    items = [_make_item(_birth_death_repr()) for _ in range(3)]
    ds = CRNTrajectoryDataset(items)
    assert ds[0] is items[0]


# ── CRNCollator ───────────────────────────────────────────────────────────────


def test_collate_single_item() -> None:
    repr_ = _birth_death_repr()
    item = _make_item(repr_, M=3, T=20)
    collator = CRNCollator()
    batch = collator([item])

    assert "stoichiometry" in batch
    assert "trajectories" in batch
    assert batch["stoichiometry"].shape == (1, repr_.n_reactions, repr_.n_species)
    assert batch["trajectories"].shape == (1, 3, 20, repr_.n_species)


def test_collate_mixed_sizes() -> None:
    repr1 = _birth_death_repr()  # 1 species, 2 reactions
    repr2 = _two_species_repr()  # 2 species, 2 reactions
    item1 = _make_item(repr1, M=2, T=10)
    item2 = _make_item(repr2, M=2, T=10)
    collator = CRNCollator()
    batch = collator([item1, item2])

    B, max_rxn, max_spe = batch["stoichiometry"].shape
    assert B == 2
    assert max_spe == 2  # padded to max species
    assert max_rxn == 2  # both have 2 reactions

    # species_mask: True for real species, False for padding
    assert batch["species_mask"][0, 0].item() is True  # item1 has species 0
    assert (
        batch["species_mask"][0, 1].item() is False
    )  # item1 has no species 1 (padded)
    assert batch["species_mask"][1, 0].item() is True
    assert batch["species_mask"][1, 1].item() is True


def test_collate_with_n_species_sde() -> None:
    repr1 = _birth_death_repr()  # 1 species
    repr2 = _two_species_repr()  # 2 species
    item1 = _make_item(repr1, M=2, T=5)
    item2 = _make_item(repr2, M=2, T=5)
    collator = CRNCollator(n_species_sde=5)
    batch = collator([item1, item2])

    assert batch["trajectories"].shape == (2, 2, 5, 5)
    assert batch["species_mask"].shape == (2, 5)


def test_collate_n_species_sde_too_small() -> None:
    repr_ = _two_species_repr()  # 2 species
    item = _make_item(repr_, M=2, T=5)
    collator = CRNCollator(n_species_sde=1)
    with pytest.raises(ValueError):
        collator([item])


def test_bipartite_edges_prebuilt() -> None:
    repr_ = _birth_death_repr()
    item = _make_item(repr_)
    collator = CRNCollator()
    batch = collator([item])
    for crn_repr in batch["crn_reprs"]:
        assert hasattr(crn_repr, "_cached_edges")
