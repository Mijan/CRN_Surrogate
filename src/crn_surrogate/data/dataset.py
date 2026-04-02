"""Dataset and collation utilities for CRN trajectory training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

from crn_surrogate.crn.inputs import EMPTY_PROTOCOL, InputProtocol
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr


@dataclass
class TrajectoryItem:
    """Single training example: a CRN tensor representation with M ground-truth SSA trajectories.

    This is intentionally a mutable dataclass (not frozen) because ``cluster_id``
    is assigned after construction by the pipeline's ``_assign_cluster_ids`` step.
    All other fields are set at creation and should not be mutated afterward.

    Attributes:
        crn_repr: Flat tensor representation of the CRN for the encoder.
        initial_state: (n_species,) initial molecule counts.
        trajectories: (M, T, n_species) M independent SSA trajectories on a regular
            time grid. M >= 2 is required to compute variance-matching loss.
        times: (T,) shared time grid for all M trajectories.
        motif_label: String label identifying the motif type (e.g., "birth_death").
            For composed motifs this is the descriptive task label set in
            GenerationTask (e.g., "toggle_switch+birth_death_readout").
        cluster_id: Integer cluster / class identifier assigned during dataset curation.
            Defaults to -1 (unassigned).
        params: Raw kinetic parameter dict used to generate this CRN instance.
        input_protocol: The InputProtocol used when simulating these trajectories.
            Defaults to EMPTY_PROTOCOL for CRNs with no external inputs.
        internal_species_mask: (n_species,) bool tensor; True for internal (non-external)
            species. When None, all species are treated as internal. Precomputed from
            crn.internal_species_mask at data generation time.
    """

    crn_repr: CRNTensorRepr
    initial_state: torch.Tensor  # (n_species,)
    trajectories: torch.Tensor  # (M, T, n_species)
    times: torch.Tensor  # (T,)
    motif_label: str = ""
    cluster_id: int = -1
    params: dict = field(default_factory=dict)
    input_protocol: InputProtocol = field(default_factory=lambda: EMPTY_PROTOCOL)
    internal_species_mask: torch.Tensor | None = None  # (n_species,) bool


class CRNTrajectoryDataset(Dataset):
    """Pre-generated Gillespie trajectories for multiple CRN instances.

    Each item contains a CRN tensor representation, an initial state, and M
    independent SSA trajectories on a regular time grid.
    """

    def __init__(self, items: list[TrajectoryItem] | str | Path) -> None:
        """Args:
        items: Either a list of pre-generated TrajectoryItem instances, or a
            path (str or Path) to a .pt file containing a saved list of items.
        """
        if isinstance(items, (str, Path)):
            loaded = torch.load(items, weights_only=False)
            self._items = loaded
        else:
            self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> TrajectoryItem:
        return self._items[idx]


class CRNCollator:
    """Pads stoichiometry matrices and trajectories to the max sizes in the batch.

    Returns padding masks for species and reactions so that message passing
    and loss computation can ignore padded entries.
    """

    def __call__(self, batch: list[TrajectoryItem]) -> dict:
        """Collate a list of TrajectoryItems into a padded batch dict.

        Args:
            batch: List of TrajectoryItem instances.

        Returns:
            Dict with keys:
              stoichiometry:           (B, max_rxn, max_species)
              dependency_matrix:       (B, max_rxn, max_species)
              propensity_params:       (B, max_rxn, max_params)
              propensity_type_ids:     (B, max_rxn) int
              initial_states:          (B, max_species)
              trajectories:            (B, M, T, max_species)
              times:                   (B, T)
              species_mask:            (B, max_species) bool, True = valid
              reaction_mask:           (B, max_rxn) bool, True = valid
              cluster_ids:             (B,) int, -1 if unassigned
              input_protocols:         list[InputProtocol] length B
              internal_species_mask:   (B, max_species) bool or None
        """
        max_species = max(item.crn_repr.n_species for item in batch)
        max_rxn = max(item.crn_repr.n_reactions for item in batch)
        max_params = max(item.crn_repr.propensity_params.shape[1] for item in batch)
        M = batch[0].trajectories.shape[0]
        T = batch[0].trajectories.shape[1]
        B = len(batch)

        stoich = torch.zeros(B, max_rxn, max_species)
        deps = torch.zeros(B, max_rxn, max_species)
        prop_params = torch.zeros(B, max_rxn, max_params)
        prop_type_ids = torch.zeros(B, max_rxn, dtype=torch.long)
        init_states = torch.zeros(B, max_species)
        trajs = torch.zeros(B, M, T, max_species)
        times = torch.zeros(B, T)
        species_mask = torch.zeros(B, max_species, dtype=torch.bool)
        reaction_mask = torch.zeros(B, max_rxn, dtype=torch.bool)
        cluster_ids = torch.full((B,), fill_value=-1, dtype=torch.long)

        # Determine whether any item has a non-None internal_species_mask.
        has_internal_mask = any(
            item.internal_species_mask is not None for item in batch
        )
        internal_species_mask_batch = (
            torch.zeros(B, max_species, dtype=torch.bool) if has_internal_mask else None
        )

        for i, item in enumerate(batch):
            ns = item.crn_repr.n_species
            nr = item.crn_repr.n_reactions
            np_ = item.crn_repr.propensity_params.shape[1]

            stoich[i, :nr, :ns] = item.crn_repr.stoichiometry
            deps[i, :nr, :ns] = item.crn_repr.dependency_matrix
            prop_params[i, :nr, :np_] = item.crn_repr.propensity_params
            prop_type_ids[i, :nr] = item.crn_repr.propensity_type_ids
            init_states[i, :ns] = item.initial_state
            trajs[i, :, :, :ns] = item.trajectories
            times[i] = item.times
            species_mask[i, :ns] = True
            reaction_mask[i, :nr] = True
            cluster_ids[i] = item.cluster_id
            if has_internal_mask:
                if item.internal_species_mask is not None:
                    internal_species_mask_batch[i, :ns] = item.internal_species_mask  # type: ignore[index]
                else:
                    # No mask provided: treat all real species as internal.
                    internal_species_mask_batch[i, :ns] = True  # type: ignore[index]

        return {
            "stoichiometry": stoich,
            "dependency_matrix": deps,
            "propensity_params": prop_params,
            "propensity_type_ids": prop_type_ids,
            "initial_states": init_states,
            "trajectories": trajs,
            "times": times,
            "species_mask": species_mask,
            "reaction_mask": reaction_mask,
            "cluster_ids": cluster_ids,
            "input_protocols": [item.input_protocol for item in batch],
            "internal_species_mask": internal_species_mask_batch,
        }
