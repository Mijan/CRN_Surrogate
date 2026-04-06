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
        scale: (n_species,) per-species normalization scale precomputed from
            ``trajectories``. When None the Trainer computes it on the fly from
            the trajectories. Storing it avoids repeated recomputation across epochs.
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
    scale: torch.Tensor | None = None  # (n_species,) normalization scale


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

    When ``n_species_sde`` is provided, the species dimension of trajectories,
    initial_states, and species_mask is padded to ``n_species_sde`` (the SDE's
    fixed state dimension) instead of the per-batch maximum. This lets
    ``_prepare_item`` skip all per-item padding tensor allocations.

    Note: the return type is a plain ``dict[str, Tensor]``. This is the one
    place in the codebase where string-keyed dicts are used as an API boundary.
    It exists because PyTorch's DataLoader requires a dict-based collation
    interface; typed alternatives would require a custom collate wrapper that
    adds more complexity than value here.
    """

    def __init__(self, n_species_sde: int | None = None) -> None:
        """Args:
        n_species_sde: When provided, pad the species dimension of trajectories,
            initial_states, and species_mask to this size. Must be >= the
            maximum n_species in any batch. When None, pads to the batch max
            (original behaviour).
        """
        self._n_species_sde = n_species_sde

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
              initial_states:          (B, n_species_pad)
              trajectories:            (B, M, T, n_species_pad)
              times:                   (B, T)
              species_mask:            (B, n_species_pad) bool, True = valid
              reaction_mask:           (B, max_rxn) bool, True = valid
              cluster_ids:             (B,) int, -1 if unassigned
              input_protocols:         list[InputProtocol] length B
              internal_species_mask:   (B, max_species) bool or None
              scales:                  (B, max_species) float or None
              crn_reprs:               list[CRNTensorRepr] length B, edges pre-built on CPU
              n_species_per_item:      list[int] length B
              n_reactions_per_item:    list[int] length B
            where n_species_pad = n_species_sde if provided, else max_species.
        """
        # Per-item integer counts — derived from source items (no tensor .item() calls)
        n_species_per_item: list[int] = [item.crn_repr.n_species for item in batch]
        n_reactions_per_item: list[int] = [item.crn_repr.n_reactions for item in batch]

        max_species = max(n_species_per_item)
        max_rxn = max(n_reactions_per_item)
        max_params = max(item.crn_repr.propensity_params.shape[1] for item in batch)
        M = batch[0].trajectories.shape[0]
        T = batch[0].trajectories.shape[1]
        B = len(batch)

        # Determine species padding size for trajectories / initial states / masks
        if self._n_species_sde is not None:
            if self._n_species_sde < max_species:
                raise ValueError(
                    f"n_species_sde={self._n_species_sde} is smaller than the maximum "
                    f"n_species in this batch ({max_species}). Cannot pad to SDE size."
                )
            n_species_pad = self._n_species_sde
        else:
            n_species_pad = max_species

        stoich = torch.zeros(B, max_rxn, max_species)
        deps = torch.zeros(B, max_rxn, max_species)
        prop_params = torch.zeros(B, max_rxn, max_params)
        prop_type_ids = torch.zeros(B, max_rxn, dtype=torch.long)
        init_states = torch.zeros(B, n_species_pad)
        trajs = torch.zeros(B, M, T, n_species_pad)
        times = torch.zeros(B, T)
        species_mask = torch.zeros(B, n_species_pad, dtype=torch.bool)
        reaction_mask = torch.zeros(B, max_rxn, dtype=torch.bool)
        cluster_ids = torch.full((B,), fill_value=-1, dtype=torch.long)

        # Determine whether any item has a non-None internal_species_mask.
        has_internal_mask = any(
            item.internal_species_mask is not None for item in batch
        )
        internal_species_mask_batch = (
            torch.zeros(B, max_species, dtype=torch.bool) if has_internal_mask else None
        )

        # Determine whether any item has a precomputed scale.
        has_scale = any(item.scale is not None for item in batch)
        scales_batch = torch.ones(B, max_species) if has_scale else None

        # Pre-build bipartite edges on CPU so _prepare_item only needs cudaMemcpy
        crn_reprs: list[CRNTensorRepr] = []
        for i, item in enumerate(batch):
            ns = n_species_per_item[i]
            nr = n_reactions_per_item[i]
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

            # Trigger edge build on CPU (fast; avoids GPU kernel launch per item)
            _ = item.crn_repr.bipartite_edges
            crn_reprs.append(item.crn_repr)

            if has_internal_mask:
                if item.internal_species_mask is not None:
                    internal_species_mask_batch[i, :ns] = item.internal_species_mask  # type: ignore[index]
                else:
                    # No mask provided: treat all real species as internal.
                    internal_species_mask_batch[i, :ns] = True  # type: ignore[index]
            if has_scale:
                if item.scale is not None:
                    scales_batch[i, :ns] = item.scale  # type: ignore[index]
                else:
                    # No scale provided: use 1.0 (identity; Trainer will recompute).
                    scales_batch[i, :ns] = 1.0  # type: ignore[index]

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
            "scales": scales_batch,
            "crn_reprs": crn_reprs,
            "n_species_per_item": n_species_per_item,
            "n_reactions_per_item": n_reactions_per_item,
        }
