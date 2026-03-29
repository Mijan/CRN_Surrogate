from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from crn_surrogate.data.crn import CRNDefinition


@dataclass
class TrajectoryItem:
    """Single training example: a CRN instance with M ground-truth SSA trajectories.

    Attributes:
        crn: The CRN definition.
        initial_state: (n_species,) initial molecule counts.
        trajectories: (M, T, n_species) M independent SSA trajectories on a regular
            time grid. M >= 2 is required to compute variance-matching loss.
        times: (T,) shared time grid for all M trajectories.
    """

    crn: CRNDefinition
    initial_state: torch.Tensor  # (n_species,)
    trajectories: torch.Tensor  # (M, T, n_species)
    times: torch.Tensor  # (T,)


class CRNTrajectoryDataset(Dataset):
    """Pre-generated Gillespie trajectories for multiple CRN instances.

    Each item contains a CRN definition, an initial state, and M independent
    SSA trajectories on a regular time grid.
    """

    def __init__(self, items: list[TrajectoryItem]) -> None:
        """Args:
        items: List of pre-generated TrajectoryItem instances.
        """
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> TrajectoryItem:
        return self._items[idx]


class CRNCollator:
    """Pads stoichiometry matrices and trajectories to the max sizes in the batch.

    Returns padding masks for species and reactions so that
    message passing and loss computation ignore padded entries.
    """

    def __call__(self, batch: list[TrajectoryItem]) -> dict:
        """Collate a list of TrajectoryItems into a padded batch dict.

        Args:
            batch: List of TrajectoryItem instances.

        Returns:
            Dict with keys:
              stoichiometry:        (B, max_rxn, max_species)
              reactant_matrix:      (B, max_rxn, max_species)
              propensity_params:    (B, max_rxn, max_params)
              propensity_type_ids:  (B, max_rxn) int
              initial_states:       (B, max_species)
              trajectories:         (B, M, T, max_species)
              times:                (B, T)
              species_mask:         (B, max_species) bool, True = valid
              reaction_mask:        (B, max_rxn) bool, True = valid
        """
        max_species = max(item.crn.n_species for item in batch)
        max_rxn = max(item.crn.n_reactions for item in batch)
        max_params = max(item.crn.propensity_params.shape[1] for item in batch)
        M = batch[0].trajectories.shape[0]
        T = batch[0].trajectories.shape[1]
        B = len(batch)

        stoich = torch.zeros(B, max_rxn, max_species)
        reactants = torch.zeros(B, max_rxn, max_species)
        prop_params = torch.zeros(B, max_rxn, max_params)
        prop_type_ids = torch.zeros(B, max_rxn, dtype=torch.long)
        init_states = torch.zeros(B, max_species)
        trajs = torch.zeros(B, M, T, max_species)
        times = torch.zeros(B, T)
        species_mask = torch.zeros(B, max_species, dtype=torch.bool)
        reaction_mask = torch.zeros(B, max_rxn, dtype=torch.bool)

        for i, item in enumerate(batch):
            ns = item.crn.n_species
            nr = item.crn.n_reactions
            np_ = item.crn.propensity_params.shape[1]

            stoich[i, :nr, :ns] = item.crn.stoichiometry
            reactants[i, :nr, :ns] = item.crn.reactant_matrix
            prop_params[i, :nr, :np_] = item.crn.propensity_params
            prop_type_ids[i, :nr] = torch.tensor(
                [pt.value for pt in item.crn.propensity_types]
            )
            init_states[i, :ns] = item.initial_state
            trajs[i, :, :, :ns] = item.trajectories
            times[i] = item.times
            species_mask[i, :ns] = True
            reaction_mask[i, :nr] = True

        return {
            "stoichiometry": stoich,
            "reactant_matrix": reactants,
            "propensity_params": prop_params,
            "propensity_type_ids": prop_type_ids,
            "initial_states": init_states,
            "trajectories": trajs,
            "times": times,
            "species_mask": species_mask,
            "reaction_mask": reaction_mask,
        }
