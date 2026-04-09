"""Pre-transfer dataset cache for eliminating per-batch CPU-to-GPU overhead.

Stacks all dataset tensors into dense arrays once at training start, transfers
them to GPU in one shot, and serves batches via index slicing on GPU-resident
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crn_surrogate.data.dataset import CRNTrajectoryDataset
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr


def _estimate_crn_repr_bytes(r: CRNTensorRepr) -> int:
    """Estimate GPU memory required for one CRNTensorRepr, including cached edges."""
    total = 0
    for attr in [
        "stoichiometry",
        "dependency_matrix",
        "propensity_type_ids",
        "propensity_params",
        "is_external",
    ]:
        t = getattr(r, attr)
        if t is not None:
            total += t.nelement() * t.element_size()
    if hasattr(r, "_cached_edges"):
        edges = r._cached_edges
        for attr in [
            "rxn_to_species_index",
            "rxn_to_species_feat",
            "species_to_rxn_index",
            "species_to_rxn_feat",
        ]:
            t = getattr(edges, attr)
            total += t.nelement() * t.element_size()
    return total


@dataclass
class DataCache:
    """Pre-stacked, optionally GPU-resident dataset tensors.

    Attributes:
        trajectories: (N, M, T, S_pad) trajectory tensor, GPU or CPU.
        times: (N, T) time grids, always on device.
        init_states: (N, S_pad) initial states, always on device.
        species_masks: (N, S_pad) bool mask (True = real species), always on device.
        crn_reprs: Length-N list of CRNTensorRepr, each on device.
        n_species_per_item: Number of active species per dataset item.
        n_reactions_per_item: Number of reactions per dataset item.
        trajectories_on_gpu: Whether trajectories were transferred to device.
        device: Target device for all on-device tensors.
    """

    trajectories: torch.Tensor
    times: torch.Tensor
    init_states: torch.Tensor
    species_masks: torch.Tensor
    crn_reprs: list[CRNTensorRepr]
    n_species_per_item: list[int]
    n_reactions_per_item: list[int]
    trajectories_on_gpu: bool
    device: torch.device

    @classmethod
    def from_dataset(
        cls,
        dataset: CRNTrajectoryDataset,
        device: torch.device,
        n_species_pad: int,
        gpu_memory_fraction: float = 0.5,
    ) -> "DataCache":
        """Build a DataCache by stacking and transferring all dataset tensors.

        Args:
            dataset: Source trajectory dataset.
            device: Target device (CPU or CUDA).
            n_species_pad: Species padding width — must be >= max n_species in dataset.
            gpu_memory_fraction: Fraction of free GPU memory allowed for trajectories.
                When the full dataset (including trajectories) exceeds this budget,
                trajectories stay on CPU and are transferred per-batch.

        Returns:
            DataCache with all metadata tensors on device, trajectories conditionally.
        """
        N = len(dataset)
        first = dataset[0]
        M = first.trajectories.shape[0]
        T = first.trajectories.shape[1]

        # Step 1: Pre-build bipartite edges on CPU for every item.
        for i in range(N):
            _ = dataset[i].crn_repr.bipartite_edges

        # Step 2: Stack uniform tensors.
        trajectories = torch.zeros(N, M, T, n_species_pad)
        times_stacked = torch.zeros(N, T)
        init_states_stacked = torch.zeros(N, n_species_pad)
        species_masks_stacked = torch.zeros(N, n_species_pad, dtype=torch.bool)

        n_species_per_item: list[int] = []
        n_reactions_per_item: list[int] = []
        crn_reprs_cpu: list[CRNTensorRepr] = []

        for i in range(N):
            item = dataset[i]
            ns = item.crn_repr.n_species
            trajectories[i, :, :, :ns] = item.trajectories
            times_stacked[i] = item.times
            init_states_stacked[i, :ns] = item.initial_state
            species_masks_stacked[i, :ns] = True
            n_species_per_item.append(ns)
            n_reactions_per_item.append(item.crn_repr.n_reactions)
            crn_reprs_cpu.append(item.crn_repr)

        # Step 3: Estimate memory requirements.
        traj_bytes = trajectories.nelement() * trajectories.element_size()
        small_bytes = sum(
            t.nelement() * t.element_size()
            for t in [times_stacked, init_states_stacked, species_masks_stacked]
        )
        crn_bytes = sum(_estimate_crn_repr_bytes(r) for r in crn_reprs_cpu)
        total_bytes = traj_bytes + small_bytes + crn_bytes

        # Step 4: Decide whether trajectories fit on GPU.
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            budget = free * gpu_memory_fraction
            transfer_trajs = total_bytes < budget
        else:
            transfer_trajs = False  # CPU device: no transfer needed

        # Step 5: Transfer tensors.
        times_dev = times_stacked.to(device)
        init_states_dev = init_states_stacked.to(device)
        species_masks_dev = species_masks_stacked.to(device)
        crn_reprs_dev = [r.to(device) for r in crn_reprs_cpu]

        if transfer_trajs:
            trajectories_out = trajectories.to(device)
            trajs_on_gpu = True
        else:
            trajectories_out = trajectories
            trajs_on_gpu = False

        total_mb = total_bytes / (1024**2)
        trajs_label = "yes" if trajs_on_gpu else "CPU"
        print(
            f"DataCache: N={N}, S_pad={n_species_pad}, {total_mb:.0f} MB"
            f" -> GPU (trajs: {trajs_label})"
        )

        # Step 6: Return cache.
        return cls(
            trajectories=trajectories_out,
            times=times_dev,
            init_states=init_states_dev,
            species_masks=species_masks_dev,
            crn_reprs=crn_reprs_dev,
            n_species_per_item=n_species_per_item,
            n_reactions_per_item=n_reactions_per_item,
            trajectories_on_gpu=trajs_on_gpu,
            device=device,
        )

    def get_batch(self, indices: torch.Tensor) -> dict:
        """Retrieve a batch by index.

        Returns a dict matching CRNCollator output format for the keys consumed
        by Trainer._prepare_batch() and _compute_batch_loss().

        Args:
            indices: 1-D tensor of integer indices into this cache.

        Returns:
            Dict with keys: trajectories, times, initial_states, species_mask,
            crn_reprs, n_species_per_item, n_reactions_per_item.
        """
        trajs = self.trajectories[indices]
        if not self.trajectories_on_gpu:
            trajs = trajs.to(self.device, non_blocking=True)
        idx_list = indices.tolist()
        return {
            "trajectories": trajs,
            "times": self.times[indices],
            "initial_states": self.init_states[indices],
            "species_mask": self.species_masks[indices],
            "crn_reprs": [self.crn_reprs[i] for i in idx_list],
            "n_species_per_item": [self.n_species_per_item[i] for i in idx_list],
            "n_reactions_per_item": [self.n_reactions_per_item[i] for i in idx_list],
        }
