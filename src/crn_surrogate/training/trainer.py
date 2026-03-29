from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crn_surrogate.configs.model_config import ModelConfig
from crn_surrogate.configs.training_config import SchedulerType, TrainingConfig
from crn_surrogate.data.crn import CRNDefinition
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset
from crn_surrogate.data.propensities import PropensityType
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.losses import CombinedTrajectoryLoss, TrajectoryLoss


@dataclass
class TrainingResult:
    """Return value of Trainer.train().

    Attributes:
        train_losses: Per-epoch mean training loss.
        val_losses: Validation losses recorded every val_every epochs.
        val_epochs: Epoch indices corresponding to val_losses (1-indexed).
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)


class Trainer:
    """Training loop for the CRN neural surrogate.

    Handles optimizer, LR scheduling, gradient clipping, checkpointing,
    and per-epoch loss tracking. Accepts the loss function as a constructor
    argument for flexibility.
    """

    def __init__(
        self,
        encoder: BipartiteGNNEncoder,
        sde: CRNNeuralSDE,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        loss_fn: TrajectoryLoss | None = None,
    ) -> None:
        """Args:
        encoder: The bipartite GNN encoder.
        sde: The neural SDE.
        model_config: Model hyperparameters.
        train_config: Training hyperparameters.
        loss_fn: Loss function to use. Defaults to CombinedTrajectoryLoss.
        """
        self._encoder = encoder
        self._sde = sde
        self._model_config = model_config
        self._train_config = train_config
        self._loss_fn = loss_fn if loss_fn is not None else CombinedTrajectoryLoss()
        self._solver = EulerMaruyamaSolver(model_config.sde)

        params = list(encoder.parameters()) + list(sde.parameters())
        self._optimizer = torch.optim.AdamW(
            params,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self._scheduler = self._build_scheduler()
        self._best_val_loss = float("inf")

    def train(
        self,
        train_dataset: CRNTrajectoryDataset,
        val_dataset: CRNTrajectoryDataset | None = None,
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            train_dataset: Training trajectories.
            val_dataset: Optional validation trajectories.

        Returns:
            TrainingResult with per-epoch train and val losses.
        """
        result = TrainingResult()
        collator = CRNCollator()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._train_config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        for epoch in range(1, self._train_config.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            result.train_losses.append(train_loss)

            val_loss: float | None = None
            if val_dataset is not None and epoch % self._train_config.val_every == 0:
                val_loss = self._validate(val_dataset)
                result.val_losses.append(val_loss)
                result.val_epochs.append(epoch)
                self._maybe_checkpoint(val_loss, epoch)
                print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
            else:
                print(f"Epoch {epoch:4d} | train={train_loss:.4f}")

            self._step_scheduler(val_loss)

        return result

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch and return mean loss."""
        self._encoder.train()
        self._sde.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="train", leave=False):
            loss = self._compute_batch_loss(batch)
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._encoder.parameters()) + list(self._sde.parameters()),
                self._train_config.grad_clip_norm,
            )
            self._optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
        """Compute mean loss over all items in the batch."""
        B = batch["stoichiometry"].shape[0]
        total = torch.zeros(1, device=batch["stoichiometry"].device)
        for idx in range(B):
            total = total + self._compute_item_loss(batch, idx)
        return total / B

    def _compute_item_loss(self, batch: dict, idx: int) -> torch.Tensor:
        """Compute loss for a single item in the batch."""
        crn = self._reconstruct_crn(batch, idx)
        n_species = int(batch["species_mask"][idx].sum().item())
        init_state = batch["initial_states"][idx, :n_species]
        # (M, T, n_species) — full set of SSA ground-truth trajectories
        true_trajs = batch["trajectories"][idx, :, :, :n_species]
        times = batch["times"][idx]

        ctx = self._encoder(crn, init_state)

        k = self._train_config.n_sde_samples
        pred_samples = [
            self._solver.solve(
                self._sde, init_state, ctx, times, self._train_config.dt
            ).states
            for _ in range(k)
        ]
        pred_states = torch.stack(pred_samples, dim=0)  # (K, T, n_species)

        species_mask = batch["species_mask"][idx, :n_species]
        return self._loss_fn.compute(pred_states, true_trajs, mask=species_mask)

    def _reconstruct_crn(self, batch: dict, idx: int) -> CRNDefinition:
        """Reconstruct a CRNDefinition from a padded batch dict for item idx."""
        n_species = int(batch["species_mask"][idx].sum().item())
        n_reactions = int(batch["reaction_mask"][idx].sum().item())

        stoich = batch["stoichiometry"][idx, :n_reactions, :n_species]
        reactants = batch["reactant_matrix"][idx, :n_reactions, :n_species]
        params = batch["propensity_params"][idx, :n_reactions]
        type_ids = batch["propensity_type_ids"][idx, :n_reactions]
        ptypes = tuple(PropensityType(int(t.item())) for t in type_ids)

        return CRNDefinition(
            stoichiometry=stoich,
            reactant_matrix=reactants,
            propensity_types=ptypes,
            propensity_params=params,
        )

    def _validate(self, val_dataset: CRNTrajectoryDataset) -> float:
        """Compute mean validation loss over the full validation dataset."""
        self._encoder.eval()
        self._sde.eval()
        collator = CRNCollator()
        loader = DataLoader(
            val_dataset,
            batch_size=self._train_config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                total_loss += self._compute_batch_loss(batch).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _build_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Instantiate the LR scheduler from TrainingConfig."""
        if self._train_config.scheduler_type == SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._train_config.max_epochs,
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=5,
            factor=0.5,
        )

    def _step_scheduler(self, val_loss: float | None) -> None:
        """Step the LR scheduler, supplying val_loss for ReduceLROnPlateau."""
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self._scheduler.step(val_loss)
        else:
            self._scheduler.step()

    def _maybe_checkpoint(self, val_loss: float, epoch: int) -> None:
        """Save a checkpoint if validation loss improved."""
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            Path(self._train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(
                self._train_config.checkpoint_dir, f"best_epoch{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state": self._encoder.state_dict(),
                    "sde_state": self._sde.state_dict(),
                    "val_loss": val_loss,
                },
                path,
            )
