"""End-to-end training loop for the CRN neural surrogate."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crn_surrogate.configs.model_config import ModelConfig
from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)
from crn_surrogate.data.dataset import CRNCollator, CRNTrajectoryDataset
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.training.checkpointing import CheckpointManager
from crn_surrogate.training.losses import (
    CombinedTrajectoryLoss,
    GaussianTransitionNLL,
    TrajectoryLoss,
)
from crn_surrogate.training.profiler import PhaseTimer, ProfileLogger, WandbLogger


@dataclass
class TrainingResult:
    """Return value of Trainer.train().

    Attributes:
        train_losses: Per-epoch mean training loss.
        val_losses: Validation rollout losses recorded every val_every epochs.
        val_nll_losses: Validation NLL losses recorded every val_every epochs.
        val_epochs: Epoch indices corresponding to val_losses (1-indexed).
        grad_norms: Per-epoch mean gradient norm (pre-clipping). Free to
            compute since clip_grad_norm_ already calculates it.
        learning_rates: Learning rate at the end of each epoch.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_nll_losses: list[float] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedItem:
    """Single batch item prepared for loss computation.

    Bundles the encoder output, padded trajectories, species mask, time grid,
    and padded initial state. All tensors use the SDE's full dimensionality
    (n_species_sde), with inactive species zeroed and masked.

    Attributes:
        context: Encoder output for this CRN.
        true_trajs_padded: (M, T, n_species_sde) ground-truth trajectories.
        species_mask: (n_species_sde,) bool mask; True for active species.
        times: (T,) evaluation time grid.
        init_state_padded: (n_species_sde,) initial state (inactive species zero).
    """

    context: CRNContext
    true_trajs_padded: torch.Tensor
    species_mask: torch.Tensor
    times: torch.Tensor
    init_state_padded: torch.Tensor


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
        loss_fn: Rollout loss function. Defaults to CombinedTrajectoryLoss.
            Only used when training_mode is FULL_ROLLOUT or SCHEDULED_SAMPLING.
        """
        self._encoder = encoder
        self._sde = sde
        self._model_config = model_config
        self._train_config = train_config
        self._nll_loss = GaussianTransitionNLL(min_variance=1e-2)
        self._rollout_loss = (
            loss_fn if loss_fn is not None else CombinedTrajectoryLoss()
        )
        self._solver = EulerMaruyamaSolver(model_config.sde)

        params = list(encoder.parameters()) + list(sde.parameters())
        self._optimizer = torch.optim.AdamW(
            params,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self._scheduler = self._build_scheduler()

        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=train_config.checkpoint_dir,
            wandb_run_name=train_config.wandb_run_name if train_config.use_wandb else None,
            checkpoint_every=train_config.checkpoint_every,
        )

        self._device = next(encoder.parameters(), torch.zeros(1)).device
        self._timer = PhaseTimer(device=self._device)
        self._csv_logger = ProfileLogger(train_config.log_dir)
        self._wandb: WandbLogger | None = None
        if train_config.use_wandb:
            import dataclasses

            self._wandb = WandbLogger(
                config={
                    **dataclasses.asdict(train_config),
                    **dataclasses.asdict(model_config),
                },
                project=train_config.wandb_project,
                run_name=train_config.wandb_run_name,
            )

    def train(
        self,
        train_dataset: CRNTrajectoryDataset,
        val_dataset: CRNTrajectoryDataset | None = None,
        start_epoch: int = 1,
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            train_dataset: Training trajectories.
            val_dataset: Optional validation trajectories.
            start_epoch: Epoch to start from (>1 when resuming). Epochs before
                start_epoch are skipped. The scheduler is NOT stepped for
                skipped epochs since its state is restored from the checkpoint.

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

        for epoch in range(start_epoch, self._train_config.max_epochs + 1):
            self._timer = PhaseTimer(device=self._timer.device)
            train_loss, mean_grad_norm = self._train_epoch(train_loader, epoch)
            result.train_losses.append(train_loss)
            result.grad_norms.append(mean_grad_norm)
            result.learning_rates.append(self._optimizer.param_groups[0]["lr"])

            val_loss: float | None = None
            val_nll: float | None = None
            if val_dataset is not None and epoch % self._train_config.val_every == 0:
                do_rollout = self._effective_mode(epoch) != TrainingMode.TEACHER_FORCING
                val_loss, val_nll = self._validate(
                    val_dataset, compute_rollout=do_rollout
                )
                result.val_losses.append(val_loss)
                result.val_nll_losses.append(val_nll)
                result.val_epochs.append(epoch)
                self._checkpoint_mgr.save_best(
                    self._build_state(epoch, val_loss=val_nll), val_nll, epoch
                )
                val_part = f"val={val_loss:.4f} | " if do_rollout else ""
                print(
                    f"Epoch {epoch:4d} | train={train_loss:.4f} | "
                    f"{val_part}val_nll={val_nll:.4f} | grad={mean_grad_norm:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch:4d} | train={train_loss:.4f} | grad={mean_grad_norm:.3f}"
                )

            self._checkpoint_mgr.save_periodic(
                self._build_state(epoch, train_loss=train_loss), epoch, train_loss
            )

            self._csv_logger.log_epoch(epoch, self._timer)
            if self._wandb is not None:
                metrics: dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "grad_norm": mean_grad_norm,
                    "lr": self._optimizer.param_groups[0]["lr"],
                }
                if val_nll is not None:
                    metrics["val_nll"] = val_nll
                if val_loss is not None and val_loss != 0.0:
                    metrics["val_loss"] = val_loss
                self._wandb.log_epoch(metrics)
                self._wandb.log_phase_timings(self._timer)

            self._step_scheduler(val_loss)

        # wandb.finish() is intentionally NOT called here.
        # The caller is responsible for closing the run after any post-training
        # logging (e.g. saving model artifacts) is complete.

        return result

    def load_checkpoint(self, checkpoint: dict) -> int:
        """Restore training state from a checkpoint.

        Args:
            checkpoint: Dict loaded from a checkpoint .pt file.

        Returns:
            The epoch to resume from (checkpoint["epoch"] + 1).
        """
        return self._checkpoint_mgr.load(
            checkpoint, self._encoder, self._sde,
            self._optimizer, self._scheduler,
        )

    def _build_state(self, epoch: int, **extra) -> dict:
        """Build a checkpoint dict from current model and optimizer state.

        Args:
            epoch: Current epoch number.
            **extra: Additional metadata fields (e.g. val_loss, train_loss).

        Returns:
            Dict suitable for torch.save, containing encoder_state, sde_state,
            optimizer_state, scheduler_state, best_val_loss, epoch, and any
            extra fields.
        """
        return {
            "epoch": epoch,
            "encoder_state": self._encoder.state_dict(),
            "sde_state": self._sde.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "scheduler_state": self._scheduler.state_dict(),
            "best_val_loss": self._checkpoint_mgr.best_val_loss,
            **extra,
        }

    def _batch_to_device(self, batch: dict) -> dict:
        """Move all tensor values in a batch dict to the training device."""
        return {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        """Run one training epoch and return (mean loss, mean pre-clip grad norm)."""
        self._encoder.train()
        self._sde.train()
        total_loss = 0.0
        batch_grad_norms: list[float] = []
        n_batches = 0

        for batch in tqdm(loader, desc="train", leave=False):
            self._timer.start_batch(n_batches=n_batches)
            batch = self._batch_to_device(batch)

            with self._timer.time("forward"):
                loss = self._compute_batch_loss(batch, epoch)

            self._optimizer.zero_grad()
            with self._timer.time("backward"):
                loss.backward()
            # clip_grad_norm_ returns the pre-clipping total norm — free diagnostic
            total_norm = nn.utils.clip_grad_norm_(
                list(self._encoder.parameters()) + list(self._sde.parameters()),
                self._train_config.grad_clip_norm,
            )
            batch_grad_norms.append(total_norm.item())
            self._optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self._timer.end_batch()

        denom = max(n_batches, 1)
        mean_grad_norm = sum(batch_grad_norms) / denom
        return total_loss / denom, mean_grad_norm

    def _prepare_item(self, batch: dict, idx: int) -> PreparedItem:
        """Extract, encode, and pad a single item from a collated batch.

        Reconstructs the CRNTensorRepr, runs the encoder, pads trajectories
        and initial state to the SDE's n_species, and builds the species mask.

        Args:
            batch: Collated batch dict from CRNCollator.
            idx: Index of the item within the batch.

        Returns:
            PreparedItem with all tensors at SDE dimensionality.
        """
        n_species_actual = int(batch["species_mask"][idx].sum().item())
        n_reactions = int(batch["reaction_mask"][idx].sum().item())
        n_species_sde = self._sde.n_species

        crn_repr = CRNTensorRepr(
            stoichiometry=batch["stoichiometry"][idx, :n_reactions, :n_species_actual],
            dependency_matrix=batch["dependency_matrix"][idx, :n_reactions, :n_species_actual],
            propensity_type_ids=batch["propensity_type_ids"][idx, :n_reactions],
            propensity_params=batch["propensity_params"][idx, :n_reactions],
        )
        ctx = self._encoder(crn_repr)

        true_trajs = batch["trajectories"][idx, :, :, :n_species_actual]  # (M, T, ns)
        device = true_trajs.device

        species_mask = torch.zeros(n_species_sde, dtype=torch.bool, device=device)
        species_mask[:n_species_actual] = True

        M, T, _ = true_trajs.shape
        true_trajs_padded = torch.zeros(M, T, n_species_sde, device=device)
        true_trajs_padded[:, :, :n_species_actual] = true_trajs

        init_state = batch["initial_states"][idx, :n_species_actual]
        init_state_padded = torch.zeros(n_species_sde, device=device)
        init_state_padded[:n_species_actual] = init_state

        return PreparedItem(
            context=ctx,
            true_trajs_padded=true_trajs_padded,
            species_mask=species_mask,
            times=batch["times"][idx],
            init_state_padded=init_state_padded,
        )

    def _compute_batch_loss(self, batch: dict, epoch: int = 1) -> torch.Tensor:
        """Compute mean loss over all items in the batch."""
        B = batch["stoichiometry"].shape[0]
        total = torch.zeros(1, device=batch["stoichiometry"].device)
        for idx in range(B):
            total = total + self._compute_item_loss(batch, idx, epoch)
        return total / B

    def _compute_item_loss(self, batch: dict, idx: int, epoch: int = 1) -> torch.Tensor:
        """Compute loss for a single item in the batch, dispatching on training mode.

        The SDE operates in raw count space; the GaussianTransitionNLL self-normalizes
        via residual² / σ² = O(1) regardless of species count scale.
        """
        item = self._prepare_item(batch, idx)
        mode = self._effective_mode(epoch)

        if mode == TrainingMode.TEACHER_FORCING:
            return self._nll_loss.compute(
                sde=self._sde,
                crn_context=item.context,
                true_trajectory=item.true_trajs_padded,
                times=item.times,
                dt=self._train_config.dt,
                mask=item.species_mask,
            )

        k = self._train_config.n_sde_samples
        pred_samples = [
            self._solver.solve(
                self._sde, item.init_state_padded.clone(), item.context,
                item.times, self._train_config.dt,
            ).states
            for _ in range(k)
        ]
        pred_states = torch.stack(pred_samples, dim=0)  # (K, T, n_species_sde)
        return self._rollout_loss.compute(
            pred_states, item.true_trajs_padded, mask=item.species_mask,
        )

    def _effective_mode(self, epoch: int) -> TrainingMode:
        """Determine effective training mode for this epoch."""
        cfg = self._train_config
        if cfg.training_mode == TrainingMode.TEACHER_FORCING:
            return TrainingMode.TEACHER_FORCING
        if cfg.training_mode == TrainingMode.FULL_ROLLOUT:
            return TrainingMode.FULL_ROLLOUT
        # Scheduled sampling: teacher forcing early, rollout later
        if epoch < cfg.scheduled_sampling_start_epoch:
            return TrainingMode.TEACHER_FORCING
        if epoch >= cfg.scheduled_sampling_end_epoch:
            return TrainingMode.FULL_ROLLOUT
        progress = (epoch - cfg.scheduled_sampling_start_epoch) / (
            cfg.scheduled_sampling_end_epoch - cfg.scheduled_sampling_start_epoch
        )
        # TODO: torch.rand is not seeded by the training seed, making scheduled
        # sampling non-reproducible. Fix by threading a seeded torch.Generator
        # through TrainingConfig when reproducibility is required.
        if torch.rand(1).item() < progress:
            return TrainingMode.FULL_ROLLOUT
        return TrainingMode.TEACHER_FORCING

    def _validate(
        self,
        val_dataset: CRNTrajectoryDataset,
        compute_rollout: bool = False,
    ) -> tuple[float, float]:
        """Compute validation losses over the full validation dataset.

        Args:
            val_dataset: Validation trajectories.
            compute_rollout: If True, also compute the full SDE rollout loss
                (expensive). If False, only compute teacher-forcing NLL.

        Returns:
            Tuple of (rollout_loss, nll_loss). rollout_loss is 0.0 when
            compute_rollout is False.
        """
        self._encoder.eval()
        self._sde.eval()
        collator = CRNCollator()
        loader = DataLoader(
            val_dataset,
            batch_size=self._train_config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        total_rollout = 0.0
        total_nll = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = self._batch_to_device(batch)
                B = batch["stoichiometry"].shape[0]

                nll_total = torch.zeros(1, device=batch["stoichiometry"].device)
                rollout_total = torch.zeros(1, device=batch["stoichiometry"].device)

                for idx in range(B):
                    item = self._prepare_item(batch, idx)
                    nll_total = nll_total + self._nll_loss.compute(
                        sde=self._sde,
                        crn_context=item.context,
                        true_trajectory=item.true_trajs_padded,
                        times=item.times,
                        dt=self._train_config.dt,
                        mask=item.species_mask,
                    )
                    if compute_rollout:
                        k = self._train_config.n_sde_samples
                        pred_samples = [
                            self._solver.solve(
                                self._sde, item.init_state_padded.clone(),
                                item.context, item.times, self._train_config.dt,
                            ).states
                            for _ in range(k)
                        ]
                        pred_states = torch.stack(pred_samples, dim=0)
                        rollout_total = rollout_total + self._rollout_loss.compute(
                            pred_states, item.true_trajs_padded, mask=item.species_mask,
                        )

                total_nll += (nll_total / B).item()
                total_rollout += (rollout_total / B).item()
                n_batches += 1

        denom = max(n_batches, 1)
        return total_rollout / denom, total_nll / denom

    def _build_scheduler(
        self,
    ) -> (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
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
