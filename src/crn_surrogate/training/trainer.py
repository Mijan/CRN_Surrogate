"""End-to-end training loop for the CRN neural surrogate."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)
from crn_surrogate.data.dataset import CRNTrajectoryDataset
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.simulator.base import SurrogateModel
from crn_surrogate.simulator.ode_solver import EulerODESolver
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.simulator.state_transform import StateTransform
from crn_surrogate.training.checkpointing import CheckpointManager
from crn_surrogate.training.data_cache import DataCache
from crn_surrogate.training.losses import (
    CombinedRolloutLoss,
    RolloutLoss,
    StepLoss,
)
from crn_surrogate.training.profiler import PhaseTimer, ProfileLogger, WandbLogger


@dataclass
class TrainingResult:
    """Return value of Trainer.train().

    Attributes:
        train_losses: Per-epoch mean training loss.
        val_losses: Validation rollout losses recorded every val_every epochs.
        val_step_losses: Validation step losses recorded every val_every epochs.
        val_epochs: Epoch indices corresponding to val_losses (1-indexed).
        grad_norms: Per-epoch mean gradient norm (pre-clipping). Free to
            compute since clip_grad_norm_ already calculates it.
        learning_rates: Learning rate at the end of each epoch.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_step_losses: list[float] = field(default_factory=list)
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
        model: SurrogateModel,
        train_config: TrainingConfig,
        simulator: EulerMaruyamaSolver | EulerODESolver,
        step_loss: StepLoss,
        rollout_loss: RolloutLoss | None = None,
    ) -> None:
        """Args:
        encoder: The bipartite GNN encoder.
        model: The surrogate model (NeuralDrift or NeuralSDE).
        train_config: Training hyperparameters.
        simulator: Solver for rollout validation and scheduled-sampling training.
        step_loss: Per-transition loss (MSEStepLoss for ODE, NLLStepLoss for SDE).
        rollout_loss: Full-trajectory loss. Defaults to CombinedRolloutLoss.
            Only used when training_mode is FULL_ROLLOUT or SCHEDULED_SAMPLING.
        """
        self._encoder = encoder
        self._model = model
        self._train_config = train_config
        self._rollout_loss = (
            rollout_loss if rollout_loss is not None else CombinedRolloutLoss()
        )
        self._state_transform: StateTransform = StateTransform()
        self._solver: EulerMaruyamaSolver | EulerODESolver = simulator
        self._step_loss = step_loss

        self._device = next(encoder.parameters(), torch.zeros(1)).device

        params = (
            list(encoder.parameters())
            + list(model.parameters())
            + step_loss.parameters()
        )
        self._optimizer = torch.optim.AdamW(
            params,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self._scheduler = self._build_scheduler()

        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=train_config.checkpoint_dir,
            wandb_run_name=train_config.wandb_run_name
            if train_config.use_wandb
            else None,
            checkpoint_every=train_config.checkpoint_every,
        )
        self._timer = PhaseTimer(device=self._device)
        self._csv_logger = ProfileLogger(train_config.log_dir)
        self._wandb: WandbLogger | None = None
        if train_config.use_wandb:
            import dataclasses

            self._wandb = WandbLogger(
                config=dataclasses.asdict(train_config),
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
        n_species_pad = self._model.n_species
        self._train_cache = DataCache.from_dataset(
            train_dataset,
            self._device,
            n_species_pad,
            gpu_memory_fraction=self._train_config.gpu_memory_fraction,
        )
        if val_dataset is not None:
            self._val_cache = DataCache.from_dataset(
                val_dataset,
                self._device,
                n_species_pad,
                gpu_memory_fraction=self._train_config.gpu_memory_fraction,
            )

        for epoch in range(start_epoch, self._train_config.max_epochs + 1):
            self._timer = PhaseTimer(device=self._timer.device)
            train_loss, mean_grad_norm = self._train_epoch(self._train_cache, epoch)
            result.train_losses.append(train_loss)
            result.grad_norms.append(mean_grad_norm)
            result.learning_rates.append(self._optimizer.param_groups[0]["lr"])

            val_loss: float | None = None
            val_step_loss: float | None = None
            if val_dataset is not None and epoch % self._train_config.val_every == 0:
                do_rollout = self._effective_mode(epoch) != TrainingMode.TEACHER_FORCING
                val_loss, val_step_loss = self._validate(
                    self._val_cache, compute_rollout=do_rollout
                )
                result.val_losses.append(val_loss)
                result.val_step_losses.append(val_step_loss)
                result.val_epochs.append(epoch)
                self._checkpoint_mgr.save_best(
                    self._build_state(epoch, val_loss=val_step_loss),
                    val_step_loss,
                    epoch,
                )
                val_part = f"val={val_loss:.4f} | " if do_rollout else ""
                print(
                    f"Epoch {epoch:4d} | train={train_loss:.4f} | "
                    f"{val_part}val_step={val_step_loss:.4f} | grad={mean_grad_norm:.3f}"
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
                if val_step_loss is not None:
                    metrics["val_step_loss"] = val_step_loss
                if val_loss is not None and val_loss != 0.0:
                    metrics["val_loss"] = val_loss
                metrics.update(self._step_loss.extra_metrics())
                self._wandb.log_epoch(metrics)
                self._wandb.log_phase_timings(self._timer)

            self._step_scheduler(val_loss)

        # wandb.finish() is intentionally NOT called here.
        # The caller is responsible for closing the run after any post-training
        # logging (e.g. saving model artifacts) is complete.

        return result

    def load_checkpoint(self, checkpoint: dict) -> int:
        """Restore training state from a checkpoint.

        Backward-compatible: handles old ``measurement_model_state`` key and new
        ``step_loss_state`` key.

        Args:
            checkpoint: Dict loaded from a checkpoint .pt file.

        Returns:
            The epoch to resume from (checkpoint["epoch"] + 1).
        """
        start_epoch = self._checkpoint_mgr.load(
            checkpoint,
            self._encoder,
            self._model,
            self._optimizer,
            self._scheduler,
        )
        if "step_loss_state" in checkpoint:
            self._step_loss.load_state_dict(checkpoint["step_loss_state"])
        elif "measurement_model_state" in checkpoint:
            # Backward compat with old checkpoints
            self._step_loss.load_state_dict(
                {"measurement_model": checkpoint["measurement_model_state"]}
            )
        return start_epoch

    def _build_state(self, epoch: int, **extra) -> dict:
        """Build a checkpoint dict from current model and optimizer state.

        Args:
            epoch: Current epoch number.
            **extra: Additional metadata fields (e.g. val_loss, train_loss).

        Returns:
            Dict suitable for torch.save, containing encoder_state, model_state,
            optimizer_state, scheduler_state, best_val_loss, epoch, and any
            extra fields.
        """
        state = {
            "epoch": epoch,
            "encoder_state": self._encoder.state_dict(),
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "scheduler_state": self._scheduler.state_dict(),
            "best_val_loss": self._checkpoint_mgr.best_val_loss,
            **extra,
        }
        step_loss_state = self._step_loss.state_dict()
        if step_loss_state:
            state["step_loss_state"] = step_loss_state
        return state

    def _make_batches(self, cache: DataCache, shuffle: bool) -> list[torch.Tensor]:
        """Split cache indices into batches, optionally shuffled.

        Args:
            cache: The DataCache to iterate over.
            shuffle: Whether to shuffle indices before splitting.

        Returns:
            List of 1-D index tensors, each of length <= batch_size.
        """
        N = cache.trajectories.shape[0]
        indices = torch.randperm(N) if shuffle else torch.arange(N)
        return list(indices.split(self._train_config.batch_size))

    def _train_epoch(self, cache: DataCache, epoch: int) -> tuple[float, float]:
        """Run one training epoch and return (mean loss, mean pre-clip grad norm)."""
        self._encoder.train()
        self._model.train()
        total_loss = 0.0
        batch_grad_norms: list[float] = []
        n_batches = 0

        batches = self._make_batches(cache, shuffle=self._train_config.shuffle_train)
        for batch_indices in tqdm(batches, desc="train", leave=False):
            self._timer.start_batch(n_batches=n_batches)
            batch = cache.get_batch(batch_indices)

            with self._timer.time("forward"):
                loss = self._compute_batch_loss(batch, epoch)

            self._optimizer.zero_grad()
            with self._timer.time("backward"):
                loss.backward()
            # clip_grad_norm_ returns the pre-clipping total norm — free diagnostic
            all_params = (
                list(self._encoder.parameters())
                + list(self._model.parameters())
                + self._step_loss.parameters()
            )
            total_norm = nn.utils.clip_grad_norm_(
                all_params,
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

    def _prepare_batch(self, batch: dict) -> list[PreparedItem]:
        """Prepare all items in a batch using a single batched encoder pass.

        Transfers all CRNTensorReprs to the training device, runs the
        encoder once on the combined disconnected graph, and assembles
        PreparedItems from the pre-padded batch tensors.

        Args:
            batch: Batch dict from DataCache.get_batch().

        Returns:
            List of B PreparedItems.
        """
        B = len(batch["crn_reprs"])
        crn_reprs = [batch["crn_reprs"][idx].to(self._device) for idx in range(B)]
        contexts = self._encoder.forward_batch(crn_reprs)

        return [
            PreparedItem(
                context=contexts[idx],
                true_trajs_padded=batch["trajectories"][idx],
                species_mask=batch["species_mask"][idx],
                times=batch["times"][idx],
                init_state_padded=batch["initial_states"][idx],
            )
            for idx in range(B)
        ]

    def _compute_batch_loss(self, batch: dict, epoch: int = 1) -> torch.Tensor:
        """Compute mean loss over all items in the batch."""
        B = len(batch["crn_reprs"])
        mode = self._effective_mode(epoch)

        # Single batched encoder pass instead of B sequential ones
        items = self._prepare_batch(batch)

        if mode == TrainingMode.TEACHER_FORCING:
            return self._compute_batch_step_loss(items)

        return self._compute_batch_rollout(B, items)

    def _compute_batch_rollout(self, B: int, items: list[PreparedItem]) -> torch.Tensor:
        """Compute rollout loss sequentially over items."""
        total = torch.zeros(1, device=self._device)
        for item in items:
            k = self._train_config.n_sde_samples
            pred_samples = [
                self._solver.solve(
                    self._model,
                    item.init_state_padded.clone(),
                    item.context,
                    item.times,
                    self._train_config.dt,
                ).states
                for _ in range(k)
            ]
            pred_states = torch.stack(pred_samples, dim=0)  # (K, T, n_species_sde)
            total = total + self._rollout_loss.compute(
                pred_states,
                item.true_trajs_padded,
                mask=item.species_mask,
            )
        return total / B

    def _compute_batch_step_loss(
        self,
        items: list[PreparedItem],
    ) -> torch.Tensor:
        """Compute mean step loss over all items using a single batched forward pass.

        Collects all M*(T-1) transitions from all B items into one large tensor,
        runs one batched forward pass via model.predict_transition, applies the
        step loss, then normalizes per-item (each item contributes equally
        regardless of its number of active species).

        Args:
            items: List of B PreparedItems from _prepare_batch().

        Returns:
            Scalar mean step loss.
        """
        B = len(items)
        dt = (items[0].times[1] - items[0].times[0]).item()

        M, T, S = items[0].true_trajs_padded.shape
        n_trans = M * (T - 1)  # transitions per item

        # Stack trajectories: (B, M, T, S)
        all_trajs = torch.stack([item.true_trajs_padded for item in items])
        if self._state_transform is not None:
            all_trajs = self._state_transform.transform_trajectory(all_trajs)

        # Extract transitions: (B * n_trans, S)
        y_t = all_trajs[:, :, :-1, :].reshape(B * n_trans, S)
        y_next = all_trajs[:, :, 1:, :].reshape(B * n_trans, S)

        # Expand context vectors: (B, d_context) -> (B * n_trans, d_context)
        ctx_vectors = torch.stack([item.context.context_vector for item in items])
        ctx_expanded = (
            ctx_vectors.unsqueeze(1).expand(B, n_trans, -1).reshape(B * n_trans, -1)
        )

        # Time tensor: all items share the same time grid
        t_single = items[0].times[:-1].repeat(M)  # (n_trans,)
        t_all = t_single.repeat(B)  # (B * n_trans,)

        # ONE batched forward pass via polymorphic predict_transition
        mu, variance = self._model.predict_transition(t_all, y_t, ctx_expanded, dt)

        # Polymorphic loss (MSE for ODE, NLL for SDE)
        element_loss = self._step_loss.compute(y_next, mu, variance)  # (B * n_trans, S)

        # Apply species masks and normalize per item
        masks = torch.stack([item.species_mask for item in items])  # (B, S)
        masks_expanded = masks.unsqueeze(1).expand(B, n_trans, S)  # (B, n_trans, S)

        element_loss_reshaped = element_loss.reshape(B, n_trans, S)
        masked_loss = element_loss_reshaped * masks_expanded.float()

        # Per-item mean: sum over transitions and species, divide by (n_trans * n_active)
        loss_per_item = masked_loss.sum(dim=(1, 2))  # (B,)
        n_dims_per_item = masks.float().sum(dim=1)  # (B,)
        loss_per_item = loss_per_item / (n_trans * n_dims_per_item)  # (B,)

        return loss_per_item.mean()

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
        val_cache: DataCache,
        compute_rollout: bool = False,
    ) -> tuple[float, float]:
        """Compute validation losses over the full validation cache.

        Args:
            val_cache: Pre-built DataCache for the validation set.
            compute_rollout: If True, also compute the full SDE rollout loss
                (expensive). If False, only compute teacher-forcing step loss.

        Returns:
            Tuple of (rollout_loss, step_loss). rollout_loss is 0.0 when
            compute_rollout is False.
        """
        self._encoder.eval()
        self._model.eval()
        total_rollout = 0.0
        total_step = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_indices in self._make_batches(val_cache, shuffle=False):
                batch = val_cache.get_batch(batch_indices)
                B = len(batch["crn_reprs"])

                # Single batched encoder pass — shared by step loss and rollout
                items = self._prepare_batch(batch)

                total_step += self._compute_batch_step_loss(items).item()

                if compute_rollout:
                    rollout_total = torch.zeros(1, device=self._device)
                    for item in items:
                        k = self._train_config.n_sde_samples
                        pred_samples = [
                            self._solver.solve(
                                self._model,
                                item.init_state_padded.clone(),
                                item.context,
                                item.times,
                                self._train_config.dt,
                            ).states
                            for _ in range(k)
                        ]
                        pred_states = torch.stack(pred_samples, dim=0)
                        rollout_total = rollout_total + self._rollout_loss.compute(
                            pred_states,
                            item.true_trajs_padded,
                            mask=item.species_mask,
                        )
                    total_rollout += (rollout_total / B).item()

                n_batches += 1

        denom = max(n_batches, 1)
        return total_rollout / denom, total_step / denom

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
