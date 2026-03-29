"""profiler.py - Lightweight training profiler and W&B logger for CUDA workloads."""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch


class PhaseTimer:
    """Accumulates per-phase wall-clock times with optional CUDA synchronization."""

    def __init__(self, device=None):
        self.device = device
        self.records = defaultdict(list)  # phase -> list of durations
        self.batch_records = []  # list of per-batch dicts
        self._current_batch = {}

    def _sync(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start_batch(self, **metadata):
        """Begin a new batch. Any keyword arguments are stored alongside timings."""
        self._current_batch = dict(metadata)

    def time(self, phase_name):
        """Context manager that times a named phase within the current batch."""
        return _PhaseContext(self, phase_name)

    def end_batch(self):
        """Finalise the current batch record, appending GPU memory stats if available."""
        if torch.cuda.is_available():
            self._current_batch["gpu_mem_MB"] = torch.cuda.memory_allocated() / 1e6
            self._current_batch["gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 1e6
        self.batch_records.append(self._current_batch)
        self._current_batch = {}

    def summary(self):
        """Print a summary table of per-phase timing statistics."""
        print(
            f"\n{'Phase':<20s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'total':>8s} {'n':>5s}"
        )
        print("-" * 65)
        total_all = 0
        for phase, times in self.records.items():
            arr = np.array(times)
            total_all += arr.sum()
            print(
                f"  {phase:<18s} {arr.mean():8.4f} {arr.std():8.4f} "
                f"{arr.min():8.4f} {arr.max():8.4f} {arr.sum():8.2f} {len(arr):5d}"
            )
        print("-" * 65)
        print(f"  {'TOTAL':<18s} {'':>8s} {'':>8s} {'':>8s} {'':>8s} {total_all:8.2f}")
        if torch.cuda.is_available():
            print(
                f"\n  GPU peak memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
            )


class ProfileLogger:
    """Writes per-batch timings and per-epoch phase summaries to CSV files.

    Produces two files inside ``log_dir``:

    - ``profiler_batches.csv``  — one row per training batch (columns inferred
      from the metadata passed to :meth:`PhaseTimer.start_batch` plus any timed
      phases and GPU metrics)
    - ``profiler_epochs.csv``   — one row per phase per epoch (mean/std/min/max/total)
    """

    _EPOCH_FIELDS = [
        "epoch",
        "phase",
        "mean_s",
        "std_s",
        "min_s",
        "max_s",
        "total_s",
        "n",
    ]

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._batch_path = os.path.join(log_dir, "profiler_batches.csv")
        self._epoch_path = os.path.join(log_dir, "profiler_epochs.csv")
        self._batch_fields: list | None = None  # determined from first record
        self._write_header(self._epoch_path, self._EPOCH_FIELDS)

    @staticmethod
    def _write_header(path: str, fields: list) -> None:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def log_epoch(self, epoch: int, timer: "PhaseTimer") -> None:
        """Append all batch records and the phase summary for one epoch."""
        if timer.batch_records:
            if self._batch_fields is None:
                # Infer column order from the first record
                sample = {"epoch": epoch, **timer.batch_records[0]}
                self._batch_fields = list(sample.keys())
                self._write_header(self._batch_path, self._batch_fields)

            with open(self._batch_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self._batch_fields,
                    extrasaction="ignore",
                    restval="",
                )
                for rec in timer.batch_records:
                    writer.writerow({"epoch": epoch, **rec})

        with open(self._epoch_path, "a", newline="") as f:
            epoch_writer = csv.writer(f)
            for phase, times in timer.records.items():
                arr = np.array(times)
                epoch_writer.writerow(
                    [
                        epoch,
                        phase,
                        f"{arr.mean():.6f}",
                        f"{arr.std():.6f}",
                        f"{arr.min():.6f}",
                        f"{arr.max():.6f}",
                        f"{arr.sum():.6f}",
                        len(arr),
                    ]
                )


class WandbLogger:
    """Thin wandb wrapper for epoch metrics and per-phase timings."""

    def __init__(
        self, config: dict, project: str = "my-project", run_name: str | None = None
    ):
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config)

    def log_epoch(self, metrics: dict) -> None:
        """Log a dict of scalar metrics (e.g. loss, accuracy, epoch)."""
        self._wandb.log(metrics)

    def log_phase_timings(self, timer: "PhaseTimer") -> None:
        """Log per-phase timing summaries from a PhaseTimer."""
        timing = {}
        for phase, times in timer.records.items():
            arr = np.array(times)
            timing[f"timing/{phase}_mean_s"] = float(arr.mean())
            timing[f"timing/{phase}_total_s"] = float(arr.sum())
        if torch.cuda.is_available():
            timing["gpu/peak_MB"] = torch.cuda.max_memory_allocated() / 1e6
        self._wandb.log(timing)

    def finish(self) -> None:
        """Mark the W&B run as finished."""
        self._wandb.finish()


class _PhaseContext:
    """Context manager for timing a single named phase."""

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer._sync()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.timer._sync()
        dt = time.time() - self.t0
        self.timer.records[self.name].append(dt)
        self.timer._current_batch[self.name] = dt
        return False
