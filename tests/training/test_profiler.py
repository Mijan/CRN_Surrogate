"""Tests for PhaseTimer and ProfileLogger."""

from __future__ import annotations

import time

from crn_surrogate.training.profiler import PhaseTimer, ProfileLogger

# ── PhaseTimer ────────────────────────────────────────────────────────────────


def test_time_context_manager() -> None:
    timer = PhaseTimer()
    with timer.time("forward"):
        time.sleep(0.01)
    assert "forward" in timer.records
    assert timer.records["forward"][0] > 0.0


def test_multiple_phases() -> None:
    timer = PhaseTimer()
    with timer.time("forward"):
        time.sleep(0.005)
    with timer.time("backward"):
        time.sleep(0.005)
    assert "forward" in timer.records
    assert "backward" in timer.records


def test_batch_lifecycle() -> None:
    timer = PhaseTimer()
    timer.start_batch(batch_idx=0)
    with timer.time("forward"):
        time.sleep(0.005)
    with timer.time("backward"):
        time.sleep(0.005)
    timer.end_batch()

    assert len(timer.batch_records) == 1
    record = timer.batch_records[0]
    assert "forward" in record
    assert "backward" in record


# ── ProfileLogger ─────────────────────────────────────────────────────────────


def test_creates_csv_files(tmp_path) -> None:
    logger = ProfileLogger(log_dir=str(tmp_path))
    timer = PhaseTimer()
    timer.start_batch(batch_idx=0)
    with timer.time("forward"):
        time.sleep(0.005)
    timer.end_batch()

    logger.log_epoch(epoch=1, timer=timer)

    batch_csv = tmp_path / "profiler_batches.csv"
    epoch_csv = tmp_path / "profiler_epochs.csv"
    assert batch_csv.exists()
    assert epoch_csv.exists()
    assert batch_csv.stat().st_size > 0
    assert epoch_csv.stat().st_size > 0
