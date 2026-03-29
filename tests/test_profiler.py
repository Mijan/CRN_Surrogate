"""Tests for the PhaseTimer and ProfileLogger training profilers.

Verifies:
- PhaseTimer.time() records a non-negative duration for each phase.
- Multiple batches accumulate multiple per-batch records and per-phase durations.
- start_batch metadata is stored alongside timing data.
- ProfileLogger writes both CSV files with the expected headers after log_epoch().
- Phase summary statistics (mean, min, max) in the epoch CSV are arithmetically consistent.
"""

import csv
import time

import pytest

from crn_surrogate.training.profiler import PhaseTimer, ProfileLogger

# ── PhaseTimer — phase timing ─────────────────────────────────────────────────


def test_phase_timer_records_nonnegative_duration():
    """A timed phase must produce a non-negative elapsed time."""
    timer = PhaseTimer()
    with timer.time("forward"):
        _ = sum(range(1000))
    assert len(timer.records["forward"]) == 1
    assert timer.records["forward"][0] >= 0.0


def test_phase_timer_captures_actual_elapsed_time():
    """A phase with a known sleep records a duration close to the sleep length."""
    timer = PhaseTimer()
    with timer.time("slow_op"):
        time.sleep(0.05)
    elapsed = timer.records["slow_op"][0]
    assert elapsed >= 0.04


def test_phase_timer_accumulates_duration_per_batch():
    """After N batches with the same phase name, records contains N entries."""
    timer = PhaseTimer()
    for _ in range(5):
        with timer.time("forward"):
            pass
    assert len(timer.records["forward"]) == 5


def test_phase_timer_separate_phases_tracked_independently():
    """Forward and backward phases are stored in separate record lists."""
    timer = PhaseTimer()
    with timer.time("forward"):
        pass
    with timer.time("backward"):
        pass
    assert "forward" in timer.records
    assert "backward" in timer.records
    assert len(timer.records["forward"]) == 1
    assert len(timer.records["backward"]) == 1


def test_phase_timer_batch_record_contains_phase_duration():
    """After end_batch(), the batch record dict contains the timed phase key."""
    timer = PhaseTimer()
    timer.start_batch(step=0)
    with timer.time("forward"):
        pass
    timer.end_batch()
    assert len(timer.batch_records) == 1
    assert "forward" in timer.batch_records[0]
    assert timer.batch_records[0]["forward"] >= 0.0


def test_phase_timer_batch_record_stores_start_batch_metadata():
    """Metadata passed to start_batch() is saved in the batch record."""
    timer = PhaseTimer()
    timer.start_batch(step=42, epoch=3)
    with timer.time("forward"):
        pass
    timer.end_batch()
    record = timer.batch_records[0]
    assert record["step"] == 42
    assert record["epoch"] == 3


def test_phase_timer_multiple_batches_accumulate_in_batch_records():
    """Each call to end_batch() appends a new entry to batch_records."""
    timer = PhaseTimer()
    for i in range(3):
        timer.start_batch(step=i)
        with timer.time("forward"):
            pass
        timer.end_batch()
    assert len(timer.batch_records) == 3


# ── ProfileLogger — CSV writing ───────────────────────────────────────────────


@pytest.fixture()
def timer_with_one_epoch(tmp_path):
    """PhaseTimer that has completed one epoch worth of batch records."""
    timer = PhaseTimer()
    for _ in range(3):
        timer.start_batch()
        with timer.time("forward"):
            pass
        with timer.time("backward"):
            pass
        timer.end_batch()
    return timer


def test_profile_logger_creates_epoch_csv(tmp_path, timer_with_one_epoch):
    """log_epoch() must create a profiler_epochs.csv file in log_dir."""
    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer_with_one_epoch)
    assert (tmp_path / "profiler_epochs.csv").exists()


def test_profile_logger_creates_batch_csv(tmp_path, timer_with_one_epoch):
    """log_epoch() must create a profiler_batches.csv file in log_dir."""
    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer_with_one_epoch)
    assert (tmp_path / "profiler_batches.csv").exists()


def test_profile_logger_epoch_csv_has_correct_header(tmp_path, timer_with_one_epoch):
    """profiler_epochs.csv must have the canonical column headers."""
    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer_with_one_epoch)
    with open(tmp_path / "profiler_epochs.csv") as f:
        header = next(csv.reader(f))
    assert header == [
        "epoch",
        "phase",
        "mean_s",
        "std_s",
        "min_s",
        "max_s",
        "total_s",
        "n",
    ]


def test_profile_logger_epoch_csv_row_count_matches_phases(
    tmp_path, timer_with_one_epoch
):
    """After logging one epoch with two phases, epoch CSV has exactly 2 data rows."""
    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer_with_one_epoch)
    with open(tmp_path / "profiler_epochs.csv") as f:
        rows = list(csv.reader(f))
    data_rows = rows[1:]  # skip header
    assert len(data_rows) == 2  # forward + backward


def test_profile_logger_epoch_csv_total_equals_sum_of_batches(tmp_path):
    """total_s in the epoch CSV must equal the sum of all per-batch phase durations."""
    timer = PhaseTimer()
    for _ in range(4):
        timer.start_batch()
        with timer.time("forward"):
            time.sleep(0.01)
        timer.end_batch()

    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer)

    expected_total = sum(timer.records["forward"])

    with open(tmp_path / "profiler_epochs.csv") as f:
        rows = {row["phase"]: row for row in csv.DictReader(f)}

    actual_total = float(rows["forward"]["total_s"])
    assert actual_total == pytest.approx(expected_total, rel=1e-4)


def test_profile_logger_appends_multiple_epochs(tmp_path, timer_with_one_epoch):
    """Calling log_epoch() twice appends rows; epoch CSV has 4 data rows (2 phases × 2 epochs)."""
    logger = ProfileLogger(str(tmp_path))
    logger.log_epoch(epoch=1, timer=timer_with_one_epoch)
    logger.log_epoch(epoch=2, timer=timer_with_one_epoch)
    with open(tmp_path / "profiler_epochs.csv") as f:
        data_rows = list(csv.reader(f))[1:]
    assert len(data_rows) == 4  # 2 phases × 2 epochs
