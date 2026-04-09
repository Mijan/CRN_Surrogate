"""Tests for CheckpointManager."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crn_surrogate.training.checkpointing import CheckpointManager


def _make_manager(tmp_path, **kwargs) -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=str(tmp_path), **kwargs)


def _dummy_state(epoch: int = 1) -> dict:
    return {
        "epoch": epoch,
        "encoder_state": nn.Linear(4, 4).state_dict(),
        "model_state": nn.Linear(4, 4).state_dict(),
        "optimizer_state": {},
        "scheduler_state": {},
        "best_val_loss": float("inf"),
    }


# ── save_best ─────────────────────────────────────────────────────────────────


def test_save_best_creates_file(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save_best(_dummy_state(1), val_loss=1.0, epoch=1)
    assert (tmp_path / "best_epoch1.pt").exists()


def test_save_best_skips_worse(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save_best(_dummy_state(1), val_loss=1.0, epoch=1)
    mgr.save_best(_dummy_state(2), val_loss=2.0, epoch=2)
    files = list(tmp_path.glob("best_epoch*.pt"))
    assert len(files) == 1
    assert files[0].name == "best_epoch1.pt"


def test_save_best_updates_on_improvement(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save_best(_dummy_state(1), val_loss=2.0, epoch=1)
    mgr.save_best(_dummy_state(2), val_loss=1.0, epoch=2)
    files = list(tmp_path.glob("best_epoch*.pt"))
    assert len(files) == 2
    assert mgr.best_val_loss == pytest.approx(1.0)


def test_save_best_updates_on_improvement_value(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    mgr.save_best(_dummy_state(1), val_loss=2.0, epoch=1)
    mgr.save_best(_dummy_state(2), val_loss=1.0, epoch=2)
    assert mgr.best_val_loss == 1.0


# ── save_periodic ─────────────────────────────────────────────────────────────


def test_save_periodic_creates_file(tmp_path) -> None:
    mgr = _make_manager(tmp_path, checkpoint_every=5)
    mgr.save_periodic(_dummy_state(5), epoch=5, train_loss=0.5)
    assert (tmp_path / "periodic_epoch5.pt").exists()


def test_save_periodic_skips_wrong_epoch(tmp_path) -> None:
    mgr = _make_manager(tmp_path, checkpoint_every=5)
    mgr.save_periodic(_dummy_state(3), epoch=3, train_loss=0.5)
    files = list(tmp_path.glob("periodic_epoch*.pt"))
    assert len(files) == 0


def test_save_periodic_cleans_old_files(tmp_path) -> None:
    mgr = _make_manager(tmp_path, checkpoint_every=5, max_periodic_kept=2)
    for epoch in [5, 10, 15]:
        mgr.save_periodic(_dummy_state(epoch), epoch=epoch, train_loss=0.5)
    files = {f.name for f in tmp_path.glob("periodic_epoch*.pt")}
    assert "periodic_epoch5.pt" not in files
    assert "periodic_epoch10.pt" in files
    assert "periodic_epoch15.pt" in files


# ── load ──────────────────────────────────────────────────────────────────────


def test_load_restores_state(tmp_path) -> None:
    encoder = nn.Linear(4, 4)
    model = nn.Linear(4, 4)

    known_enc_state = {k: torch.ones_like(v) for k, v in encoder.state_dict().items()}
    known_model_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}

    checkpoint = {
        "epoch": 10,
        "encoder_state": known_enc_state,
        "model_state": known_model_state,
        "best_val_loss": 0.42,
    }

    optimizer = torch.optim.AdamW(encoder.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    mgr = _make_manager(tmp_path)
    mgr.load(checkpoint, encoder, model, optimizer, scheduler)

    for key, val in encoder.state_dict().items():
        assert (val == known_enc_state[key]).all()
    for key, val in model.state_dict().items():
        assert (val == known_model_state[key]).all()


def test_load_returns_next_epoch(tmp_path) -> None:
    encoder = nn.Linear(4, 4)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(encoder.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    checkpoint = {
        "epoch": 10,
        "encoder_state": encoder.state_dict(),
        "model_state": model.state_dict(),
    }
    mgr = _make_manager(tmp_path)
    next_epoch = mgr.load(checkpoint, encoder, model, optimizer, scheduler)
    assert next_epoch == 11
