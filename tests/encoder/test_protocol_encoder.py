"""Tests for ProtocolEncoder."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import ProtocolEncoderConfig
from crn_surrogate.crn.inputs import (
    EMPTY_PROTOCOL,
    InputProtocol,
    PulseEvent,
    PulseSchedule,
)
from crn_surrogate.encoder.protocol_encoder import ProtocolEncoder


def _make_config() -> ProtocolEncoderConfig:
    return ProtocolEncoderConfig(d_event=32, d_protocol=64, n_layers=2)


def _make_single_pulse_protocol(amplitude: float = 1.0) -> InputProtocol:
    event = PulseEvent(t_start=0.0, t_end=1.0, amplitude=amplitude)
    schedule = PulseSchedule(events=[event], baseline=0.0)
    return InputProtocol(schedules={0: schedule})


def test_empty_protocol_produces_zeros() -> None:
    enc = ProtocolEncoder(_make_config())
    out = enc([EMPTY_PROTOCOL])
    assert out.shape == (1, _make_config().d_protocol)
    assert torch.allclose(out, torch.zeros_like(out))


def test_single_pulse_output_shape() -> None:
    enc = ProtocolEncoder(_make_config())
    protocol = _make_single_pulse_protocol()
    out = enc([protocol])
    assert out.shape == (1, _make_config().d_protocol)


def test_batch_output_shape() -> None:
    enc = ProtocolEncoder(_make_config())
    protocols = [_make_single_pulse_protocol() for _ in range(4)]
    out = enc(protocols)
    assert out.shape == (4, _make_config().d_protocol)


def test_nonempty_produces_nonzero() -> None:
    enc = ProtocolEncoder(_make_config())
    protocol = _make_single_pulse_protocol()
    out = enc([protocol])
    assert out.abs().sum().item() > 0.0


def test_different_protocols_different_embeddings() -> None:
    enc = ProtocolEncoder(_make_config())
    p1 = _make_single_pulse_protocol(amplitude=1.0)
    p2 = _make_single_pulse_protocol(amplitude=5.0)
    with torch.no_grad():
        out1 = enc([p1])
        out2 = enc([p2])
    assert not torch.allclose(out1, out2)
