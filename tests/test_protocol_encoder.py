"""Tests for ProtocolEncoder and ProtocolEncoderConfig.

Covers:
- EMPTY_PROTOCOL encodes to zeros.
- Single-species protocol with events encodes to nonzero vector.
- Batch of protocols with different event counts produces correct shape.
- Permutation invariance: reordered events give identical output.
- Batch with mixed empty/non-empty: empty items give zeros, non-empty give nonzero.
- SpeciesEmbedding is_external flag: shape preserved, values differ.
"""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.configs.model_config import ProtocolEncoderConfig
from crn_surrogate.crn.inputs import (
    EMPTY_PROTOCOL,
    InputProtocol,
    PulseEvent,
    PulseSchedule,
)
from crn_surrogate.encoder.protocol_encoder import ProtocolEncoder

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def cfg() -> ProtocolEncoderConfig:
    return ProtocolEncoderConfig(d_event=16, d_protocol=32, n_layers=2)


@pytest.fixture()
def encoder(cfg: ProtocolEncoderConfig) -> ProtocolEncoder:
    return ProtocolEncoder(cfg)


def _make_protocol_one_species(n_pulses: int = 3) -> InputProtocol:
    events = tuple(
        PulseEvent(
            t_start=float(i * 10), t_end=float(i * 10 + 5), amplitude=float(i + 1) * 10
        )
        for i in range(n_pulses)
    )
    return InputProtocol(schedules={0: PulseSchedule(events=events)})


# ── EMPTY_PROTOCOL ─────────────────────────────────────────────────────────────


def test_protocol_encoder_empty_protocol_returns_zeros(
    encoder: ProtocolEncoder,
) -> None:
    """Encoding EMPTY_PROTOCOL must produce a zero vector."""
    out = encoder([EMPTY_PROTOCOL])
    assert out.shape == (1, encoder._config.d_protocol)
    assert out.abs().max().item() == 0.0


# ── Single protocol ───────────────────────────────────────────────────────────


def test_protocol_encoder_single_protocol_nonzero(encoder: ProtocolEncoder) -> None:
    """A non-empty protocol must produce a nonzero embedding."""
    protocol = _make_protocol_one_species(n_pulses=3)
    out = encoder([protocol])
    assert out.shape == (1, encoder._config.d_protocol)
    assert out.abs().max().item() > 0.0


def test_protocol_encoder_output_shape_single(encoder: ProtocolEncoder) -> None:
    protocol = _make_protocol_one_species(n_pulses=1)
    out = encoder([protocol])
    assert out.shape == (1, encoder._config.d_protocol)


# ── Batch of protocols ─────────────────────────────────────────────────────────


def test_protocol_encoder_batch_output_shape(encoder: ProtocolEncoder) -> None:
    """Batch of 4 protocols with different event counts must produce (4, d_protocol)."""
    protocols = [
        _make_protocol_one_species(n_pulses=1),
        _make_protocol_one_species(n_pulses=3),
        _make_protocol_one_species(n_pulses=5),
        EMPTY_PROTOCOL,
    ]
    out = encoder(protocols)
    assert out.shape == (4, encoder._config.d_protocol)


# ── Permutation invariance ────────────────────────────────────────────────────


def test_protocol_encoder_permutation_invariance(encoder: ProtocolEncoder) -> None:
    """Reordering events within a protocol must not change the output."""
    e0 = PulseEvent(t_start=0.0, t_end=5.0, amplitude=10.0)
    e1 = PulseEvent(t_start=10.0, t_end=15.0, amplitude=30.0)
    e2 = PulseEvent(t_start=20.0, t_end=25.0, amplitude=20.0)

    # PulseSchedule auto-sorts by t_start, so both constructions produce the same
    # sorted order and therefore the same output.
    sched_abc = PulseSchedule(events=(e0, e1, e2))
    sched_bca = PulseSchedule(events=(e1, e2, e0))  # will be sorted to same order

    p_abc = InputProtocol(schedules={0: sched_abc})
    p_bca = InputProtocol(schedules={0: sched_bca})

    out_abc = encoder([p_abc])
    out_bca = encoder([p_bca])

    torch.testing.assert_close(out_abc, out_bca, atol=1e-6, rtol=0.0)


# ── Mixed empty/non-empty batch ───────────────────────────────────────────────


def test_protocol_encoder_mixed_batch_empty_items_are_zero(
    encoder: ProtocolEncoder,
) -> None:
    """In a mixed batch, empty-protocol items produce zero, non-empty produce nonzero."""
    protocols = [
        _make_protocol_one_species(n_pulses=2),
        EMPTY_PROTOCOL,
        _make_protocol_one_species(n_pulses=1),
        EMPTY_PROTOCOL,
    ]
    out = encoder(protocols)
    assert out.shape == (4, encoder._config.d_protocol)

    # Non-empty items (indices 0, 2)
    assert out[0].abs().max().item() > 0.0
    assert out[2].abs().max().item() > 0.0
    # Empty items (indices 1, 3)
    assert out[1].abs().max().item() == 0.0
    assert out[3].abs().max().item() == 0.0


# ── SpeciesEmbedding is_external ──────────────────────────────────────────────


def test_species_embedding_with_is_external_preserves_shape() -> None:
    """SpeciesEmbedding.forward with is_external=None must produce same shape."""
    from crn_surrogate.configs.model_config import EncoderConfig
    from crn_surrogate.encoder.embeddings import SpeciesEmbedding

    cfg = EncoderConfig(d_model=16, max_species=8)
    embed = SpeciesEmbedding(cfg)

    out_none = embed(n_species=3, is_external=None)
    out_flag = embed(n_species=3, is_external=torch.zeros(3, dtype=torch.bool))

    assert out_none.shape == (3, 16)
    assert out_flag.shape == (3, 16)


def test_species_embedding_external_flag_changes_values() -> None:
    """Marking one species external must produce different embeddings."""
    from crn_surrogate.configs.model_config import EncoderConfig
    from crn_surrogate.encoder.embeddings import SpeciesEmbedding

    cfg = EncoderConfig(d_model=16, max_species=8)
    embed = SpeciesEmbedding(cfg)

    is_external = torch.tensor([False, True])
    out_with = embed(n_species=2, is_external=is_external)
    out_without = embed(n_species=2, is_external=None)

    # The external-species row should differ.
    assert not torch.allclose(out_with[1], out_without[1])
    # The all-False flag for internal species is equivalent to is_external=None
    # only if _external_proj(0) == 0, which is not guaranteed with random weights.
    # So we just check shape here.
    assert out_with.shape == out_without.shape
