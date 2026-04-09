"""Tests for PulseEvent, PulseSchedule, InputProtocol, and factory functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crn_surrogate.crn.inputs import (
    EMPTY_PROTOCOL,
    InputProtocol,
    PulseEvent,
    PulseSchedule,
    ResolvedProtocol,
    repeated_pulse,
    single_pulse,
    step_sequence,
)

# ── PulseEvent ────────────────────────────────────────────────────────────────


def test_pulse_event_valid():
    e = PulseEvent(t_start=1.0, t_end=2.0, amplitude=5.0)
    assert e.t_start == 1.0
    assert e.t_end == 2.0
    assert e.amplitude == 5.0


def test_pulse_event_rejects_negative_start():
    with pytest.raises(ValueError):
        PulseEvent(t_start=-1.0, t_end=1.0, amplitude=5.0)


def test_pulse_event_rejects_end_before_start():
    with pytest.raises(ValueError):
        PulseEvent(t_start=2.0, t_end=1.0, amplitude=5.0)


def test_pulse_event_rejects_zero_amplitude():
    with pytest.raises(ValueError):
        PulseEvent(t_start=0.0, t_end=1.0, amplitude=0.0)


# ── PulseSchedule ─────────────────────────────────────────────────────────────


def _simple_schedule() -> PulseSchedule:
    return PulseSchedule(
        events=(PulseEvent(t_start=1.0, t_end=3.0, amplitude=10.0),),
        baseline=0.0,
    )


def test_schedule_evaluate_during_pulse():
    s = _simple_schedule()
    assert s.evaluate(2.0) == pytest.approx(10.0)


def test_schedule_evaluate_outside_pulse():
    s = _simple_schedule()
    assert s.evaluate(0.5) == pytest.approx(0.0)


def test_schedule_evaluate_at_boundary():
    s = _simple_schedule()
    assert s.evaluate(1.0) == pytest.approx(10.0)  # t_start inclusive
    assert s.evaluate(3.0) == pytest.approx(0.0)  # t_end exclusive


def test_schedule_auto_sorts():
    e1 = PulseEvent(t_start=5.0, t_end=6.0, amplitude=1.0)
    e2 = PulseEvent(t_start=1.0, t_end=2.0, amplitude=2.0)
    s = PulseSchedule(events=(e1, e2))
    assert s.events[0].t_start == 1.0
    assert s.events[1].t_start == 5.0


def test_schedule_rejects_overlapping():
    e1 = PulseEvent(t_start=1.0, t_end=4.0, amplitude=1.0)
    e2 = PulseEvent(t_start=3.0, t_end=5.0, amplitude=1.0)
    with pytest.raises(ValueError):
        PulseSchedule(events=(e1, e2))


def test_schedule_breakpoints():
    e1 = PulseEvent(t_start=1.0, t_end=3.0, amplitude=1.0)
    e2 = PulseEvent(t_start=5.0, t_end=7.0, amplitude=1.0)
    s = PulseSchedule(events=(e1, e2))
    assert s.breakpoints() == (1.0, 3.0, 5.0, 7.0)


def test_schedule_evaluate_array():
    s = _simple_schedule()
    times = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    result = s.evaluate_array(times)
    expected = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


# ── InputProtocol ─────────────────────────────────────────────────────────────


def _two_species_protocol() -> InputProtocol:
    return InputProtocol(
        schedules={
            0: single_pulse(1.0, 3.0, amplitude=10.0),
            2: PulseSchedule(
                events=(PulseEvent(t_start=0.0, t_end=100.0, amplitude=5.0),)
            ),
        }
    )


def test_protocol_evaluate():
    protocol = _two_species_protocol()
    values = protocol.evaluate(2.0)  # inside pulse for species 0
    assert values[0] == pytest.approx(10.0)
    assert values[2] == pytest.approx(5.0)


def test_protocol_input_species_indices():
    protocol = _two_species_protocol()
    assert protocol.input_species_indices() == frozenset({0, 2})


def test_protocol_breakpoints_merged():
    s1 = single_pulse(1.0, 3.0, amplitude=1.0)
    s2 = single_pulse(5.0, 7.0, amplitude=1.0)
    protocol = InputProtocol(schedules={0: s1, 1: s2})
    bps = protocol.breakpoints()
    assert 1.0 in bps
    assert 3.0 in bps
    assert 5.0 in bps
    assert 7.0 in bps


def test_empty_protocol():
    values = EMPTY_PROTOCOL.evaluate(5.0)
    assert values == {}
    assert EMPTY_PROTOCOL.breakpoints() == ()


# ── Factories ─────────────────────────────────────────────────────────────────


def test_single_pulse_factory():
    s = single_pulse(1.0, 3.0, amplitude=10.0)
    assert s.evaluate(2.0) == pytest.approx(10.0)
    assert s.evaluate(0.5) == pytest.approx(0.0)
    assert s.evaluate(4.0) == pytest.approx(0.0)


def test_repeated_pulse_factory():
    s = repeated_pulse(period=4.0, duty_cycle=0.5, amplitude=5.0, n_pulses=3)
    assert len(s.events) == 3
    # First pulse: [0, 2), second: [4, 6), third: [8, 10)
    assert s.events[0].t_start == pytest.approx(0.0)
    assert s.events[0].t_end == pytest.approx(2.0)
    assert s.events[2].t_start == pytest.approx(8.0)


def test_repeated_pulse_rejects_bad_duty_cycle():
    with pytest.raises(ValueError):
        repeated_pulse(period=4.0, duty_cycle=0.0, amplitude=1.0, n_pulses=2)
    with pytest.raises(ValueError):
        repeated_pulse(period=4.0, duty_cycle=1.5, amplitude=1.0, n_pulses=2)


def test_step_sequence_factory():
    s = step_sequence(times=[0.0, 1.0, 2.0, 3.0], amplitudes=[5.0, 0.0, 10.0])
    assert s.evaluate(0.5) == pytest.approx(5.0)
    assert s.evaluate(1.5) == pytest.approx(0.0)  # baseline
    assert s.evaluate(2.5) == pytest.approx(10.0)


def test_step_sequence_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        step_sequence(times=[0.0, 1.0, 2.0, 3.0], amplitudes=[5.0, 10.0])


# ── ResolvedProtocol ──────────────────────────────────────────────────────────


def test_resolved_protocol_construction():
    protocol = EMPTY_PROTOCOL
    embedding = torch.zeros(8)
    mask = torch.zeros(3, dtype=torch.bool)
    resolved = ResolvedProtocol(
        protocol=protocol,
        embedding=embedding,
        external_species_mask=mask,
    )
    assert resolved.protocol is protocol
    assert resolved.embedding is embedding
    assert resolved.external_species_mask is mask
