"""External input protocol: pulsatile input schedules for pseudo-species in CRNs.

External inputs model experimental protocols (microfluidic ligand pulses,
optogenetic stimulation, etc.) as pseudo-species whose dynamics are prescribed
by a pulse schedule rather than CRN kinetics. The pulse schedule is
experiment-level, not CRN-level: the same CRN with different input protocols
produces the same CRNContext from the encoder.

Boundary convention: a pulse is active for ``t_start <= t < t_end``
(left-inclusive, right-exclusive).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    # Data structures
    "PulseEvent",
    "PulseSchedule",
    "InputProtocol",
    "EMPTY_PROTOCOL",
    # Factories
    "constant_input",
    "single_pulse",
    "repeated_pulse",
    "step_sequence",
    "random_protocol",
    "random_input_protocol",
]


# ── PulseEvent ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PulseEvent:
    """A single rectangular pulse active on [t_start, t_end).

    Attributes:
        t_start: Start time (inclusive). Must be >= 0.
        t_end: End time (exclusive). Must be > t_start.
        amplitude: Pulse amplitude. Must be > 0.
    """

    t_start: float
    t_end: float
    amplitude: float

    def __post_init__(self) -> None:
        if self.t_start < 0:
            raise ValueError(
                f"PulseEvent.t_start must be >= 0, got {self.t_start}"
            )
        if self.t_end <= self.t_start:
            raise ValueError(
                f"PulseEvent.t_end must be > t_start, "
                f"got t_start={self.t_start}, t_end={self.t_end}"
            )
        if self.amplitude <= 0:
            raise ValueError(
                f"PulseEvent.amplitude must be > 0 (zero amplitude is 'no input'), "
                f"got {self.amplitude}"
            )


# ── PulseSchedule ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PulseSchedule:
    """Pulse train for a single input species.

    Attributes:
        events: Sorted, non-overlapping sequence of PulseEvents.
        baseline: Value when no pulse is active. Must be >= 0.
    """

    events: tuple[PulseEvent, ...]
    baseline: float = 0.0

    def __post_init__(self) -> None:
        if self.baseline < 0:
            raise ValueError(
                f"PulseSchedule.baseline must be >= 0, got {self.baseline}"
            )
        # Auto-sort by t_start (frozen dataclass pattern: use object.__setattr__)
        sorted_events = tuple(sorted(self.events, key=lambda e: e.t_start))
        object.__setattr__(self, "events", sorted_events)
        # Validate non-overlapping
        for i in range(len(sorted_events) - 1):
            e_i = sorted_events[i]
            e_next = sorted_events[i + 1]
            if e_i.t_end > e_next.t_start:
                raise ValueError(
                    f"PulseEvents overlap: event {i} ends at {e_i.t_end} but "
                    f"event {i + 1} starts at {e_next.t_start}"
                )

    def evaluate(self, t: float) -> float:
        """Return the amplitude at time t, or baseline if no pulse is active.

        Boundary convention: pulse is active for t_start <= t < t_end.

        Uses binary search for O(log n) lookup.

        Args:
            t: Query time.

        Returns:
            Amplitude of the active pulse, or baseline.
        """
        if not self.events:
            return self.baseline
        # Find the last event whose t_start <= t
        starts = [e.t_start for e in self.events]
        idx = bisect.bisect_right(starts, t) - 1
        if idx >= 0 and self.events[idx].t_end > t:
            return self.events[idx].amplitude
        return self.baseline

    def breakpoints(self) -> tuple[float, ...]:
        """Return sorted unique t_start and t_end values across all events.

        Returns:
            Sorted tuple of breakpoint times.
        """
        pts: set[float] = set()
        for e in self.events:
            pts.add(e.t_start)
            pts.add(e.t_end)
        return tuple(sorted(pts))

    def evaluate_array(self, times: np.ndarray) -> np.ndarray:
        """Evaluate the schedule at an array of times.

        Args:
            times: Array of query times, any shape.

        Returns:
            Array of the same shape as times with schedule values.
        """
        result = np.full(times.shape, self.baseline, dtype=float)
        for event in self.events:
            mask = (times >= event.t_start) & (times < event.t_end)
            result[mask] = event.amplitude
        return result


# ── InputProtocol ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InputProtocol:
    """Pulse schedules for all external input species in a CRN.

    Attributes:
        schedules: Maps species index to its PulseSchedule.
    """

    schedules: dict[int, PulseSchedule]

    def input_species_indices(self) -> frozenset[int]:
        """Return the set of species indices that are externally controlled.

        Returns:
            frozenset of integer species indices.
        """
        return frozenset(self.schedules.keys())

    def n_input_species(self) -> int:
        """Number of externally controlled species.

        Returns:
            Integer count.
        """
        return len(self.schedules)

    def breakpoints(self) -> tuple[float, ...]:
        """Return merged sorted unique breakpoints across all schedules.

        Returns:
            Sorted tuple of breakpoint times.
        """
        pts: set[float] = set()
        for schedule in self.schedules.values():
            pts.update(schedule.breakpoints())
        return tuple(sorted(pts))

    def evaluate(self, t: float) -> dict[int, float]:
        """Return {species_index: value} for all input species at time t.

        Args:
            t: Query time.

        Returns:
            Dict mapping species index to its current value.
        """
        return {idx: schedule.evaluate(t) for idx, schedule in self.schedules.items()}

    def evaluate_array(self, times: np.ndarray) -> dict[int, np.ndarray]:
        """Return vectorized schedule values for all input species.

        Args:
            times: Array of query times.

        Returns:
            Dict mapping species index to array of values (same shape as times).
        """
        return {
            idx: schedule.evaluate_array(times)
            for idx, schedule in self.schedules.items()
        }


EMPTY_PROTOCOL: InputProtocol = InputProtocol(schedules={})
"""Sentinel for CRNs with no external inputs. Avoids None checks in callers."""


# ── Convenience factories ──────────────────────────────────────────────────────


def constant_input(
    amplitude: float,
    t_start: float = 0.0,
    t_end: float = float("inf"),
) -> PulseSchedule:
    """Constant-amplitude input from t_start to t_end.

    Implemented as a single long pulse. For truly infinite duration, use the
    default t_end=inf and rely on the simulator's t_max to truncate.

    Args:
        amplitude: Constant amplitude (must be > 0).
        t_start: Start time (default 0.0).
        t_end: End time (default inf).

    Returns:
        PulseSchedule with one pulse covering [t_start, t_end).
    """
    return PulseSchedule(events=(PulseEvent(t_start=t_start, t_end=t_end, amplitude=amplitude),))


def single_pulse(
    t_start: float,
    t_end: float,
    amplitude: float,
    baseline: float = 0.0,
) -> PulseSchedule:
    """Single rectangular pulse on [t_start, t_end).

    Args:
        t_start: Pulse start time.
        t_end: Pulse end time (exclusive).
        amplitude: Pulse amplitude (must be > 0).
        baseline: Value outside the pulse (default 0.0).

    Returns:
        PulseSchedule with one event.
    """
    return PulseSchedule(
        events=(PulseEvent(t_start=t_start, t_end=t_end, amplitude=amplitude),),
        baseline=baseline,
    )


def repeated_pulse(
    period: float,
    duty_cycle: float,
    amplitude: float,
    n_pulses: int,
    t_start: float = 0.0,
    baseline: float = 0.0,
) -> PulseSchedule:
    """Periodic square wave: n_pulses repetitions of a rectangular pulse.

    Each pulse has duration period * duty_cycle; the gap between pulses is
    period * (1 - duty_cycle).

    Args:
        period: Duration of one full on+off cycle (must be > 0).
        duty_cycle: Fraction of period that is "on", in (0, 1).
        amplitude: Pulse amplitude (must be > 0).
        n_pulses: Number of repetitions (must be >= 1).
        t_start: Start time of the first pulse (default 0.0).
        baseline: Value during "off" periods (default 0.0).

    Returns:
        PulseSchedule with n_pulses non-overlapping events.

    Raises:
        ValueError: If period <= 0, duty_cycle not in (0, 1), or n_pulses < 1.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if not (0 < duty_cycle < 1):
        raise ValueError(f"duty_cycle must be in (0, 1), got {duty_cycle}")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")

    on_duration = period * duty_cycle
    events: list[PulseEvent] = []
    for i in range(n_pulses):
        pulse_start = t_start + i * period
        pulse_end = pulse_start + on_duration
        events.append(PulseEvent(t_start=pulse_start, t_end=pulse_end, amplitude=amplitude))
    return PulseSchedule(events=tuple(events), baseline=baseline)


def step_sequence(
    times: Sequence[float],
    amplitudes: Sequence[float],
    baseline: float = 0.0,
) -> PulseSchedule:
    """Piecewise-constant signal from a sequence of transition times and amplitudes.

    Interval [times[i], times[i+1]) has amplitude amplitudes[i].
    Intervals where amplitude equals baseline produce no PulseEvent.

    Args:
        times: Transition times [t0, t1, t2, ...]. Must be strictly increasing,
            len >= 2.
        amplitudes: Amplitude for each interval. len(amplitudes) == len(times) - 1.
        baseline: Value outside the defined intervals (default 0.0).

    Returns:
        PulseSchedule encoding the step sequence.

    Raises:
        ValueError: If len(amplitudes) != len(times) - 1, or times not increasing.
    """
    times_list = list(times)
    amplitudes_list = list(amplitudes)
    if len(amplitudes_list) != len(times_list) - 1:
        raise ValueError(
            f"len(amplitudes) must equal len(times) - 1, "
            f"got {len(amplitudes_list)} amplitudes and {len(times_list)} times"
        )
    for i in range(len(times_list) - 1):
        if times_list[i + 1] <= times_list[i]:
            raise ValueError(
                f"times must be strictly increasing: times[{i}]={times_list[i]}, "
                f"times[{i+1}]={times_list[i+1]}"
            )
    events: list[PulseEvent] = []
    for i, amp in enumerate(amplitudes_list):
        if amp != baseline and amp > 0:
            events.append(
                PulseEvent(
                    t_start=times_list[i],
                    t_end=times_list[i + 1],
                    amplitude=amp,
                )
            )
    return PulseSchedule(events=tuple(events), baseline=baseline)


def random_protocol(
    t_max: float,
    n_pulses_range: tuple[int, int] = (1, 5),
    duration_range: tuple[float, float] = (0.5, 5.0),
    amplitude_range: tuple[float, float] = (1.0, 100.0),
    gap_range: tuple[float, float] = (0.5, 3.0),
    baseline: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PulseSchedule:
    """Sample a random pulse train for training data generation.

    Generates n_pulses (drawn uniformly from n_pulses_range) pulses placed
    sequentially in time. Duration is drawn uniformly; amplitude is log-uniform;
    inter-pulse gap is drawn uniformly. Pulses that would exceed t_max are
    truncated. Truncated pulses with zero or negative duration are dropped.

    Args:
        t_max: Maximum time. Pulses are truncated at t_max.
        n_pulses_range: (min, max) inclusive range for number of pulses.
        duration_range: (min, max) range for pulse duration.
        amplitude_range: (min, max) range for pulse amplitude (log-uniform).
        gap_range: (min, max) range for gap before each pulse.
        baseline: Value during gaps (default 0.0).
        rng: NumPy random generator (default: new default_rng()).

    Returns:
        PulseSchedule with non-overlapping pulses within [0, t_max).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_pulses = int(rng.integers(n_pulses_range[0], n_pulses_range[1] + 1))
    # First pulse start: uniform in [0, t_max / 4]
    t_current = float(rng.uniform(0.0, t_max / 4))

    events: list[PulseEvent] = []
    for _ in range(n_pulses):
        duration = float(rng.uniform(duration_range[0], duration_range[1]))
        amplitude = float(
            np.exp(rng.uniform(np.log(amplitude_range[0]), np.log(amplitude_range[1])))
        )
        t_start = t_current
        t_end = min(t_start + duration, t_max)
        if t_end > t_start:
            events.append(PulseEvent(t_start=t_start, t_end=t_end, amplitude=amplitude))
        t_current = t_end + float(rng.uniform(gap_range[0], gap_range[1]))
        if t_current >= t_max:
            break

    return PulseSchedule(events=tuple(events), baseline=baseline)


def random_input_protocol(
    input_species_indices: Sequence[int],
    t_max: float,
    n_pulses_range: tuple[int, int] = (1, 5),
    duration_range: tuple[float, float] = (0.5, 5.0),
    amplitude_range: tuple[float, float] = (1.0, 100.0),
    gap_range: tuple[float, float] = (0.5, 3.0),
    baseline: float = 0.0,
    rng: np.random.Generator | None = None,
) -> InputProtocol:
    """Generate independent random PulseSchedules for each input species.

    Args:
        input_species_indices: Species indices to generate schedules for.
        t_max: Maximum time.
        n_pulses_range: (min, max) inclusive range for number of pulses.
        duration_range: (min, max) range for pulse duration.
        amplitude_range: (min, max) range for amplitude (log-uniform).
        gap_range: (min, max) range for inter-pulse gap.
        baseline: Value during gaps (default 0.0).
        rng: NumPy random generator (shared across all species).

    Returns:
        InputProtocol with one independent schedule per species.
    """
    if rng is None:
        rng = np.random.default_rng()

    schedules = {
        idx: random_protocol(
            t_max=t_max,
            n_pulses_range=n_pulses_range,
            duration_range=duration_range,
            amplitude_range=amplitude_range,
            gap_range=gap_range,
            baseline=baseline,
            rng=rng,
        )
        for idx in input_species_indices
    }
    return InputProtocol(schedules=schedules)
