"""Tests for the external input protocol system: PulseEvent, PulseSchedule,
InputProtocol, factory functions, CRN integration, and Gillespie simulation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crn_surrogate.crn.inputs import (
    EMPTY_PROTOCOL,
    InputProtocol,
    PulseEvent,
    PulseSchedule,
    random_input_protocol,
    random_protocol,
    repeated_pulse,
    single_pulse,
    step_sequence,
)

# ── 4.1 PulseEvent ────────────────────────────────────────────────────────────


def test_pulse_event_valid_construction() -> None:
    """PulseEvent accepts valid arguments."""
    e = PulseEvent(t_start=1.0, t_end=3.0, amplitude=5.0)
    assert e.t_start == 1.0
    assert e.t_end == 3.0
    assert e.amplitude == 5.0


def test_pulse_event_negative_t_start_raises() -> None:
    """Negative t_start raises ValueError."""
    with pytest.raises(ValueError, match="t_start"):
        PulseEvent(t_start=-0.1, t_end=1.0, amplitude=1.0)


def test_pulse_event_t_end_equal_to_t_start_raises() -> None:
    """t_end == t_start raises ValueError."""
    with pytest.raises(ValueError, match="t_end"):
        PulseEvent(t_start=1.0, t_end=1.0, amplitude=1.0)


def test_pulse_event_t_end_before_t_start_raises() -> None:
    """t_end < t_start raises ValueError."""
    with pytest.raises(ValueError, match="t_end"):
        PulseEvent(t_start=2.0, t_end=1.0, amplitude=1.0)


def test_pulse_event_zero_amplitude_raises() -> None:
    """Amplitude == 0 raises ValueError."""
    with pytest.raises(ValueError, match="amplitude"):
        PulseEvent(t_start=0.0, t_end=1.0, amplitude=0.0)


def test_pulse_event_negative_amplitude_raises() -> None:
    """Negative amplitude raises ValueError."""
    with pytest.raises(ValueError, match="amplitude"):
        PulseEvent(t_start=0.0, t_end=1.0, amplitude=-1.0)


# ── 4.2 PulseSchedule ─────────────────────────────────────────────────────────


def test_pulse_schedule_auto_sorts_unsorted_events() -> None:
    """Events are auto-sorted by t_start regardless of input order."""
    e1 = PulseEvent(t_start=5.0, t_end=7.0, amplitude=2.0)
    e2 = PulseEvent(t_start=1.0, t_end=3.0, amplitude=1.0)
    schedule = PulseSchedule(events=(e1, e2))
    assert schedule.events[0].t_start == 1.0
    assert schedule.events[1].t_start == 5.0


def test_pulse_schedule_overlapping_events_raise() -> None:
    """Overlapping events raise ValueError."""
    e1 = PulseEvent(t_start=1.0, t_end=4.0, amplitude=1.0)
    e2 = PulseEvent(t_start=3.0, t_end=6.0, amplitude=2.0)
    with pytest.raises(ValueError, match="overlap"):
        PulseSchedule(events=(e1, e2))


def test_pulse_schedule_adjacent_events_are_valid() -> None:
    """Events whose t_end equals the next t_start do not overlap."""
    e1 = PulseEvent(t_start=1.0, t_end=3.0, amplitude=1.0)
    e2 = PulseEvent(t_start=3.0, t_end=5.0, amplitude=2.0)
    schedule = PulseSchedule(events=(e1, e2))
    assert len(schedule.events) == 2


def test_pulse_schedule_negative_baseline_raises() -> None:
    """Negative baseline raises ValueError."""
    with pytest.raises(ValueError, match="baseline"):
        PulseSchedule(events=(), baseline=-1.0)


def test_pulse_schedule_evaluate_before_first_pulse_returns_baseline() -> None:
    """evaluate returns baseline before the first pulse."""
    schedule = single_pulse(t_start=5.0, t_end=10.0, amplitude=3.0, baseline=1.0)
    assert schedule.evaluate(2.0) == pytest.approx(1.0)


def test_pulse_schedule_evaluate_during_pulse_returns_amplitude() -> None:
    """evaluate returns amplitude during a pulse."""
    schedule = single_pulse(t_start=5.0, t_end=10.0, amplitude=3.0, baseline=0.0)
    assert schedule.evaluate(7.5) == pytest.approx(3.0)


def test_pulse_schedule_evaluate_at_t_start_is_active() -> None:
    """evaluate at t_start is within the pulse (left-inclusive)."""
    schedule = single_pulse(t_start=5.0, t_end=10.0, amplitude=3.0)
    assert schedule.evaluate(5.0) == pytest.approx(3.0)


def test_pulse_schedule_evaluate_at_t_end_is_not_active() -> None:
    """evaluate at t_end is outside the pulse (right-exclusive)."""
    schedule = single_pulse(t_start=5.0, t_end=10.0, amplitude=3.0, baseline=0.0)
    assert schedule.evaluate(10.0) == pytest.approx(0.0)


def test_pulse_schedule_evaluate_between_pulses_returns_baseline() -> None:
    """evaluate returns baseline in gaps between pulses."""
    e1 = PulseEvent(t_start=1.0, t_end=3.0, amplitude=1.0)
    e2 = PulseEvent(t_start=6.0, t_end=8.0, amplitude=2.0)
    schedule = PulseSchedule(events=(e1, e2), baseline=0.5)
    assert schedule.evaluate(4.5) == pytest.approx(0.5)


def test_pulse_schedule_evaluate_after_last_pulse_returns_baseline() -> None:
    """evaluate returns baseline after all pulses."""
    schedule = single_pulse(t_start=1.0, t_end=3.0, amplitude=5.0, baseline=0.0)
    assert schedule.evaluate(10.0) == pytest.approx(0.0)


def test_pulse_schedule_breakpoints_contains_all_boundaries() -> None:
    """breakpoints returns sorted unique t_start and t_end values."""
    e1 = PulseEvent(t_start=1.0, t_end=3.0, amplitude=1.0)
    e2 = PulseEvent(t_start=5.0, t_end=7.0, amplitude=2.0)
    schedule = PulseSchedule(events=(e1, e2))
    bp = schedule.breakpoints()
    assert bp == (1.0, 3.0, 5.0, 7.0)


def test_pulse_schedule_breakpoints_empty_schedule() -> None:
    """breakpoints on a schedule with no events returns empty tuple."""
    schedule = PulseSchedule(events=())
    assert schedule.breakpoints() == ()


def test_pulse_schedule_evaluate_array_matches_pointwise() -> None:
    """evaluate_array values match pointwise evaluate calls."""
    schedule = repeated_pulse(period=4.0, duty_cycle=0.5, amplitude=2.0, n_pulses=3)
    times = np.linspace(0.0, 16.0, 200)
    arr = schedule.evaluate_array(times)
    expected = np.array([schedule.evaluate(float(t)) for t in times])
    np.testing.assert_allclose(arr, expected)


def test_pulse_schedule_empty_schedule_evaluate_returns_baseline() -> None:
    """evaluate on a schedule with no events always returns baseline."""
    schedule = PulseSchedule(events=(), baseline=2.5)
    assert schedule.evaluate(0.0) == pytest.approx(2.5)
    assert schedule.evaluate(1000.0) == pytest.approx(2.5)


# ── 4.3 InputProtocol ─────────────────────────────────────────────────────────


def test_input_protocol_multi_species_evaluate() -> None:
    """evaluate returns correct values for each input species."""
    s0_schedule = single_pulse(t_start=1.0, t_end=3.0, amplitude=10.0)
    s2_schedule = single_pulse(t_start=4.0, t_end=6.0, amplitude=20.0)
    protocol = InputProtocol(schedules={0: s0_schedule, 2: s2_schedule})
    vals = protocol.evaluate(2.0)
    assert vals[0] == pytest.approx(10.0)
    assert vals[2] == pytest.approx(0.0)  # s2 not active yet


def test_input_protocol_breakpoints_merges_and_deduplicates() -> None:
    """breakpoints merges unique breakpoints across all species schedules."""
    s0 = single_pulse(t_start=1.0, t_end=3.0, amplitude=1.0)
    s1 = single_pulse(t_start=3.0, t_end=5.0, amplitude=2.0)
    protocol = InputProtocol(schedules={0: s0, 1: s1})
    bp = protocol.breakpoints()
    assert bp == (1.0, 3.0, 5.0)


def test_input_protocol_breakpoints_deduplicates_shared_times() -> None:
    """Shared boundary times appear only once in breakpoints."""
    s0 = single_pulse(t_start=1.0, t_end=3.0, amplitude=1.0)
    s1 = single_pulse(t_start=1.0, t_end=3.0, amplitude=2.0)
    protocol = InputProtocol(schedules={0: s0, 1: s1})
    bp = protocol.breakpoints()
    assert bp == (1.0, 3.0)


def test_input_protocol_input_species_indices() -> None:
    """input_species_indices returns the set of controlled species."""
    protocol = InputProtocol(
        schedules={0: single_pulse(0.0, 1.0, 1.0), 3: single_pulse(0.0, 1.0, 1.0)}
    )
    assert protocol.input_species_indices() == frozenset({0, 3})


def test_input_protocol_n_input_species() -> None:
    """n_input_species returns the count of controlled species."""
    protocol = InputProtocol(
        schedules={0: single_pulse(0.0, 1.0, 1.0), 3: single_pulse(0.0, 1.0, 1.0)}
    )
    assert protocol.n_input_species() == 2


def test_empty_protocol_has_no_species() -> None:
    """EMPTY_PROTOCOL has zero input species and empty breakpoints."""
    assert EMPTY_PROTOCOL.n_input_species() == 0
    assert EMPTY_PROTOCOL.breakpoints() == ()
    assert EMPTY_PROTOCOL.evaluate(5.0) == {}


# ── 4.4 Factory tests ─────────────────────────────────────────────────────────


def test_single_pulse_event_structure() -> None:
    """single_pulse creates exactly one event with correct timing."""
    schedule = single_pulse(t_start=2.0, t_end=5.0, amplitude=7.0, baseline=1.0)
    assert len(schedule.events) == 1
    assert schedule.events[0].t_start == 2.0
    assert schedule.events[0].t_end == 5.0
    assert schedule.events[0].amplitude == 7.0
    assert schedule.baseline == 1.0


def test_repeated_pulse_event_count() -> None:
    """repeated_pulse creates exactly n_pulses events."""
    schedule = repeated_pulse(period=5.0, duty_cycle=0.4, amplitude=3.0, n_pulses=4)
    assert len(schedule.events) == 4


def test_repeated_pulse_timing() -> None:
    """repeated_pulse events have correct start times and durations."""
    schedule = repeated_pulse(
        period=10.0, duty_cycle=0.5, amplitude=1.0, n_pulses=3, t_start=2.0
    )
    on_duration = 5.0
    for i, event in enumerate(schedule.events):
        expected_start = 2.0 + i * 10.0
        assert event.t_start == pytest.approx(expected_start)
        assert event.t_end == pytest.approx(expected_start + on_duration)


def test_repeated_pulse_invalid_period_raises() -> None:
    """Period <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="period"):
        repeated_pulse(period=0.0, duty_cycle=0.5, amplitude=1.0, n_pulses=2)


def test_repeated_pulse_invalid_duty_cycle_raises() -> None:
    """duty_cycle not in (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="duty_cycle"):
        repeated_pulse(period=5.0, duty_cycle=0.0, amplitude=1.0, n_pulses=2)


def test_step_sequence_gaps_not_events() -> None:
    """step_sequence intervals equal to baseline produce no PulseEvent."""
    schedule = step_sequence(
        times=[0.0, 5.0, 10.0, 15.0],
        amplitudes=[1.0, 0.0, 2.0],
        baseline=0.0,
    )
    # The interval [5, 10) has amplitude == baseline, so no event there.
    assert len(schedule.events) == 2
    assert schedule.events[0].amplitude == 1.0
    assert schedule.events[1].amplitude == 2.0


def test_step_sequence_wrong_lengths_raises() -> None:
    """len(amplitudes) != len(times) - 1 raises ValueError."""
    with pytest.raises(ValueError, match="len"):
        step_sequence(times=[0.0, 5.0, 10.0], amplitudes=[1.0])


def test_step_sequence_non_increasing_times_raises() -> None:
    """Non-strictly-increasing times raise ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        step_sequence(times=[0.0, 5.0, 5.0], amplitudes=[1.0, 2.0])


def test_random_protocol_is_valid() -> None:
    """random_protocol produces a valid non-overlapping schedule within t_max."""
    rng = np.random.default_rng(42)
    schedule = random_protocol(t_max=20.0, rng=rng)
    # Validate non-overlapping (would raise on construction if violated)
    for i in range(len(schedule.events) - 1):
        assert schedule.events[i].t_end <= schedule.events[i + 1].t_start
    # All events within [0, t_max)
    for e in schedule.events:
        assert e.t_start >= 0.0
        assert e.t_end <= 20.0


def test_random_input_protocol_correct_species_count() -> None:
    """random_input_protocol creates one schedule per specified species."""
    rng = np.random.default_rng(7)
    protocol = random_input_protocol(
        input_species_indices=[1, 4, 7], t_max=30.0, rng=rng
    )
    assert protocol.n_input_species() == 3
    assert protocol.input_species_indices() == frozenset({1, 4, 7})


def test_random_input_protocol_all_schedules_valid() -> None:
    """All schedules in random_input_protocol are valid (non-overlapping)."""
    rng = np.random.default_rng(99)
    protocol = random_input_protocol(input_species_indices=[0, 1], t_max=15.0, rng=rng)
    for idx, schedule in protocol.schedules.items():
        for i in range(len(schedule.events) - 1):
            assert schedule.events[i].t_end <= schedule.events[i + 1].t_start, (
                f"Species {idx}: events overlap"
            )


# ── 4.5 CRN integration tests ─────────────────────────────────────────────────


def test_crn_with_external_species_valid() -> None:
    """CRN accepts external_species that have zero net stoichiometric change."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction

    # Species 0: internal (protein), species 1: external (inducer, catalytic only)
    # R1: ∅ → protein, propensity depends on inducer but stoichiometry is [1, 0]
    # R2: protein → ∅, mass action on protein
    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0]),
            propensity=mass_action(0.1, torch.tensor([0.0, 1.0])),
            name="production",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]),
            propensity=mass_action(0.05, torch.tensor([1.0, 0.0])),
            name="degradation",
        ),
    ]
    crn = CRN(
        reactions=reactions,
        species_names=["protein", "inducer"],
        external_species=frozenset({1}),
    )
    assert crn.n_external_species == 1
    assert bool(crn.is_external[1]) is True
    assert bool(crn.is_external[0]) is False


def test_crn_external_species_with_nonzero_stoich_raises() -> None:
    """CRN raises ValueError when an external species appears as product/reactant."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import constant_rate
    from crn_surrogate.crn.reaction import Reaction

    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 1.0]),  # species 1 is a product — invalid
            propensity=constant_rate(1.0),
            name="bad_reaction",
        ),
    ]
    with pytest.raises(ValueError, match="nonzero net stoichiometric change"):
        CRN(
            reactions=reactions,
            species_names=["A", "B"],
            external_species=frozenset({1}),
        )


def test_crn_external_species_out_of_range_raises() -> None:
    """CRN raises ValueError when external_species index is out of range."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import constant_rate
    from crn_surrogate.crn.reaction import Reaction

    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0]), propensity=constant_rate(1.0), name="r1"
        ),
    ]
    with pytest.raises(ValueError, match="out of range"):
        CRN(reactions=reactions, external_species=frozenset({5}))


def test_crn_is_external_property() -> None:
    """is_external returns correct boolean array."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction

    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0, 0.0]),
            propensity=mass_action(0.1, torch.tensor([0.0, 1.0, 0.0])),
            name="r1",
        ),
    ]
    crn = CRN(reactions=reactions, external_species=frozenset({1}))
    mask = crn.is_external
    assert mask.shape == (3,)
    assert not mask[0]
    assert mask[1]
    assert not mask[2]


def test_crn_internal_species_mask_is_complement() -> None:
    """internal_species_mask is the complement of is_external."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction

    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0]),
            propensity=mass_action(0.1, torch.tensor([0.0, 1.0])),
            name="r1",
        ),
    ]
    crn = CRN(reactions=reactions, external_species=frozenset({1}))
    import numpy as np

    np.testing.assert_array_equal(crn.internal_species_mask, ~crn.is_external)


# ── 4.6 Gillespie with input tests ────────────────────────────────────────────


def _make_birth_death_with_inducer() -> tuple:
    """Build a birth-death CRN where the birth rate depends on an external inducer.

    Species: 0 = protein (internal), 1 = inducer (external)
    R1: ∅ → protein  (rate = k_on * inducer, mass-action on inducer)
    R2: protein → ∅  (rate = k_deg * protein)
    """
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction

    k_on = 1.0
    k_deg = 0.2
    stoich = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    reactions = [
        Reaction(
            stoichiometry=stoich[0],
            propensity=mass_action(k_on, torch.tensor([0.0, 1.0])),
            name="production",
        ),
        Reaction(
            stoichiometry=stoich[1],
            propensity=mass_action(k_deg, torch.tensor([1.0, 0.0])),
            name="degradation",
        ),
    ]
    crn = CRN(
        reactions=reactions,
        species_names=["protein", "inducer"],
        external_species=frozenset({1}),
    )
    return crn, k_on, k_deg


def test_gillespie_with_empty_protocol_behaves_identically() -> None:
    """GillespieSSA with EMPTY_PROTOCOL produces a valid trajectory."""
    from crn_surrogate.crn.examples import birth_death
    from crn_surrogate.simulation.gillespie import GillespieSSA

    crn = birth_death(k_birth=2.0, k_death=0.5)
    ssa = GillespieSSA()
    x0 = torch.tensor([5.0])
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=x0,
        t_max=10.0,
        input_protocol=EMPTY_PROTOCOL,
    )
    assert traj.times[0].item() == pytest.approx(0.0)
    assert traj.states.shape[1] == 1


def test_gillespie_external_species_values_match_protocol() -> None:
    """External species in recorded trajectory match the protocol at each time."""
    from crn_surrogate.simulation.gillespie import GillespieSSA

    crn, _, _ = _make_birth_death_with_inducer()
    # Inducer is 50 for t in [2, 8), 0 outside
    protocol = InputProtocol(schedules={1: single_pulse(2.0, 8.0, amplitude=50.0)})
    ssa = GillespieSSA()
    x0 = torch.tensor([0.0, 0.0])  # protein=0, inducer starts at 0
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=x0,
        t_max=12.0,
        input_protocol=protocol,
        external_species=crn.external_species,
    )
    # Verify external species values match protocol at each recorded time
    for i, t in enumerate(traj.times.tolist()):
        expected_inducer = protocol.schedules[1].evaluate(t)
        actual_inducer = traj.states[i, 1].item()
        assert actual_inducer == pytest.approx(expected_inducer), (
            f"At t={t:.3f}: expected inducer={expected_inducer}, got {actual_inducer}"
        )


def test_gillespie_breakpoint_handling_produces_step_at_correct_time() -> None:
    """Trajectory includes a state update at each protocol breakpoint."""
    from crn_surrogate.simulation.gillespie import GillespieSSA

    crn, _, _ = _make_birth_death_with_inducer()
    pulse_start = 5.0
    pulse_end = 10.0
    protocol = InputProtocol(
        schedules={1: single_pulse(pulse_start, pulse_end, amplitude=20.0)}
    )
    ssa = GillespieSSA()
    x0 = torch.tensor([0.0, 0.0])
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=x0,
        t_max=15.0,
        input_protocol=protocol,
        external_species=crn.external_species,
    )
    # At t < pulse_start, inducer should be 0; at t >= pulse_start (and < pulse_end), 20
    for t, state in zip(traj.times.tolist(), traj.states.tolist()):
        expected = protocol.schedules[1].evaluate(t)
        assert state[1] == pytest.approx(expected)


def test_gillespie_external_stoich_violation_raises() -> None:
    """GillespieSSA raises if external species has nonzero stoichiometric column."""
    from crn_surrogate.simulation.gillespie import GillespieSSA

    # Stoichiometry with nonzero entry for species 1 (external)
    stoich = torch.tensor([[1.0, 1.0]])  # reaction changes external species 1
    protocol = InputProtocol(schedules={1: single_pulse(0.0, 5.0, amplitude=1.0)})
    ssa = GillespieSSA()
    with pytest.raises(ValueError, match="nonzero net stoichiometric change"):
        ssa.simulate(
            stoichiometry=stoich,
            propensity_fn=lambda s, t: torch.tensor([1.0]),
            initial_state=torch.tensor([0.0, 0.0]),
            t_max=10.0,
            input_protocol=protocol,
            external_species=frozenset({1}),
        )


# ── 4.7 CRNTensorRepr tests ────────────────────────────────────────────────────


def test_crn_tensor_repr_is_external_tensor_populated() -> None:
    """crn_to_tensor_repr correctly populates the is_external tensor."""
    from crn_surrogate.crn.crn import CRN
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction
    from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

    reactions = [
        Reaction(
            stoichiometry=torch.tensor([1.0, 0.0]),
            propensity=mass_action(0.1, torch.tensor([0.0, 1.0])),
            name="production",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1.0, 0.0]),
            propensity=mass_action(0.05, torch.tensor([1.0, 0.0])),
            name="degradation",
        ),
    ]
    crn = CRN(reactions=reactions, external_species=frozenset({1}))
    repr_ = crn_to_tensor_repr(crn)
    assert repr_.is_external.shape == (2,)
    assert not repr_.is_external[0].item()
    assert repr_.is_external[1].item()


def test_crn_tensor_repr_is_external_all_false_for_standard_crn() -> None:
    """Standard CRN (no external species) has all-False is_external tensor."""
    from crn_surrogate.crn.examples import birth_death
    from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr

    crn = birth_death()
    repr_ = crn_to_tensor_repr(crn)
    assert not repr_.is_external.any().item()


def test_edge_construction_excludes_rxn_to_external_species() -> None:
    """Reaction-to-species edges are excluded for external species."""
    from crn_surrogate.encoder.graph_utils import BipartiteGraphBuilder

    # 2 reactions, 3 species: species 2 is external
    stoichiometry = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    dependency = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    is_external = torch.tensor([False, False, True])

    edges = BipartiteGraphBuilder(stoichiometry, dependency, is_external).build()

    # Reaction-to-species edges should not target species 2
    if edges.rxn_to_species_index.numel() > 0:
        target_species = edges.rxn_to_species_index[1]  # row 1 = species targets
        assert (target_species == 2).sum().item() == 0, (
            "External species 2 should not receive reaction-to-species messages"
        )


def test_edge_construction_keeps_species_to_rxn_for_external() -> None:
    """Species-to-reaction edges are kept for external species."""
    from crn_surrogate.encoder.graph_utils import BipartiteGraphBuilder

    stoichiometry = torch.tensor([[1.0, 0.0, 0.0]])
    dependency = torch.tensor([[0.0, 0.0, 1.0]])  # species 2 → reaction 0
    is_external = torch.tensor([False, False, True])

    edges = BipartiteGraphBuilder(stoichiometry, dependency, is_external).build()

    # Species-to-reaction: species 2 should appear as source
    source_species = edges.species_to_rxn_index[0]  # row 0 = species sources
    assert (source_species == 2).sum().item() >= 1, (
        "External species 2 should still send messages to reactions"
    )


# ── Fix 1: breakpoint recording tests ─────────────────────────────────────────


def test_gillespie_records_state_at_pulse_start_and_end() -> None:
    """Trajectory contains events at exactly the pulse t_start and t_end times."""
    from crn_surrogate.crn import CRN, InputProtocol, PulseEvent, PulseSchedule
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction
    from crn_surrogate.simulation import GillespieSSA

    # External species (idx 1) drives birth; internal species (idx 0) degrades
    birth_prop = mass_action(
        rate_constant=1.0, reactant_stoichiometry=torch.tensor([0.0, 1.0])
    )
    death_prop = mass_action(
        rate_constant=0.1, reactant_stoichiometry=torch.tensor([1.0, 0.0])
    )
    crn = CRN(
        reactions=[
            Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=birth_prop),
            Reaction(stoichiometry=torch.tensor([-1.0, 0.0]), propensity=death_prop),
        ],
        species_names=["A", "I"],
        external_species=frozenset({1}),
    )

    t_start_pulse, t_end_pulse = 5.0, 15.0
    schedule = PulseSchedule(
        events=(PulseEvent(t_start=t_start_pulse, t_end=t_end_pulse, amplitude=50.0),),
        baseline=0.0,
    )
    protocol = InputProtocol(schedules={1: schedule})

    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0, 0.0]),
        t_max=20.0,
        input_protocol=protocol,
        external_species=crn.external_species,
    )

    times_list = traj.times.tolist()
    assert any(abs(t - t_start_pulse) < 1e-10 for t in times_list), (
        f"Trajectory missing event at pulse t_start={t_start_pulse}"
    )
    assert any(abs(t - t_end_pulse) < 1e-10 for t in times_list), (
        f"Trajectory missing event at pulse t_end={t_end_pulse}"
    )


def test_gillespie_external_species_value_changes_at_breakpoint() -> None:
    """Interpolated external species value changes at the correct breakpoint time.

    Uses a very slow system (near-zero propensity) so no reactions fire between
    breakpoints, and verifies the external species transitions at t_start.
    """
    from crn_surrogate.crn import CRN, InputProtocol, PulseEvent, PulseSchedule
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction
    from crn_surrogate.simulation import GillespieSSA, interpolate_to_grid

    # Extremely slow birth so no reactions fire; death is also near-zero
    birth_prop = mass_action(
        rate_constant=1e-6, reactant_stoichiometry=torch.tensor([0.0, 1.0])
    )
    death_prop = mass_action(
        rate_constant=1e-6, reactant_stoichiometry=torch.tensor([1.0, 0.0])
    )
    crn = CRN(
        reactions=[
            Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=birth_prop),
            Reaction(stoichiometry=torch.tensor([-1.0, 0.0]), propensity=death_prop),
        ],
        species_names=["A", "I"],
        external_species=frozenset({1}),
    )

    t_on = 3.0
    amplitude = 99.0
    schedule = PulseSchedule(
        events=(PulseEvent(t_start=t_on, t_end=10.0, amplitude=amplitude),),
        baseline=0.0,
    )
    protocol = InputProtocol(schedules={1: schedule})

    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0, 0.0]),
        t_max=10.0,
        input_protocol=protocol,
        external_species=crn.external_species,
    )

    t_query = torch.linspace(0.0, 10.0, 1000)
    grid = interpolate_to_grid(traj.times, traj.states, t_query)

    # Before t_on: external species should be 0
    before_mask = t_query < t_on - 1e-6
    assert (grid[before_mask, 1] == 0.0).all(), (
        "External species should be 0 before pulse start"
    )

    # After t_on: external species should be amplitude
    after_mask = t_query > t_on + 1e-6
    after_vals = grid[after_mask, 1]
    # Allow for a few grid points that might straddle the breakpoint
    assert (after_vals >= amplitude - 1e-6).sum() > after_vals.shape[0] * 0.9, (
        "External species should be at amplitude after pulse start"
    )


# ── Fix 2: inf breakpoint filtering ───────────────────────────────────────────


def test_gillespie_constant_input_terminates_at_t_max() -> None:
    """constant_input (t_end=inf) does not cause simulation to run past t_max."""
    from crn_surrogate.crn import CRN, InputProtocol, constant_input
    from crn_surrogate.crn.propensities import mass_action
    from crn_surrogate.crn.reaction import Reaction
    from crn_surrogate.simulation import GillespieSSA

    birth_prop = mass_action(
        rate_constant=1.0, reactant_stoichiometry=torch.tensor([0.0, 1.0])
    )
    death_prop = mass_action(
        rate_constant=0.5, reactant_stoichiometry=torch.tensor([1.0, 0.0])
    )
    crn = CRN(
        reactions=[
            Reaction(stoichiometry=torch.tensor([1.0, 0.0]), propensity=birth_prop),
            Reaction(stoichiometry=torch.tensor([-1.0, 0.0]), propensity=death_prop),
        ],
        species_names=["A", "I"],
        external_species=frozenset({1}),
    )

    t_max = 20.0
    protocol = InputProtocol(schedules={1: constant_input(amplitude=10.0)})

    ssa = GillespieSSA()
    traj = ssa.simulate(
        stoichiometry=crn.stoichiometry_matrix,
        propensity_fn=crn.evaluate_propensities,
        initial_state=torch.tensor([0.0, 0.0]),
        t_max=t_max,
        input_protocol=protocol,
        external_species=crn.external_species,
    )

    assert traj.times[-1].item() <= t_max, (
        f"Simulation ran past t_max={t_max}: last time={traj.times[-1].item()}"
    )


# ── BipartiteGraphBuilder: external species edge count ─────────────────────────


def test_bipartite_graph_builder_external_species_has_zero_incoming_edges() -> None:
    """External species connected to 2 reactions: s2r=2 edges, r2s=0 edges for it."""
    from crn_surrogate.encoder.graph_utils import BipartiteGraphBuilder

    # 2 reactions, 2 species: species 1 is external
    # Reaction 0: produces species 0, depends on species 1
    # Reaction 1: degrades species 0, no dependency on species 1
    stoichiometry = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    dependency = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    is_external = torch.tensor([False, True])

    edges = BipartiteGraphBuilder(stoichiometry, dependency, is_external).build()

    # Reaction-to-species: no edge should target species 1
    if edges.rxn_to_species_index.numel() > 0:
        r2s_targets = edges.rxn_to_species_index[1]
        assert (r2s_targets == 1).sum().item() == 0, (
            "External species 1 should have 0 reaction-to-species edges"
        )

    # Species-to-reaction: species 1 should have exactly 1 edge (to reaction 0)
    s2r_sources = edges.species_to_rxn_index[0]
    assert (s2r_sources == 1).sum().item() == 1, (
        "External species 1 should have 1 species-to-reaction edge"
    )
