"""Tests for CRN motif factories: structure, stoichiometry, and parameter ranges."""

from __future__ import annotations

import pytest
import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.configs import SamplingConfig
from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.auto_catalysis import (
    AutoCatalysisFactory,
    AutoCatalysisParams,
)
from crn_surrogate.data.generation.motifs.base import (
    InitialStateRange,
    MotifFactory,
    MotifParams,
    ParameterRange,
    extract_parameter_ranges,
)
from crn_surrogate.data.generation.motifs.birth_death import (
    BirthDeathFactory,
    BirthDeathParams,
)
from crn_surrogate.data.generation.motifs.enzymatic_catalysis import (
    EnzymaticCatalysisFactory,
    EnzymaticCatalysisParams,
)
from crn_surrogate.data.generation.motifs.feedforward_loop import (
    IncoherentFeedforwardFactory,
    IncoherentFeedforwardParams,
)
from crn_surrogate.data.generation.motifs.negative_autoregulation import (
    NegativeAutoregulationFactory,
)
from crn_surrogate.data.generation.motifs.repressilator import (
    RepressilatorFactory,
    RepressilatorParams,
)
from crn_surrogate.data.generation.motifs.substrate_inhibition_motif import (
    SubstrateInhibitionMotifFactory,
    SubstrateInhibitionParams,
)
from crn_surrogate.data.generation.motifs.toggle_switch import (
    ToggleSwitchFactory,
    ToggleSwitchParams,
)
from crn_surrogate.data.generation.parameter_sampling import ParameterSampler


def _min_params(factory: MotifFactory) -> MotifParams:
    """Build a typed params instance at the lower bound of each parameter range.

    Only valid for factories with no cross-parameter constraints (e.g. k_deg > k_cat).
    For constrained factories use _valid_params() instead.
    """
    ranges = extract_parameter_ranges(factory.params_type)
    return factory.params_from_dict({name: r.low for name, r in ranges.items()})


def _max_params(factory: MotifFactory) -> MotifParams:
    """Build a typed params instance at the upper bound of each parameter range.

    Only valid for factories with no cross-parameter constraints (e.g. K_i > K_m).
    For constrained factories use _valid_params() instead.
    """
    ranges = extract_parameter_ranges(factory.params_type)
    return factory.params_from_dict({name: r.high for name, r in ranges.items()})


def _valid_params(factory: MotifFactory) -> MotifParams:
    """Sample one valid params instance for the factory.

    Uses the parameter sampler with a fixed seed, so constraints are respected
    via rejection sampling. Works for all factories including those with
    cross-parameter constraints (AutoCatalysis, SubstrateInhibition).
    """
    sampler = ParameterSampler(SamplingConfig(random_seed=0))
    return sampler.sample(factory, n_samples=1)[0]


# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def birth_death() -> BirthDeathFactory:
    """BirthDeathFactory instance."""
    return BirthDeathFactory()


@pytest.fixture
def auto_catalysis() -> AutoCatalysisFactory:
    """AutoCatalysisFactory instance."""
    return AutoCatalysisFactory()


@pytest.fixture
def negative_autoregulation() -> NegativeAutoregulationFactory:
    """NegativeAutoregulationFactory instance."""
    return NegativeAutoregulationFactory()


@pytest.fixture
def toggle_switch() -> ToggleSwitchFactory:
    """ToggleSwitchFactory instance."""
    return ToggleSwitchFactory()


@pytest.fixture
def enzymatic_catalysis() -> EnzymaticCatalysisFactory:
    """EnzymaticCatalysisFactory instance."""
    return EnzymaticCatalysisFactory()


@pytest.fixture
def feedforward_loop() -> IncoherentFeedforwardFactory:
    """IncoherentFeedforwardFactory instance."""
    return IncoherentFeedforwardFactory()


@pytest.fixture
def repressilator() -> RepressilatorFactory:
    """RepressilatorFactory instance."""
    return RepressilatorFactory()


@pytest.fixture
def substrate_inhibition() -> SubstrateInhibitionMotifFactory:
    """SubstrateInhibitionMotifFactory instance."""
    return SubstrateInhibitionMotifFactory()


ALL_FACTORIES = [
    pytest.param(BirthDeathFactory, id="birth_death"),
    pytest.param(AutoCatalysisFactory, id="auto_catalysis"),
    pytest.param(NegativeAutoregulationFactory, id="negative_autoregulation"),
    pytest.param(ToggleSwitchFactory, id="toggle_switch"),
    pytest.param(EnzymaticCatalysisFactory, id="enzymatic_catalysis"),
    pytest.param(IncoherentFeedforwardFactory, id="incoherent_feedforward"),
    pytest.param(RepressilatorFactory, id="repressilator"),
    pytest.param(SubstrateInhibitionMotifFactory, id="substrate_inhibition"),
]

# Factories with no cross-parameter constraints: boundary params are always valid.
# AutoCatalysis (k_deg > k_cat) and SubstrateInhibition (K_i > K_m) are excluded
# because at both min and max boundaries both constrained params are equal, violating
# the strict inequality.
UNCONSTRAINED_FACTORIES = [
    pytest.param(BirthDeathFactory, id="birth_death"),
    pytest.param(NegativeAutoregulationFactory, id="negative_autoregulation"),
    pytest.param(ToggleSwitchFactory, id="toggle_switch"),
    pytest.param(EnzymaticCatalysisFactory, id="enzymatic_catalysis"),
    pytest.param(IncoherentFeedforwardFactory, id="incoherent_feedforward"),
    pytest.param(RepressilatorFactory, id="repressilator"),
]


# --- Structural tests (all factories) -----------------------------------
# These use _valid_params so they work for constrained factories too.


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_create_returns_crn(factory_cls: type[MotifFactory]) -> None:
    """create() with valid params returns a CRN instance."""
    factory = factory_cls()
    crn = factory.create(_valid_params(factory))
    assert isinstance(crn, CRN)


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_n_species_consistent(factory_cls: type[MotifFactory]) -> None:
    """CRN n_species matches factory.n_species and stoichiometry matrix column count."""
    factory = factory_cls()
    crn = factory.create(_valid_params(factory))
    assert crn.n_species == factory.n_species
    assert crn.stoichiometry_matrix.shape[1] == factory.n_species


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_n_reactions_consistent(factory_cls: type[MotifFactory]) -> None:
    """CRN n_reactions matches factory.n_reactions and stoichiometry matrix row count."""
    factory = factory_cls()
    crn = factory.create(_valid_params(factory))
    assert crn.n_reactions == factory.n_reactions
    assert crn.stoichiometry_matrix.shape[0] == factory.n_reactions


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_stoichiometry_shape(factory_cls: type[MotifFactory]) -> None:
    """Stoichiometry matrix has shape (n_reactions, n_species)."""
    factory = factory_cls()
    crn = factory.create(_valid_params(factory))
    assert crn.stoichiometry_matrix.shape == torch.Size(
        (factory.n_reactions, factory.n_species)
    )


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_propensities_non_negative(factory_cls: type[MotifFactory]) -> None:
    """All propensities are non-negative for a typical positive state."""
    factory = factory_cls()
    crn = factory.create(_valid_params(factory))
    state = torch.ones(crn.n_species) * 10.0
    propensities = crn.evaluate_propensities(state)
    assert (propensities >= 0.0).all(), f"Negative propensity in {factory_cls.__name__}"


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_parameter_ranges_non_empty(factory_cls: type[MotifFactory]) -> None:
    """extract_parameter_ranges() returns a non-empty dict of ParameterRange values."""
    factory = factory_cls()
    ranges = extract_parameter_ranges(factory.params_type)
    assert isinstance(ranges, dict)
    assert len(ranges) > 0
    assert all(isinstance(r, ParameterRange) for r in ranges.values())


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_initial_state_ranges_keys_match_species(
    factory_cls: type[MotifFactory],
) -> None:
    """initial_state_ranges() keys match species_names."""
    factory = factory_cls()
    state_ranges = factory.initial_state_ranges()
    assert isinstance(state_ranges, dict)
    assert all(isinstance(r, InitialStateRange) for r in state_ranges.values())
    assert set(state_ranges.keys()) == set(factory.species_names)


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_motif_type_is_enum(factory_cls: type[MotifFactory]) -> None:
    """motif_type is a MotifType enum value."""
    factory = factory_cls()
    assert isinstance(factory.motif_type, MotifType)


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_params_type_matches_create_arg(factory_cls: type[MotifFactory]) -> None:
    """params_from_dict produces an instance of params_type."""
    factory = factory_cls()
    params = _valid_params(factory)
    assert isinstance(params, factory.params_type)


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_validate_params_rejects_wrong_type(factory_cls: type[MotifFactory]) -> None:
    """validate_params raises TypeError when given the wrong type."""
    factory = factory_cls()
    with pytest.raises(TypeError):
        factory.validate_params("not_a_params_instance")  # type: ignore[arg-type]


# --- Boundary tests (unconstrained factories only) ----------------------
# AutoCatalysis and SubstrateInhibition are excluded because their parameter
# spaces have a strict inequality constraint (k_deg > k_cat, K_i > K_m), making
# the simultaneous minimum or maximum of all parameters invalid by definition.


@pytest.mark.parametrize("factory_cls", UNCONSTRAINED_FACTORIES)
def test_create_at_min_boundary_does_not_raise(factory_cls: type[MotifFactory]) -> None:
    """create() at the lower parameter boundary does not raise."""
    factory = factory_cls()
    factory.create(_min_params(factory))


@pytest.mark.parametrize("factory_cls", UNCONSTRAINED_FACTORIES)
def test_create_at_max_boundary_does_not_raise(factory_cls: type[MotifFactory]) -> None:
    """create() at the upper parameter boundary does not raise."""
    factory = factory_cls()
    factory.create(_max_params(factory))


# --- MotifParams inheritance -------------------------------------------


@pytest.mark.parametrize("factory_cls", ALL_FACTORIES)
def test_params_type_is_motif_params_subclass(factory_cls: type[MotifFactory]) -> None:
    """Every factory's params_type is a subclass of MotifParams."""
    factory = factory_cls()
    assert issubclass(factory.params_type, MotifParams)


# --- Species names -------------------------------------------------------


def test_birth_death_species(birth_death: BirthDeathFactory) -> None:
    """BirthDeath has exactly one species named 'A' by default."""
    assert birth_death.species_names == ("A",)


def test_toggle_switch_species(toggle_switch: ToggleSwitchFactory) -> None:
    """ToggleSwitch has species A and B in that order by default."""
    assert toggle_switch.species_names == ("A", "B")


def test_enzymatic_catalysis_species(
    enzymatic_catalysis: EnzymaticCatalysisFactory,
) -> None:
    """EnzymaticCatalysis has species S, E, C, P in that order by default."""
    assert enzymatic_catalysis.species_names == ("S", "E", "C", "P")


def test_repressilator_species(repressilator: RepressilatorFactory) -> None:
    """Repressilator has species A, B, C in that order by default."""
    assert repressilator.species_names == ("A", "B", "C")


def test_feedforward_species(feedforward_loop: IncoherentFeedforwardFactory) -> None:
    """FeedforwardLoop has species X, Y, Z in that order by default."""
    assert feedforward_loop.species_names == ("X", "Y", "Z")


# --- Configurable species names -----------------------------------------


def test_species_name_override_birth_death() -> None:
    """BirthDeathFactory with species_names override uses the custom name."""
    factory = BirthDeathFactory(species_names=("X",))
    assert factory.species_names == ("X",)
    crn = factory.create(BirthDeathParams(k_prod=5.0, k_deg=0.1))
    assert list(crn.species_names) == ["X"]


def test_species_name_override_produces_correct_crn() -> None:
    """CRN produced with custom species name contains that name, not the default."""
    factory = BirthDeathFactory(species_names=("mRNA",))
    crn = factory.create(BirthDeathParams(k_prod=10.0, k_deg=0.2))
    assert "mRNA" in list(crn.species_names)
    assert "A" not in list(crn.species_names)


def test_species_name_wrong_length_raises() -> None:
    """Passing wrong number of species names raises ValueError."""
    with pytest.raises(ValueError, match="requires 1 species"):
        BirthDeathFactory(species_names=("A", "B"))


def test_initial_state_ranges_keys_follow_custom_names() -> None:
    """initial_state_ranges() uses the overridden species name as key."""
    factory = BirthDeathFactory(species_names=("X",))
    state_ranges = factory.initial_state_ranges()
    assert "X" in state_ranges
    assert "A" not in state_ranges


# --- validate_params ----------------------------------------------------


def test_autocatalysis_validate_rejects_k_cat_ge_k_deg() -> None:
    """AutoCatalysisFactory.validate_params raises ValueError when k_cat >= k_deg."""
    factory = AutoCatalysisFactory()
    with pytest.raises(ValueError, match="k_deg > k_cat"):
        factory.validate_params(AutoCatalysisParams(k_basal=1.0, k_cat=0.5, k_deg=0.1))


def test_autocatalysis_validate_accepts_valid_params() -> None:
    """AutoCatalysisFactory.validate_params does not raise for valid params."""
    factory = AutoCatalysisFactory()
    factory.validate_params(AutoCatalysisParams(k_basal=1.0, k_cat=0.05, k_deg=0.2))


def test_substrate_inhibition_validate_rejects_ki_le_km() -> None:
    """SubstrateInhibitionMotifFactory.validate_params raises ValueError when K_i <= K_m."""
    factory = SubstrateInhibitionMotifFactory()
    with pytest.raises(ValueError, match="K_i > K_m"):
        factory.validate_params(
            SubstrateInhibitionParams(k_in=1.0, V_max=10.0, K_m=50.0, K_i=5.0, k_deg=0.1)
        )


def test_substrate_inhibition_validate_accepts_valid_params() -> None:
    """SubstrateInhibitionMotifFactory.validate_params does not raise for valid params."""
    factory = SubstrateInhibitionMotifFactory()
    factory.validate_params(
        SubstrateInhibitionParams(k_in=1.0, V_max=10.0, K_m=10.0, K_i=50.0, k_deg=0.1)
    )


# --- Stoichiometry values -----------------------------------------------
# Each test uses explicit, human-readable parameter values so the expected
# stoichiometry can be verified by inspection.


def test_birth_death_stoichiometry(birth_death: BirthDeathFactory) -> None:
    """Birth reaction adds 1, death reaction removes 1."""
    crn = birth_death.create(BirthDeathParams(k_prod=1.0, k_deg=0.1))
    stoich = crn.stoichiometry_matrix
    # R1: empty -> A
    assert stoich[0, 0].item() == pytest.approx(1.0)
    # R2: A -> empty
    assert stoich[1, 0].item() == pytest.approx(-1.0)


def test_toggle_switch_stoichiometry(toggle_switch: ToggleSwitchFactory) -> None:
    """Toggle switch stoichiometry has correct species-specific signs."""
    params = ToggleSwitchParams(
        k_max_A=10.0, k_max_B=10.0,
        k_half_A=5.0, k_half_B=5.0,
        n_A=2.0, n_B=2.0,
        k_deg_A=0.1, k_deg_B=0.1,
    )
    crn = toggle_switch.create(params)
    stoich = crn.stoichiometry_matrix
    # R1: empty -> A  (+A, 0B)
    assert stoich[0, 0].item() == pytest.approx(1.0)
    assert stoich[0, 1].item() == pytest.approx(0.0)
    # R2: A -> empty  (-A, 0B)
    assert stoich[1, 0].item() == pytest.approx(-1.0)
    assert stoich[1, 1].item() == pytest.approx(0.0)
    # R3: empty -> B  (0A, +B)
    assert stoich[2, 0].item() == pytest.approx(0.0)
    assert stoich[2, 1].item() == pytest.approx(1.0)
    # R4: B -> empty  (0A, -B)
    assert stoich[3, 0].item() == pytest.approx(0.0)
    assert stoich[3, 1].item() == pytest.approx(-1.0)


def test_enzymatic_catalysis_stoichiometry(
    enzymatic_catalysis: EnzymaticCatalysisFactory,
) -> None:
    """Enzymatic catalysis binding reaction consumes S and E, produces C."""
    params = EnzymaticCatalysisParams(
        k_on=1.0, k_off=0.1, k_cat=0.5, k_prod=2.0, k_deg_P=0.1
    )
    crn = enzymatic_catalysis.create(params)
    stoich = crn.stoichiometry_matrix
    # R1: S + E -> C  (-S, -E, +C, 0P)
    assert stoich[0, 0].item() == pytest.approx(-1.0)  # S
    assert stoich[0, 1].item() == pytest.approx(-1.0)  # E
    assert stoich[0, 2].item() == pytest.approx(1.0)   # C
    assert stoich[0, 3].item() == pytest.approx(0.0)   # P


def test_repressilator_cyclic_repression(repressilator: RepressilatorFactory) -> None:
    """Each production reaction only changes its own species (cyclic topology)."""
    params = RepressilatorParams(
        k_max_A=10.0, k_max_B=10.0, k_max_C=10.0,
        k_half_A=5.0, k_half_B=5.0, k_half_C=5.0,
        n_A=2.0, n_B=2.0, n_C=2.0,
        k_deg_A=0.1, k_deg_B=0.1, k_deg_C=0.1,
    )
    crn = repressilator.create(params)
    stoich = crn.stoichiometry_matrix
    # R1: empty -> A  (+A only)
    assert stoich[0, 0].item() == pytest.approx(1.0)
    assert stoich[0, 1].item() == pytest.approx(0.0)
    assert stoich[0, 2].item() == pytest.approx(0.0)
    # R3: empty -> B  (+B only)
    assert stoich[2, 0].item() == pytest.approx(0.0)
    assert stoich[2, 1].item() == pytest.approx(1.0)
    # R5: empty -> C  (+C only)
    assert stoich[4, 0].item() == pytest.approx(0.0)
    assert stoich[4, 2].item() == pytest.approx(1.0)


def test_substrate_inhibition_stoichiometry(
    substrate_inhibition: SubstrateInhibitionMotifFactory,
) -> None:
    """Substrate inhibition conversion reaction removes S and adds P."""
    # K_i=100 > K_m=10 satisfies the required constraint
    params = SubstrateInhibitionParams(
        k_in=1.0, V_max=10.0, K_m=10.0, K_i=100.0, k_deg=0.1
    )
    crn = substrate_inhibition.create(params)
    stoich = crn.stoichiometry_matrix
    # R2: S -> P  (-S, +P)
    assert stoich[1, 0].item() == pytest.approx(-1.0)
    assert stoich[1, 1].item() == pytest.approx(1.0)
