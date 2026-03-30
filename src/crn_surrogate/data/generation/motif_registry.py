"""Registry mapping MotifType enum values to their factory instances."""

from __future__ import annotations

from crn_surrogate.data.generation.motif_type import MotifType
from crn_surrogate.data.generation.motifs.auto_catalysis import AutoCatalysisFactory
from crn_surrogate.data.generation.motifs.base import MotifFactory
from crn_surrogate.data.generation.motifs.birth_death import BirthDeathFactory
from crn_surrogate.data.generation.motifs.enzymatic_catalysis import (
    EnzymaticCatalysisFactory,
)
from crn_surrogate.data.generation.motifs.feedforward_loop import (
    IncoherentFeedforwardFactory,
)
from crn_surrogate.data.generation.motifs.negative_autoregulation import (
    NegativeAutoregulationFactory,
)
from crn_surrogate.data.generation.motifs.repressilator import RepressilatorFactory
from crn_surrogate.data.generation.motifs.substrate_inhibition_motif import (
    SubstrateInhibitionMotifFactory,
)
from crn_surrogate.data.generation.motifs.toggle_switch import ToggleSwitchFactory

_REGISTRY: dict[MotifType, type[MotifFactory]] = {
    MotifType.BIRTH_DEATH: BirthDeathFactory,
    MotifType.AUTO_CATALYSIS: AutoCatalysisFactory,
    MotifType.NEGATIVE_AUTOREGULATION: NegativeAutoregulationFactory,
    MotifType.TOGGLE_SWITCH: ToggleSwitchFactory,
    MotifType.ENZYMATIC_CATALYSIS: EnzymaticCatalysisFactory,
    MotifType.INCOHERENT_FEEDFORWARD: IncoherentFeedforwardFactory,
    MotifType.REPRESSILATOR: RepressilatorFactory,
    MotifType.SUBSTRATE_INHIBITION: SubstrateInhibitionMotifFactory,
}


def get_factory(motif_type: MotifType) -> MotifFactory:
    """Return a fresh factory instance for the given motif type.

    Args:
        motif_type: The MotifType enum value to look up.

    Returns:
        A new instance of the corresponding MotifFactory subclass.

    Raises:
        KeyError: If the motif_type is not registered (should not occur for
            valid MotifType enum values).
    """
    cls = _REGISTRY[motif_type]
    return cls()
