"""Named reference CRN constructor functions.

These are specific, well-studied Chemical Reaction Networks used for testing,
visualization, and qualitative benchmarks. Unlike the motif factories in
``motifs/``, these are not parameterized for random sampling over a training
distribution — they encode specific kinetic regimes of interest.
"""

from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import enzyme_michaelis_menten, mass_action
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.data.generation.mass_action_topology import (
    birth_death_topology,
    lotka_volterra_topology,
)
from crn_surrogate.data.generation.motifs.toggle_switch import (
    ToggleSwitchFactory,
    ToggleSwitchParams,
)

__all__ = [
    "birth_death",
    "lotka_volterra",
    "schlogl",
    "simple_mapk_cascade",
    "toggle_switch",
]


def birth_death(k_birth: float = 1.0, k_death: float = 0.1) -> CRN:
    """Single-species birth-death process: ∅ → A (rate k_birth), A → ∅ (rate k_death·A)."""
    return birth_death_topology().to_crn([k_birth, k_death])


def lotka_volterra(
    k_prey_birth: float = 1.0,
    k_predation: float = 0.01,
    k_predator_death: float = 0.5,
) -> CRN:
    """Two-species Lotka-Volterra: A → 2A, A+B → 2B, B → ∅."""
    return lotka_volterra_topology().to_crn(
        [k_prey_birth, k_predation, k_predator_death]
    )


def toggle_switch(
    alpha1: float = 5.0,
    alpha2: float = 5.0,
    beta: float = 1.0,
    hill_n: float = 2.0,
) -> CRN:
    """Gardner toggle switch (mutual Hill repression, k_half=1.0 for both species).

    Args:
        alpha1: Maximum production rate of A.
        alpha2: Maximum production rate of B.
        beta: Degradation rate (same for both species).
        hill_n: Hill exponent (same for both species).

    Returns:
        CRN for the toggle switch (2 species, 4 reactions).
    """
    return ToggleSwitchFactory().create(
        ToggleSwitchParams(
            k_max_A=alpha1,
            k_max_B=alpha2,
            k_half_A=1.0,
            k_half_B=1.0,
            n_A=hill_n,
            n_B=hill_n,
            k_deg_A=beta,
            k_deg_B=beta,
        )
    )


def schlogl(
    k1: float = 3e-7,
    k2: float = 1e-4,
    k3: float = 1e-3,
    k4: float = 3.5,
) -> CRN:
    """Schlögl model: bistable single-species network.

    The buffer species B is held constant at B = 1e5 and its effect is
    absorbed into k1_eff = k1 * B.

    Reactions:
        2A → 3A   (rate k1_eff * A^2, higher-order autocatalysis)
        3A → 2A   (rate k2 * A^3, reverse autocatalysis)
        ∅ → A     (rate k3, basal production)
        A → ∅     (rate k4 * A, degradation)

    Args:
        k1: Forward autocatalytic rate (effective rate = k1 * 1e5).
        k2: Reverse autocatalytic rate.
        k3: Zero-order production rate.
        k4: First-order degradation rate.

    Returns:
        CRN for the Schlögl model (1 species, 4 reactions).
    """
    k1_eff = k1 * 1e5
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1]),
                propensity=mass_action(k1_eff, torch.tensor([2.0])),
                name="autocatalysis forward",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1]),
                propensity=mass_action(k2, torch.tensor([3.0])),
                name="autocatalysis reverse",
            ),
            Reaction(
                stoichiometry=torch.tensor([1]),
                propensity=mass_action(k3, torch.tensor([0.0])),
                name="production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1]),
                propensity=mass_action(k4, torch.tensor([1.0])),
                name="degradation",
            ),
        ],
        species_names=["A"],
    )


def simple_mapk_cascade(
    k_cat_kkk: float = 0.1,
    k_cat_kk: float = 0.1,
    k_cat_k: float = 0.1,
    k_m: float = 10.0,
    k_cat_phos_kkk: float = 0.05,
    k_cat_phos_kk: float = 0.05,
    k_cat_phos_k: float = 0.05,
    k_m_phos: float = 10.0,
) -> CRN:
    """Three-tier MAPK cascade with enzymatic activation and deactivation.

    Species (7 total):
        0: MAPKKK   (inactive tier-1 kinase)
        1: MAPKKK*  (active tier-1 kinase)
        2: MAPKK    (inactive tier-2 kinase)
        3: MAPKK*   (active tier-2 kinase)
        4: MAPK     (inactive tier-3 kinase)
        5: MAPK*    (active tier-3 kinase)
        6: Phosphatase (catalytic, zero net stoichiometry in all reactions)

    Reactions (6):
        MAPKKK → MAPKKK*: basal activation (first-order in MAPKKK)
        MAPKKK* → MAPKKK: phosphatase deactivation (enzymatic)
        MAPKK → MAPKK*:   MAPKKK* activates MAPKK (enzymatic)
        MAPKK* → MAPKK:   phosphatase deactivation (enzymatic)
        MAPK → MAPK*:     MAPKK* activates MAPK (enzymatic)
        MAPK* → MAPK:     phosphatase deactivation (enzymatic)

    Args:
        k_cat_kkk: Basal activation rate for MAPKKK (first-order).
        k_cat_kk: Catalytic rate for MAPKK activation by MAPKKK*.
        k_cat_k: Catalytic rate for MAPK activation by MAPKK*.
        k_m: Michaelis constant for activation reactions.
        k_cat_phos_kkk: Phosphatase catalytic rate for MAPKKK* deactivation.
        k_cat_phos_kk: Phosphatase catalytic rate for MAPKK* deactivation.
        k_cat_phos_k: Phosphatase catalytic rate for MAPK* deactivation.
        k_m_phos: Michaelis constant for phosphatase reactions.

    Returns:
        CRN for the MAPK cascade (7 species, 6 reactions).
    """
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([-1, 1, 0, 0, 0, 0, 0]),
                propensity=mass_action(
                    k_cat_kkk, torch.tensor([1.0, 0, 0, 0, 0, 0, 0])
                ),
                name="MAPKKK activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([1, -1, 0, 0, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_phos_kkk, k_m_phos, enzyme_index=6, substrate_index=1
                ),
                name="MAPKKK* deactivation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, -1, 1, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_kk, k_m, enzyme_index=1, substrate_index=2
                ),
                name="MAPKK activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, 1, -1, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_phos_kk, k_m_phos, enzyme_index=6, substrate_index=3
                ),
                name="MAPKK* deactivation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, 0, 0, -1, 1, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_k, k_m, enzyme_index=3, substrate_index=4
                ),
                name="MAPK activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, 0, 0, 1, -1, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_phos_k, k_m_phos, enzyme_index=6, substrate_index=5
                ),
                name="MAPK* deactivation",
            ),
        ],
        species_names=[
            "MAPKKK",
            "MAPKKK*",
            "MAPKK",
            "MAPKK*",
            "MAPK",
            "MAPK*",
            "Phosphatase",
        ],
    )
