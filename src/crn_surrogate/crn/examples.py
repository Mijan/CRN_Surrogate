"""Factory functions for well-known Chemical Reaction Networks.

Each function returns a fully constructed CRN object with descriptive parameter
names and documented biological/mathematical interpretation.
"""
from __future__ import annotations

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import hill, mass_action
from crn_surrogate.crn.reaction import Reaction


def birth_death(
    k_birth: float = 1.0,
    k_death: float = 0.1,
) -> CRN:
    """Single-species birth-death process.

    Reactions:
        ∅ → A   (rate k_birth)
        A → ∅   (rate k_death * A)

    The stationary distribution is Poisson(k_birth / k_death).

    Args:
        k_birth: Zero-order birth rate.
        k_death: First-order death rate.

    Returns:
        CRN for the birth-death process.
    """
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1]),
                propensity=mass_action(k_birth, torch.tensor([0.0])),
                name="birth",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1]),
                propensity=mass_action(k_death, torch.tensor([1.0])),
                name="death",
            ),
        ],
        species_names=["A"],
    )


def lotka_volterra(
    k_prey_birth: float = 1.0,
    k_predation: float = 0.01,
    k_predator_death: float = 0.5,
) -> CRN:
    """Two-species predator-prey system (Lotka-Volterra).

    Reactions:
        A → 2A          (prey birth, rate k_prey_birth * prey)
        A + B → 2B      (predation, rate k_predation * prey * predator)
        B → ∅           (predator death, rate k_predator_death * predator)

    Args:
        k_prey_birth: First-order prey birth rate.
        k_predation: Second-order predation rate.
        k_predator_death: First-order predator death rate.

    Returns:
        CRN for the Lotka-Volterra system.
    """
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1, 0]),
                propensity=mass_action(k_prey_birth, torch.tensor([1.0, 0.0])),
                name="prey birth",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1, 1]),
                propensity=mass_action(k_predation, torch.tensor([1.0, 1.0])),
                name="predation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, -1]),
                propensity=mass_action(k_predator_death, torch.tensor([0.0, 1.0])),
                name="predator death",
            ),
        ],
        species_names=["prey", "predator"],
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
        2A → 3A   (rate k1_eff * A*(A-1)/2, approximated as k1_eff * A^2)
        3A → 2A   (rate k2 * A^3)
        ∅ → A     (rate k3)
        A → ∅     (rate k4 * A)

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


def toggle_switch(
    alpha1: float = 5.0,
    alpha2: float = 5.0,
    beta: float = 1.0,
    hill_n: float = 2.0,
) -> CRN:
    """Gardner genetic toggle switch: two mutually activating genes (simplified).

    Uses Hill activation kinetics with k_m=1.0. Each gene's production rate
    increases with the other species and degrades at rate beta.

    Reactions:
        ∅ → A   (Hill activation by B, v_max=alpha1, k_m=1, n=hill_n)
        A → ∅   (degradation, rate beta)
        ∅ → B   (Hill activation by A, v_max=alpha2, k_m=1, n=hill_n)
        B → ∅   (degradation, rate beta)

    Args:
        alpha1: Maximum production rate of A.
        alpha2: Maximum production rate of B.
        beta: Degradation rate.
        hill_n: Hill exponent.

    Returns:
        CRN for the toggle switch (2 species, 4 reactions).
    """
    k_m = 1.0
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1, 0]),
                propensity=hill(
                    v_max=alpha1, k_m=k_m, hill_coefficient=hill_n, species_index=1
                ),
                name="A production",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1, 0]),
                propensity=mass_action(beta, torch.tensor([1.0, 0.0])),
                name="A degradation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 1]),
                propensity=hill(
                    v_max=alpha2, k_m=k_m, hill_coefficient=hill_n, species_index=0
                ),
                name="B production",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, -1]),
                propensity=mass_action(beta, torch.tensor([0.0, 1.0])),
                name="B degradation",
            ),
        ],
        species_names=["A", "B"],
    )


def simple_mapk_cascade(
    k_act: float = 0.01,
    k_deact: float = 0.1,
) -> CRN:
    """Simple three-tier MAPK cascade: MAPKKK* → MAPKK* → MAPK* activation.

    Each tier has an activation reaction (driven by the upstream species)
    and a phosphatase-driven deactivation. The top-tier input is zero-order.

    Reactions:
        ∅ → MAPKKK*         (zero-order input activation, rate k_act)
        MAPKKK* → ∅         (deactivation, rate k_deact)
        MAPKKK* → MAPKK*    (downstream activation, rate k_act)
        MAPKK* → ∅          (deactivation, rate k_deact)
        MAPKK* → MAPK*      (downstream activation, rate k_act)
        MAPK* → ∅           (deactivation, rate k_deact)

    Args:
        k_act: Activation rate constant (shared across all tiers).
        k_deact: Deactivation rate constant (shared across all tiers).

    Returns:
        CRN for the MAPK cascade (3 species, 6 reactions).
    """
    return CRN(
        reactions=[
            Reaction(
                stoichiometry=torch.tensor([1, 0, 0]),
                propensity=mass_action(k_act, torch.tensor([0.0, 0.0, 0.0])),
                name="MAPKKK* input activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([-1, 0, 0]),
                propensity=mass_action(k_deact, torch.tensor([1.0, 0.0, 0.0])),
                name="MAPKKK* deactivation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 1, 0]),
                propensity=mass_action(k_act, torch.tensor([1.0, 0.0, 0.0])),
                name="MAPKK* activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, -1, 0]),
                propensity=mass_action(k_deact, torch.tensor([0.0, 1.0, 0.0])),
                name="MAPKK* deactivation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, 1]),
                propensity=mass_action(k_act, torch.tensor([0.0, 1.0, 0.0])),
                name="MAPK* activation",
            ),
            Reaction(
                stoichiometry=torch.tensor([0, 0, -1]),
                propensity=mass_action(k_deact, torch.tensor([0.0, 0.0, 1.0])),
                name="MAPK* deactivation",
            ),
        ],
        species_names=["MAPKKK*", "MAPKK*", "MAPK*"],
    )
