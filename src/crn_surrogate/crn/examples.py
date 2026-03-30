"""Factory functions for well-known Chemical Reaction Networks.

Each function returns a fully constructed CRN object with descriptive parameter
names and documented biological/mathematical interpretation.

Companion ``_analytical`` functions return the exact CLE drift and diffusion
as callables, providing ground-truth references for dynamics visualization.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import (
    constant_rate,
    enzyme_michaelis_menten,
    hill,
    mass_action,
)
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
                propensity=constant_rate(k_birth),
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
        MAPKKK* → MAPKKK: phosphatase deactivates MAPKKK* (enzymatic)
        MAPKK → MAPKK*:   MAPKKK* activates MAPKK (enzymatic, MAPKKK* catalytic)
        MAPKK* → MAPKK:   phosphatase deactivates MAPKK* (enzymatic)
        MAPK → MAPK*:     MAPKK* activates MAPK (enzymatic, MAPKK* catalytic)
        MAPK* → MAPK:     phosphatase deactivates MAPK* (enzymatic)

    The three deactivation reactions use enzyme_michaelis_menten with the
    shared phosphatase as enzyme. The two downstream activation reactions use
    enzyme_michaelis_menten with the upstream active kinase as enzyme.

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
            # MAPKKK → MAPKKK*: basal first-order activation
            Reaction(
                stoichiometry=torch.tensor([-1, 1, 0, 0, 0, 0, 0]),
                propensity=mass_action(
                    k_cat_kkk, torch.tensor([1.0, 0, 0, 0, 0, 0, 0])
                ),
                name="MAPKKK activation",
            ),
            # MAPKKK* → MAPKKK: phosphatase deactivation (enzyme=Phosphatase idx=6, substrate=MAPKKK* idx=1)
            Reaction(
                stoichiometry=torch.tensor([1, -1, 0, 0, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_phos_kkk, k_m_phos, enzyme_index=6, substrate_index=1
                ),
                name="MAPKKK* deactivation",
            ),
            # MAPKK → MAPKK*: MAPKKK* activates (enzyme=MAPKKK* idx=1, substrate=MAPKK idx=2)
            Reaction(
                stoichiometry=torch.tensor([0, 0, -1, 1, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_kk, k_m, enzyme_index=1, substrate_index=2
                ),
                name="MAPKK activation",
            ),
            # MAPKK* → MAPKK: phosphatase deactivation (enzyme=Phosphatase idx=6, substrate=MAPKK* idx=3)
            Reaction(
                stoichiometry=torch.tensor([0, 0, 1, -1, 0, 0, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_phos_kk, k_m_phos, enzyme_index=6, substrate_index=3
                ),
                name="MAPKK* deactivation",
            ),
            # MAPK → MAPK*: MAPKK* activates (enzyme=MAPKK* idx=3, substrate=MAPK idx=4)
            Reaction(
                stoichiometry=torch.tensor([0, 0, 0, 0, -1, 1, 0]),
                propensity=enzyme_michaelis_menten(
                    k_cat_k, k_m, enzyme_index=3, substrate_index=4
                ),
                name="MAPK activation",
            ),
            # MAPK* → MAPK: phosphatase deactivation (enzyme=Phosphatase idx=6, substrate=MAPK* idx=5)
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


# ── Analytical CLE references ─────────────────────────────────────────────────


def birth_death_analytical(
    k_birth: float = 1.0,
    k_death: float = 0.1,
) -> dict[str, Callable | float]:
    """Analytical CLE drift and diffusion for the birth-death process.

    The Chemical Langevin Equation for birth-death is:
        dX = (k_birth − k_death · X) dt + sqrt(k_birth + k_death · X) dW

    The stationary distribution is Poisson(k_birth / k_death), so the
    stationary mean equals the stationary variance.

    Args:
        k_birth: Zero-order birth rate.
        k_death: First-order death rate.

    Returns:
        Dict with keys:
            ``drift``: callable ``(x: Tensor) -> Tensor``, vectorized over x.
            ``diffusion``: callable ``(x: Tensor) -> Tensor``, noise amplitude.
            ``stationary_mean``: float, k_birth / k_death.
            ``stationary_var``: float, k_birth / k_death (Poisson identity).
    """
    return {
        "drift": lambda x: torch.as_tensor(k_birth, dtype=x.dtype) - k_death * x,
        "diffusion": lambda x: (
            (torch.as_tensor(k_birth, dtype=x.dtype) + k_death * x)
            .clamp(min=0.0)
            .sqrt()
        ),
        "stationary_mean": k_birth / k_death,
        "stationary_var": k_birth / k_death,
    }


def lotka_volterra_analytical(
    k_prey_birth: float = 1.0,
    k_predation: float = 0.01,
    k_predator_death: float = 0.5,
) -> dict[str, Callable]:
    """Analytical CLE drift for the Lotka-Volterra predator-prey system.

    The deterministic part of the CLE is:
        d(prey)/dt   = k_prey_birth · prey − k_predation · prey · pred
        d(pred)/dt   = k_predation · prey · pred − k_predator_death · pred

    No closed-form stationary distribution exists (the system oscillates).

    Args:
        k_prey_birth: First-order prey birth rate.
        k_predation: Second-order predation rate.
        k_predator_death: First-order predator death rate.

    Returns:
        Dict with key:
            ``drift``: callable ``(prey: Tensor, pred: Tensor) -> Tensor(2,)``.
    """
    return {
        "drift": lambda prey, pred: torch.stack(
            [
                k_prey_birth * prey - k_predation * prey * pred,
                k_predation * prey * pred - k_predator_death * pred,
            ]
        ),
    }
