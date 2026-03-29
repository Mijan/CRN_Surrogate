from __future__ import annotations

import torch

from crn_surrogate.data.crn import CRNDefinition
from crn_surrogate.data.propensities import make_propensity
from crn_surrogate.simulator.trajectory import Trajectory


class GillespieSSA:
    """Exact stochastic simulation algorithm (Gillespie 1977, direct method)."""

    def simulate(
        self,
        crn: CRNDefinition,
        initial_state: torch.Tensor,
        t_max: float,
        max_reactions: int = 100_000,
    ) -> Trajectory:
        """Run exact SSA and return trajectory at reaction event times.

        Args:
            crn: The CRN to simulate.
            initial_state: (n_species,) initial molecule counts.
            t_max: Maximum simulation time.
            max_reactions: Safety cap on number of reaction events.

        Returns:
            Trajectory with event times and states.
        """
        propensities = [
            make_propensity(crn.propensity_types[r], crn.reactant_matrix[r])
            for r in range(crn.n_reactions)
        ]

        state = initial_state.float().clone()
        t = torch.zeros(1)

        times = [t.item()]
        states = [state.clone()]

        for _ in range(max_reactions):
            a = self._compute_propensities(state, propensities, crn)
            a_total = a.sum()
            if a_total <= 0:
                break

            dt = self._sample_wait_time(a_total)
            t = t + dt
            if t.item() >= t_max:
                break

            r = self._select_reaction(a, a_total)
            state = state + crn.stoichiometry[r].float()
            state = state.clamp(min=0.0)

            times.append(t.item())
            states.append(state.clone())

        times_tensor = torch.tensor(times)
        states_tensor = torch.stack(states, dim=0)
        return Trajectory(times=times_tensor, states=states_tensor)

    def _compute_propensities(
        self,
        state: torch.Tensor,
        propensities: list,
        crn: CRNDefinition,
    ) -> torch.Tensor:
        """Evaluate all propensities for the current state."""
        a = torch.stack(
            [
                propensities[r].evaluate(state, crn.propensity_params[r])
                for r in range(crn.n_reactions)
            ]
        )
        return a.clamp(min=0.0)

    def _sample_wait_time(self, a_total: torch.Tensor) -> torch.Tensor:
        """Sample exponential waiting time."""
        u = torch.rand(1).clamp(min=1e-10)
        return -torch.log(u) / a_total

    def _select_reaction(self, a: torch.Tensor, a_total: torch.Tensor) -> int:
        """Sample which reaction fires proportional to propensities."""
        probs = a / a_total
        return int(torch.multinomial(probs, num_samples=1).item())


def interpolate_to_grid(
    event_times: torch.Tensor,
    event_states: torch.Tensor,
    time_grid: torch.Tensor,
) -> torch.Tensor:
    """Zero-order hold interpolation of SSA trajectory onto a regular grid.

    Args:
        event_times: (n_events,) sorted event times.
        event_states: (n_events, n_species) states at each event.
        time_grid: (T,) regular time grid to interpolate onto.

    Returns:
        (T, n_species) states at each grid time.
    """
    indices = torch.searchsorted(event_times, time_grid, right=True) - 1
    indices = indices.clamp(min=0, max=event_states.shape[0] - 1)
    return event_states[indices]


# ── Reference CRN library ────────────────────────────────────────────────────


def birth_death_crn(k1: float = 1.0, k2: float = 0.5) -> CRNDefinition:
    """Birth-death process: ∅ → A (rate k1), A → ∅ (rate k2).

    Analytical stationary distribution: Poisson(k1/k2).

    Args:
        k1: Birth rate.
        k2: Death rate.

    Returns:
        CRNDefinition for the birth-death process.
    """
    from crn_surrogate.data.propensities import PropensityType

    stoich = torch.tensor([[1.0], [-1.0]])
    reactants = torch.tensor([[0.0], [1.0]])
    params = torch.tensor([[k1, 0.0, 0.0, 0.0], [k2, 0.0, 0.0, 0.0]])
    return CRNDefinition(
        stoichiometry=stoich,
        reactant_matrix=reactants,
        propensity_types=(PropensityType.MASS_ACTION, PropensityType.MASS_ACTION),
        propensity_params=params,
        species_names=("A",),
    )


def lotka_volterra_crn(
    k1: float = 1.0, k2: float = 0.01, k3: float = 0.5
) -> CRNDefinition:
    """Lotka-Volterra predator-prey: A→2A, A+B→2B, B→∅.

    Args:
        k1: Prey birth rate.
        k2: Predation rate.
        k3: Predator death rate.

    Returns:
        CRNDefinition for Lotka-Volterra dynamics.
    """
    from crn_surrogate.data.propensities import PropensityType

    # species: [A (prey), B (predator)]
    stoich = torch.tensor(
        [
            [1.0, 0.0],  # A → 2A
            [-1.0, 1.0],  # A+B → 2B
            [0.0, -1.0],  # B → ∅
        ]
    )
    reactants = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    params = torch.tensor(
        [
            [k1, 0.0, 0.0, 0.0],
            [k2, 0.0, 0.0, 0.0],
            [k3, 0.0, 0.0, 0.0],
        ]
    )
    return CRNDefinition(
        stoichiometry=stoich,
        reactant_matrix=reactants,
        propensity_types=(
            PropensityType.MASS_ACTION,
            PropensityType.MASS_ACTION,
            PropensityType.MASS_ACTION,
        ),
        propensity_params=params,
        species_names=("prey", "predator"),
    )


def schlogl_crn(
    k1: float = 3e-7, k2: float = 1e-4, k3: float = 1e-3, k4: float = 3.5
) -> CRNDefinition:
    """Schlögl model: bistable single-species network.

    Reactions (species A, buffer B held constant at B=1e5):
      2A + B → 3A  (effective: 2A → 3A with rate k1*B)
      3A → 2A + B  (effective: 3A → 2A with rate k2)
      ∅ → A  (rate k3)
      A → ∅  (rate k4)

    Args:
        k1: Forward autocatalytic rate.
        k2: Reverse autocatalytic rate.
        k3: Production rate.
        k4: Degradation rate.

    Returns:
        CRNDefinition for the Schlögl model (1 species, 4 reactions).
    """
    from crn_surrogate.data.propensities import PropensityType

    # Absorb buffer B = 1e5 into k1
    k1_eff = k1 * 1e5
    stoich = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    reactants = torch.tensor([[2.0], [3.0], [0.0], [1.0]])
    params = torch.tensor(
        [
            [k1_eff, 0.0, 0.0, 0.0],
            [k2, 0.0, 0.0, 0.0],
            [k3, 0.0, 0.0, 0.0],
            [k4, 0.0, 0.0, 0.0],
        ]
    )
    return CRNDefinition(
        stoichiometry=stoich,
        reactant_matrix=reactants,
        propensity_types=(
            PropensityType.MASS_ACTION,
            PropensityType.MASS_ACTION,
            PropensityType.MASS_ACTION,
            PropensityType.MASS_ACTION,
        ),
        propensity_params=params,
        species_names=("A",),
    )


def toggle_switch_crn(
    v_max1: float = 10.0,
    k_m1: float = 5.0,
    n1: float = 2.0,
    v_max2: float = 10.0,
    k_m2: float = 5.0,
    n2: float = 2.0,
    d1: float = 1.0,
    d2: float = 1.0,
) -> CRNDefinition:
    """Gardner et al. toggle switch: two mutually repressing genes.

    Species: [A, B].
    Reactions:
      ∅ → A  (Hill repression by B): params [v_max1, k_m1, n1, 1 (species B)]
      A → ∅  (degradation, rate d1): mass action
      ∅ → B  (Hill repression by A): params [v_max2, k_m2, n2, 0 (species A)]
      B → ∅  (degradation, rate d2): mass action

    Returns:
        CRNDefinition for the toggle switch.
    """
    from crn_surrogate.data.propensities import PropensityType

    stoich = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    reactants = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = torch.tensor(
        [
            [v_max1, k_m1, n1, 1.0],  # Hill, repressed by B (index 1)
            [d1, 0.0, 0.0, 0.0],
            [v_max2, k_m2, n2, 0.0],  # Hill, repressed by A (index 0)
            [d2, 0.0, 0.0, 0.0],
        ]
    )
    return CRNDefinition(
        stoichiometry=stoich,
        reactant_matrix=reactants,
        propensity_types=(
            PropensityType.HILL,
            PropensityType.MASS_ACTION,
            PropensityType.HILL,
            PropensityType.MASS_ACTION,
        ),
        propensity_params=params,
        species_names=("A", "B"),
    )


def mapk_cascade_crn(
    k_act: float = 0.01,
    k_deact: float = 0.1,
) -> CRNDefinition:
    """Simple 3-tier MAPK cascade: MAPKKK* → MAPKK* → MAPK* activation.

    Species: [MAPKKK*, MAPKK*, MAPK*] (active fractions).
    Each tier: activation (mass-action) + phosphatase deactivation (mass-action).
    6 reactions total.

    Args:
        k_act: Activation rate constant.
        k_deact: Deactivation rate constant.

    Returns:
        CRNDefinition for the MAPK cascade.
    """
    from crn_surrogate.data.propensities import PropensityType

    stoich = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # ∅ → MAPKKK*  (input activation)
            [-1.0, 0.0, 0.0],  # MAPKKK* → ∅  (deactivation)
            [0.0, 1.0, 0.0],  # MAPKKK* → MAPKK*  (activation by MAPKKK*)
            [0.0, -1.0, 0.0],  # MAPKK* → ∅
            [0.0, 0.0, 1.0],  # MAPKK* → MAPK*
            [0.0, 0.0, -1.0],  # MAPK* → ∅
        ]
    )
    reactants = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    params = torch.tensor(
        [
            [k_act, 0.0, 0.0, 0.0],
            [k_deact, 0.0, 0.0, 0.0],
            [k_act, 0.0, 0.0, 0.0],
            [k_deact, 0.0, 0.0, 0.0],
            [k_act, 0.0, 0.0, 0.0],
            [k_deact, 0.0, 0.0, 0.0],
        ]
    )
    return CRNDefinition(
        stoichiometry=stoich,
        reactant_matrix=reactants,
        propensity_types=(PropensityType.MASS_ACTION,) * 6,
        propensity_params=params,
        species_names=("MAPKKK*", "MAPKK*", "MAPK*"),
    )
