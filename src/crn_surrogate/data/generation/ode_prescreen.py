"""ODE-based pre-screening for CRN dynamical diversity.

Runs a fast deterministic ODE integration to classify the qualitative
dynamics before committing to expensive SSA simulation. Rejects CRNs
that produce trivially boring dynamics (all-decay, all-blowup).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_ivp

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.configs import ODEPreScreenConfig


class DynamicsType(enum.Enum):
    """Classification of ODE trajectory dynamics."""

    DECAY_TO_ZERO = "decay_to_zero"
    DECAY_TO_NONZERO = "decay_to_nonzero"
    GROWTH = "growth"
    SUSTAINED = "sustained"
    TRANSIENT_PEAK = "transient_peak"
    OSCILLATORY = "oscillatory"
    BLOWUP = "blowup"


@dataclass(frozen=True)
class PreScreenResult:
    """Result of ODE pre-screening.

    Attributes:
        accepted: Whether the CRN passes the pre-screen.
        dynamics_type: Classification of the dominant dynamics.
        max_value: Maximum value reached during integration.
        final_values: Final state values from the ODE.
    """

    accepted: bool
    dynamics_type: DynamicsType
    max_value: float
    final_values: np.ndarray


class ODEPreScreen:
    """Fast ODE-based pre-screening for CRN dynamical diversity.

    Integrates the deterministic mass-action ODE for a short time to
    classify the qualitative dynamics. Rejects CRNs where all species
    monotonically decay to zero (boring) or blow up (problematic).
    """

    def __init__(self, config: ODEPreScreenConfig) -> None:
        self._config = config

    def check(self, crn: CRN, initial_state: torch.Tensor) -> PreScreenResult:
        """Run ODE integration and classify the dynamics.

        Args:
            crn: The CRN to evaluate.
            initial_state: (n_species,) initial molecule counts.

        Returns:
            PreScreenResult with acceptance decision and classification.
        """
        cfg = self._config
        x0 = initial_state.numpy().astype(np.float64)

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            x_clamped = np.maximum(x, 0.0)
            x_tensor = torch.tensor(x_clamped, dtype=torch.float32)
            propensities = crn.evaluate_propensities(x_tensor, t)
            stoich = crn.stoichiometry_matrix.numpy().T  # (n_species, n_reactions)
            return stoich @ propensities.numpy()

        try:
            sol = solve_ivp(
                rhs,
                (0.0, cfg.t_max),
                x0,
                method="RK45",
                max_step=cfg.max_step,
                atol=1e-6,
                rtol=1e-3,
            )
        except Exception:
            return PreScreenResult(
                accepted=False,
                dynamics_type=DynamicsType.BLOWUP,
                max_value=float("inf"),
                final_values=x0,
            )

        if not sol.success:
            return PreScreenResult(
                accepted=False,
                dynamics_type=DynamicsType.BLOWUP,
                max_value=float("inf"),
                final_values=x0,
            )

        trajectory = sol.y.T  # (T, n_species)
        return self._classify(trajectory, x0)

    def _classify(self, trajectory: np.ndarray, x0: np.ndarray) -> PreScreenResult:
        """Classify ODE trajectory and decide acceptance.

        Args:
            trajectory: (T, n_species) ODE solution.
            x0: Initial state.

        Returns:
            PreScreenResult with classification.
        """
        cfg = self._config
        T, S = trajectory.shape
        half = T // 2

        max_val = float(np.max(np.abs(trajectory)))
        final = trajectory[-1]

        if max_val > cfg.blowup_threshold:
            return PreScreenResult(
                accepted=False,
                dynamics_type=DynamicsType.BLOWUP,
                max_value=max_val,
                final_values=final,
            )

        species_types = []
        for s in range(S):
            y = trajectory[:, s]
            start_val = np.mean(y[: max(3, T // 20)])
            end_val = np.mean(y[-max(3, T // 20) :])
            scale = max(np.mean(np.abs(y)), 1e-6)
            rel_change = (end_val - start_val) / scale

            y_late = y[half:]
            if len(y_late) > 5:
                dy = np.diff(y_late)
                sign_changes = np.sum(np.abs(np.diff(np.sign(dy))) > 0)
                late_range = np.max(y_late) - np.min(y_late)
                late_mean = np.mean(np.abs(y_late))
                is_oscillatory = (
                    sign_changes > len(y_late) * 0.2
                    and late_range > 0.2 * late_mean
                    and late_mean > cfg.min_sustained_level
                )
            else:
                is_oscillatory = False

            if is_oscillatory:
                species_types.append(DynamicsType.OSCILLATORY)
            elif rel_change > 0.3 and end_val > start_val:
                species_types.append(DynamicsType.GROWTH)
            elif abs(rel_change) < 0.15 and end_val > cfg.min_sustained_level:
                species_types.append(DynamicsType.SUSTAINED)
            elif end_val > cfg.min_sustained_level:
                species_types.append(DynamicsType.DECAY_TO_NONZERO)
            else:
                peak_idx = np.argmax(y)
                peak_val = y[peak_idx]
                if (
                    peak_idx < T * 0.5
                    and peak_val > start_val * 1.5
                    and end_val < peak_val * 0.5
                    and peak_val > cfg.min_sustained_level
                ):
                    species_types.append(DynamicsType.TRANSIENT_PEAK)
                else:
                    species_types.append(DynamicsType.DECAY_TO_ZERO)

        priority = [
            DynamicsType.OSCILLATORY,
            DynamicsType.TRANSIENT_PEAK,
            DynamicsType.GROWTH,
            DynamicsType.SUSTAINED,
            DynamicsType.DECAY_TO_NONZERO,
            DynamicsType.DECAY_TO_ZERO,
        ]
        item_type = DynamicsType.DECAY_TO_ZERO
        for p in priority:
            if p in species_types:
                item_type = p
                break

        all_decay = all(st == DynamicsType.DECAY_TO_ZERO for st in species_types)
        accepted = not all_decay

        return PreScreenResult(
            accepted=accepted,
            dynamics_type=item_type,
            max_value=max_val,
            final_values=final,
        )
