"""Generate SDE rollouts from a trained encoder + SDE pair."""

from __future__ import annotations

import torch

from crn_surrogate.configs.model_config import SDEConfig
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.encoder.tensor_repr import CRNTensorRepr
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver


class ModelEvaluator:
    """Generate SDE rollouts from a trained encoder + SDE pair.

    Encodes the CRN context once per call to rollout() and runs K
    independent Euler-Maruyama trajectories.
    """

    def __init__(
        self,
        encoder: BipartiteGNNEncoder,
        sde: CRNNeuralSDE,
        sde_config: SDEConfig,
    ) -> None:
        """Args:
        encoder: Trained bipartite GNN encoder.
        sde: Trained neural SDE.
        sde_config: SDE configuration (used to build the solver).
        """
        self._encoder = encoder
        self._sde = sde
        self._solver = EulerMaruyamaSolver(sde_config)

    def rollout(
        self,
        crn_repr: CRNTensorRepr,
        initial_state: torch.Tensor,
        times: torch.Tensor,
        dt: float,
        n_rollouts: int = 50,
    ) -> torch.Tensor:
        """Generate K independent SDE rollouts.

        Args:
            crn_repr: Tensor representation of the CRN.
            initial_state: (n_species,) initial state.
            times: (T,) evaluation time grid.
            dt: Solver step size.
            n_rollouts: Number of independent rollouts K.

        Returns:
            (K, T, n_species) stacked trajectories.
        """
        device = next(self._encoder.parameters()).device
        crn_repr = crn_repr.to(device)
        initial_state = initial_state.to(device)
        times = times.to(device)

        self._encoder.eval()
        self._sde.eval()
        with torch.no_grad():
            ctx = self._encoder(crn_repr)
            trajectories = [
                self._solver.solve(
                    self._sde, initial_state.clone(), ctx, times, dt
                ).states
                for _ in range(n_rollouts)
            ]
        return torch.stack(trajectories, dim=0)  # (K, T, n_species)
