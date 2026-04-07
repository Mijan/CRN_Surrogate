"""State space transforms for the neural SDE.

Provides forward/inverse transforms between raw molecule counts and the
space in which the SDE operates. Currently supports identity (raw counts)
and log1p (log(1+x)) transforms.
"""

from __future__ import annotations

import torch


class StateTransform:
    """Base class for state transforms. Identity by default."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from raw counts to SDE space."""
        return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from SDE space back to raw counts."""
        return z

    def transform_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Transform a full trajectory tensor.

        Args:
            traj: (..., T, n_species) trajectory in raw count space.

        Returns:
            Same shape tensor in transformed space.
        """
        return self.forward(traj)

    def inverse_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Inverse-transform a full trajectory tensor.

        Args:
            traj: (..., T, n_species) trajectory in transformed space.

        Returns:
            Same shape tensor in raw count space.
        """
        return self.inverse(traj)


class Log1pTransform(StateTransform):
    """Log1p state transform: z = log(1 + x), x = exp(z) - 1.

    Balances gradient contributions across different count scales.
    A change from 5 to 10 molecules gets similar gradient weight as
    a change from 50 to 100 molecules.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform raw counts to log1p space."""
        return torch.log1p(x.clamp(min=0.0))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform log1p space back to raw counts."""
        return torch.expm1(z).clamp(min=0.0)


def get_state_transform(use_log1p: bool) -> StateTransform:
    """Factory function for state transforms.

    Args:
        use_log1p: If True, return Log1pTransform. Otherwise identity.

    Returns:
        StateTransform instance.
    """
    if use_log1p:
        return Log1pTransform()
    return StateTransform()
