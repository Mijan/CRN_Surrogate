"""Mass-action CRN topology: reaction structure without rate constants.

A mass-action topology fully specifies which reactions exist and what they
consume and produce. Rate constants are assigned separately via ``to_crn()``,
cleanly separating structure from kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import constant_rate, mass_action
from crn_surrogate.crn.reaction import Reaction

__all__ = [
    "MassActionTopology",
    "auto_catalysis_topology",
    "birth_death_topology",
    "enzymatic_catalysis_topology",
    "lotka_volterra_topology",
]


@dataclass(frozen=True)
class MassActionTopology:
    """A mass-action CRN structure without rate constants.

    Each reaction is defined by its reactant and product stoichiometry vectors.
    The propensity type is inferred from the reactant order:
    - Order 0 (all-zero reactant row): constant_rate
    - Order >= 1: mass_action with the reactant vector

    Structural invariants enforced in __post_init__:
    - reactant_matrix and product_matrix have identical shape (n_reactions, n_species).
    - All entries are non-negative.
    - No reaction has zero net stoichiometry (no-op).
    - Every species participates in at least one reaction.
    - No duplicate (reactant, product) row pairs.

    Attributes:
        reactant_matrix: (n_reactions, n_species) non-negative entries.
            Row r is the reactant stoichiometry for reaction r.
        product_matrix: (n_reactions, n_species) non-negative entries.
            Row r is the product stoichiometry for reaction r.
        species_names: Human-readable species names. Defaults to S0, S1, ...
        reaction_names: Human-readable reaction names. Defaults to R0, R1, ...
    """

    reactant_matrix: torch.Tensor
    product_matrix: torch.Tensor
    species_names: tuple[str, ...] = ()
    reaction_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        R, S = self.reactant_matrix.shape
        if self.product_matrix.shape != (R, S):
            raise ValueError(
                f"Shape mismatch: reactant_matrix {tuple(self.reactant_matrix.shape)} "
                f"vs product_matrix {tuple(self.product_matrix.shape)}"
            )
        if (self.reactant_matrix < 0).any():
            raise ValueError("reactant_matrix must be non-negative")
        if (self.product_matrix < 0).any():
            raise ValueError("product_matrix must be non-negative")

        net = self.net_stoichiometry
        # No no-op reactions
        noop_mask = net.abs().sum(dim=1) == 0
        if noop_mask.any():
            noop_indices = noop_mask.nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"Reactions {noop_indices} have zero net stoichiometry (no-ops)")

        # Every species participates in at least one reaction
        inactive = net.abs().sum(dim=0) == 0
        if inactive.any():
            inactive_indices = inactive.nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"Species {inactive_indices} do not participate in any reaction"
            )

        # No duplicate (reactant, product) row pairs
        seen: set[tuple[tuple, tuple]] = set()
        for r in range(R):
            key = (
                tuple(self.reactant_matrix[r].tolist()),
                tuple(self.product_matrix[r].tolist()),
            )
            if key in seen:
                raise ValueError(f"Duplicate reaction at index {r}: {key}")
            seen.add(key)

        # Default species names
        if not self.species_names:
            object.__setattr__(self, "species_names", tuple(f"S{i}" for i in range(S)))
        elif len(self.species_names) != S:
            raise ValueError(
                f"species_names length {len(self.species_names)} != n_species {S}"
            )

        # Default reaction names
        if not self.reaction_names:
            object.__setattr__(self, "reaction_names", tuple(f"R{i}" for i in range(R)))
        elif len(self.reaction_names) != R:
            raise ValueError(
                f"reaction_names length {len(self.reaction_names)} != n_reactions {R}"
            )

    @property
    def net_stoichiometry(self) -> torch.Tensor:
        """(n_reactions, n_species) net change matrix."""
        return self.product_matrix - self.reactant_matrix

    @property
    def n_species(self) -> int:
        """Number of species."""
        return int(self.reactant_matrix.shape[1])

    @property
    def n_reactions(self) -> int:
        """Number of reactions."""
        return int(self.reactant_matrix.shape[0])

    def reaction_orders(self) -> torch.Tensor:
        """(n_reactions,) total reactant order per reaction."""
        return self.reactant_matrix.sum(dim=1)

    def to_crn(self, rate_constants: Sequence[float]) -> CRN:
        """Instantiate a CRN by assigning a rate constant to each reaction.

        The propensity type is inferred from the reactant order:
        - Order 0: constant_rate(k)
        - Order >= 1: mass_action(k, reactant_stoichiometry)

        Args:
            rate_constants: One rate constant per reaction, in reaction order.

        Returns:
            A fully specified CRN ready for simulation.

        Raises:
            ValueError: If len(rate_constants) != n_reactions.
        """
        if len(rate_constants) != self.n_reactions:
            raise ValueError(
                f"Expected {self.n_reactions} rate constants, got {len(rate_constants)}"
            )
        reactions = []
        for r in range(self.n_reactions):
            net = self.net_stoichiometry[r]
            reactant_vec = self.reactant_matrix[r]
            k = float(rate_constants[r])
            order = int(reactant_vec.sum().item())

            if order == 0:
                propensity = constant_rate(k=k)
            else:
                propensity = mass_action(
                    rate_constant=k,
                    reactant_stoichiometry=reactant_vec.float(),
                )

            reactions.append(
                Reaction(
                    stoichiometry=net.float(),
                    propensity=propensity,
                    name=self.reaction_names[r],
                )
            )

        return CRN(
            reactions=reactions,
            species_names=list(self.species_names),
        )

    def has_production(self) -> bool:
        """True if at least one reaction is zero-order (constitutive production)."""
        return bool((self.reactant_matrix.sum(dim=1) == 0).any())

    def has_degradation_for_all(self) -> bool:
        """True if every species has at least one reaction with negative net stoichiometry."""
        net = self.net_stoichiometry
        for s in range(self.n_species):
            if not (net[:, s] < 0).any():
                return False
        return True

    def summary(self) -> str:
        """Human-readable string summarizing the topology.

        Returns:
            Multi-line string listing all reactions in arrow notation.
        """
        lines = [
            f"MassActionTopology: {self.n_species} species, {self.n_reactions} reactions"
        ]
        for r in range(self.n_reactions):
            reactants = []
            products = []
            for s in range(self.n_species):
                count = int(self.reactant_matrix[r, s].item())
                if count > 0:
                    label = f"{count}{self.species_names[s]}" if count > 1 else self.species_names[s]
                    reactants.append(label)
            for s in range(self.n_species):
                count = int(self.product_matrix[r, s].item())
                if count > 0:
                    label = f"{count}{self.species_names[s]}" if count > 1 else self.species_names[s]
                    products.append(label)
            lhs = " + ".join(reactants) if reactants else "0"
            rhs = " + ".join(products) if products else "0"
            lines.append(f"  {self.reaction_names[r]}: {lhs} -> {rhs}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MassActionTopology(n_species={self.n_species}, "
            f"n_reactions={self.n_reactions}, "
            f"species={self.species_names})"
        )


# ── Named topology factories ──────────────────────────────────────────────────


def birth_death_topology() -> MassActionTopology:
    """Return the birth-death topology: 0 -> A, A -> 0."""
    return MassActionTopology(
        reactant_matrix=torch.tensor([[0.0], [1.0]]),
        product_matrix=torch.tensor([[1.0], [0.0]]),
        species_names=("A",),
        reaction_names=("birth", "death"),
    )


def auto_catalysis_topology() -> MassActionTopology:
    """Return the auto-catalysis topology: 0 -> A (basal), A -> 2A, A -> 0."""
    return MassActionTopology(
        reactant_matrix=torch.tensor([[0.0], [1.0], [1.0]]),
        product_matrix=torch.tensor([[1.0], [2.0], [0.0]]),
        species_names=("A",),
        reaction_names=("basal_production", "autocatalysis", "degradation"),
    )


def lotka_volterra_topology() -> MassActionTopology:
    """Return the Lotka-Volterra topology: prey birth, predation, predator death."""
    return MassActionTopology(
        reactant_matrix=torch.tensor([
            [1.0, 0.0],  # prey -> 2 prey (first-order birth)
            [1.0, 1.0],  # prey + predator -> 2 predator
            [0.0, 1.0],  # predator -> 0
        ]),
        product_matrix=torch.tensor([
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]),
        species_names=("prey", "predator"),
        reaction_names=("prey_birth", "predation", "predator_death"),
    )


def enzymatic_catalysis_topology() -> MassActionTopology:
    """Return the enzymatic Michaelis-Menten topology (S, E, C, P)."""
    return MassActionTopology(
        reactant_matrix=torch.tensor([
            [1.0, 1.0, 0.0, 0.0],  # S + E -> C
            [0.0, 0.0, 1.0, 0.0],  # C -> S + E
            [0.0, 0.0, 1.0, 0.0],  # C -> E + P
            [0.0, 0.0, 0.0, 0.0],  # 0 -> S
            [0.0, 0.0, 0.0, 1.0],  # P -> 0
        ]),
        product_matrix=torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        species_names=("S", "E", "C", "P"),
        reaction_names=(
            "binding",
            "unbinding",
            "catalysis",
            "substrate_input",
            "product_degradation",
        ),
    )
