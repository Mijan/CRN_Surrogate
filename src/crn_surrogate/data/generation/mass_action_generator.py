"""Random mass-action CRN generator.

Separates topology sampling (RandomTopologySampler) from rate assignment
(MassActionCRNGenerator), making it easy to work with both random and named
topologies.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.data.generation.mass_action_topology import MassActionTopology

__all__ = [
    "MassActionCRNGenerator",
    "MassActionGeneratorConfig",
    "RandomTopologyConfig",
    "RandomTopologySampler",
]

# Biologically motivated order weights: [P(order=0), P(order=1), P(order=2)]
_ORDER_WEIGHTS: list[float] = [0.15, 0.55, 0.30]


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RandomTopologyConfig:
    """Configuration for random mass-action topology sampling.

    Attributes:
        n_species_range: Inclusive (min, max) number of species.
        n_reactions_range: Inclusive (min, max) number of reactions.
        max_reactant_order: Maximum total reactant order per reaction.
        max_product_count: Max total product molecules per reaction.
        require_production: At least one zero-order production reaction.
        require_degradation: Every species has a net-negative reaction.
        max_attempts: Max retries per sample() call.
    """

    n_species_range: tuple[int, int] = (1, 3)
    n_reactions_range: tuple[int, int] = (2, 6)
    max_reactant_order: int = 2
    max_product_count: int = 2
    require_production: bool = True
    require_degradation: bool = True
    max_attempts: int = 100


@dataclass(frozen=True)
class MassActionGeneratorConfig:
    """Configuration for mass-action CRN generation.

    Combines topology sampling config with rate constant sampling range.

    Attributes:
        topology: Configuration for the random topology sampler.
        rate_constant_range: Log-uniform range for rate constant sampling.
    """

    topology: RandomTopologyConfig = field(default_factory=RandomTopologyConfig)
    rate_constant_range: tuple[float, float] = (0.01, 10.0)


# ── RandomTopologySampler ─────────────────────────────────────────────────────


class RandomTopologySampler:
    """Samples random MassActionTopology instances with structural constraints.

    Samples reaction vectors independently, then applies iterative structural
    repair until all configured constraints are satisfied or max_attempts
    is reached.

    Uses torch RNG for reproducibility under torch.manual_seed().
    """

    def __init__(self, config: RandomTopologyConfig) -> None:
        """Args:
        config: Topology sampling configuration.
        """
        self._config = config

    def sample(self) -> MassActionTopology:
        """Sample a single valid topology.

        Returns:
            A structurally valid MassActionTopology.

        Raises:
            RuntimeError: If constraints cannot be satisfied within max_attempts.
        """
        cfg = self._config
        for _ in range(cfg.max_attempts):
            n_species = int(
                torch.randint(
                    cfg.n_species_range[0], cfg.n_species_range[1] + 1, (1,)
                ).item()
            )
            n_reactions = int(
                torch.randint(
                    cfg.n_reactions_range[0], cfg.n_reactions_range[1] + 1, (1,)
                ).item()
            )
            try:
                reactant_rows, product_rows = self._sample_reaction_rows(
                    n_species, n_reactions
                )
                reactant_rows, product_rows = self._repair(
                    reactant_rows, product_rows, n_species
                )
                reactant_rows, product_rows = self._dedup(reactant_rows, product_rows)
                return MassActionTopology(
                    reactant_matrix=torch.stack(reactant_rows),
                    product_matrix=torch.stack(product_rows),
                )
            except (ValueError, RuntimeError):
                continue

        raise RuntimeError(
            f"RandomTopologySampler failed after {cfg.max_attempts} attempts. "
            "Check config constraints."
        )

    def sample_batch(self, n: int) -> list[MassActionTopology]:
        """Sample n topologies, warning on partial failures.

        Args:
            n: Target number of topologies to generate.

        Returns:
            List of MassActionTopology instances.

        Raises:
            RuntimeError: If more than half of n attempts fail.
        """
        results = []
        failures = 0
        for _ in range(n):
            try:
                results.append(self.sample())
            except RuntimeError:
                failures += 1
        if len(results) < n // 2:
            raise RuntimeError(
                f"Sampler failed on {failures}/{n} attempts. Check config."
            )
        if failures > 0:
            warnings.warn(
                f"Sampler failed on {failures}/{n} attempts.",
                RuntimeWarning,
                stacklevel=2,
            )
        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    def _sample_order(self) -> int:
        """Sample a reaction order from {0, …, max_reactant_order}."""
        max_order = self._config.max_reactant_order
        weights = _ORDER_WEIGHTS[: max_order + 1]
        total = sum(weights)
        r = torch.rand(1).item() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return max_order

    def _sample_product_vec(
        self, n_species: int, reactant_vec: torch.Tensor
    ) -> torch.Tensor:
        """Sample a product stoichiometry with non-trivial net change.

        Args:
            n_species: Number of species.
            reactant_vec: (n_species,) reactant stoichiometry.

        Returns:
            (n_species,) product stoichiometry with net != 0.
        """
        max_products = self._config.max_product_count
        for _ in range(20):
            product_vec = torch.zeros(n_species)
            if torch.rand(1).item() < 0.3:
                pass  # degradation: zero products
            else:
                n_prod = int(torch.randint(1, max_products + 1, (1,)).item())
                for _ in range(n_prod):
                    s = int(torch.randint(0, n_species, (1,)).item())
                    product_vec[s] += 1.0
            if not (product_vec - reactant_vec == 0).all():
                return product_vec
        # Guaranteed non-trivial: degradation gives net = -reactant != 0
        return torch.zeros(n_species)

    def _sample_reaction_vecs(
        self, n_species: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Sample (reactant_vec, product_vec, order) for one reaction.

        Args:
            n_species: Number of species.

        Returns:
            Tuple of (reactant_vec, product_vec, order).
        """
        order = self._sample_order()
        reactant_vec = torch.zeros(n_species)

        if order == 0:
            s = int(torch.randint(0, n_species, (1,)).item())
            product_vec = torch.zeros(n_species)
            product_vec[s] = 1.0
        elif order == 1:
            s = int(torch.randint(0, n_species, (1,)).item())
            reactant_vec[s] = 1.0
            product_vec = self._sample_product_vec(n_species, reactant_vec)
        else:
            if torch.rand(1).item() < 0.5 or n_species == 1:
                s = int(torch.randint(0, n_species, (1,)).item())
                reactant_vec[s] = 2.0
            else:
                idx = torch.randperm(n_species)[:2]
                reactant_vec[idx[0]] = 1.0
                reactant_vec[idx[1]] = 1.0
            product_vec = self._sample_product_vec(n_species, reactant_vec)

        return reactant_vec, product_vec, order

    def _sample_reaction_rows(
        self, n_species: int, n_reactions: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sample reactant and product rows for n_reactions with deduplication.

        Args:
            n_species: Number of species.
            n_reactions: Target number of reactions.

        Returns:
            Tuple of (reactant_rows, product_rows) lists of length n_reactions.
        """
        reactant_rows: list[torch.Tensor] = []
        product_rows: list[torch.Tensor] = []
        seen_pairs: set[tuple] = set()

        for _ in range(n_reactions):
            for _ in range(10):  # retry on duplicates
                rv, pv, _ = self._sample_reaction_vecs(n_species)
                pair = (tuple(rv.tolist()), tuple(pv.tolist()))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    reactant_rows.append(rv)
                    product_rows.append(pv)
                    break
            else:
                # Accept last sampled even if duplicate; repair/dedup will clean it up
                reactant_rows.append(rv)
                product_rows.append(pv)

        return reactant_rows, product_rows

    def _find_violations(
        self,
        reactant_rows: list[torch.Tensor],
        product_rows: list[torch.Tensor],
        n_species: int,
    ) -> list[tuple[str, int | None]]:
        """Return list of (kind, data) constraint violations.

        Args:
            reactant_rows: List of reactant stoichiometry vectors.
            product_rows: List of product stoichiometry vectors.
            n_species: Number of species.

        Returns:
            Ordered list of violations: ('non_participating', s),
            ('no_production', None), ('no_degradation', s).
        """
        cfg = self._config
        violations: list[tuple[str, int | None]] = []
        reactant_mat = torch.stack(reactant_rows)
        product_mat = torch.stack(product_rows)
        net = product_mat - reactant_mat

        for s in range(n_species):
            if net[:, s].abs().sum() == 0:
                violations.append(("non_participating", s))

        if cfg.require_production:
            orders = reactant_mat.sum(dim=1)
            if not (orders == 0).any():
                violations.append(("no_production", None))

        if cfg.require_degradation:
            for s in range(n_species):
                if not (net[:, s] < 0).any():
                    violations.append(("no_degradation", s))

        return violations

    def _fix_one_violation(
        self,
        violation: tuple[str, int | None],
        reactant_rows: list[torch.Tensor],
        product_rows: list[torch.Tensor],
        n_species: int,
    ) -> None:
        """Apply one in-place repair to satisfy the given violation.

        Args:
            violation: (kind, data) tuple from _find_violations.
            reactant_rows: Mutable list of reactant vectors.
            product_rows: Mutable list of product vectors.
            n_species: Number of species.
        """
        kind, data = violation
        n_reactions = len(reactant_rows)

        if kind == "non_participating":
            s = int(data)  # type: ignore[arg-type]
            # Replace a random reaction with degradation of species s
            idx = int(torch.randint(0, n_reactions, (1,)).item())
            rv = torch.zeros(n_species)
            rv[s] = 1.0
            reactant_rows[idx] = rv
            product_rows[idx] = torch.zeros(n_species)

        elif kind == "no_production":
            # Replace a non-zero-order reaction with zero-order production
            reactant_mat = torch.stack(reactant_rows)
            orders = reactant_mat.sum(dim=1)
            non_zero_idx = (orders > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_idx) > 0:
                choice = int(torch.randint(0, len(non_zero_idx), (1,)).item())
                idx = int(non_zero_idx[choice].item())
            else:
                idx = int(torch.randint(0, n_reactions, (1,)).item())
            s = int(torch.randint(0, n_species, (1,)).item())
            pv = torch.zeros(n_species)
            pv[s] = 1.0
            reactant_rows[idx] = torch.zeros(n_species)
            product_rows[idx] = pv

        elif kind == "no_degradation":
            s = int(data)  # type: ignore[arg-type]
            # Append a degradation reaction for species s
            rv = torch.zeros(n_species)
            rv[s] = 1.0
            reactant_rows.append(rv)
            product_rows.append(torch.zeros(n_species))

    def _repair(
        self,
        reactant_rows: list[torch.Tensor],
        product_rows: list[torch.Tensor],
        n_species: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Iteratively fix constraint violations until resolved or max passes reached.

        Args:
            reactant_rows: Mutable list of reactant vectors.
            product_rows: Mutable list of product vectors.
            n_species: Number of species.

        Returns:
            (reactant_rows, product_rows) after repair.

        Raises:
            ValueError: If violations remain after 10 passes.
        """
        for _ in range(10):
            violations = self._find_violations(reactant_rows, product_rows, n_species)
            if not violations:
                break
            self._fix_one_violation(
                violations[0], reactant_rows, product_rows, n_species
            )
        else:
            violations = self._find_violations(reactant_rows, product_rows, n_species)
            if violations:
                raise ValueError(f"Repair did not converge: {violations}")

        return reactant_rows, product_rows

    @staticmethod
    def _dedup(
        reactant_rows: list[torch.Tensor],
        product_rows: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Remove duplicate (reactant, product) row pairs, keeping last occurrence.

        Args:
            reactant_rows: List of reactant vectors.
            product_rows: List of product vectors.

        Returns:
            Deduplicated (reactant_rows, product_rows).
        """
        seen: dict[tuple, int] = {}
        for i, (rv, pv) in enumerate(zip(reactant_rows, product_rows)):
            key = (tuple(rv.tolist()), tuple(pv.tolist()))
            seen[key] = i
        indices = sorted(seen.values())
        return [reactant_rows[i] for i in indices], [product_rows[i] for i in indices]


# ── MassActionCRNGenerator ────────────────────────────────────────────────────


class MassActionCRNGenerator:
    """Generates random mass-action CRNs by composing topology + rate sampling.

    Works with random topologies (via RandomTopologySampler) or named topologies
    (via sample_from_topology).
    """

    def __init__(self, config: MassActionGeneratorConfig) -> None:
        """Args:
        config: Generator configuration (topology + rate constant range).
        """
        self._config = config
        self._topology_sampler = RandomTopologySampler(config.topology)

    def sample(self) -> CRN:
        """Sample a random topology and assign random rate constants.

        Returns:
            A fully specified CRN.

        Raises:
            RuntimeError: Propagated from the topology sampler on failure.
        """
        topology = self._topology_sampler.sample()
        rates = self._sample_rates(topology.n_reactions)
        return topology.to_crn(rates)

    def sample_batch(self, n: int) -> list[CRN]:
        """Sample n CRNs, warning on partial failures.

        Args:
            n: Target number of CRNs.

        Returns:
            List of CRN instances.

        Raises:
            RuntimeError: If more than half of n attempts fail.
        """
        results = []
        failures = 0
        for _ in range(n):
            try:
                results.append(self.sample())
            except RuntimeError:
                failures += 1
        if len(results) < n // 2:
            raise RuntimeError(
                f"Generator failed on {failures}/{n} attempts. Check config."
            )
        if failures > 0:
            warnings.warn(
                f"Generator failed on {failures}/{n} attempts.",
                RuntimeWarning,
                stacklevel=2,
            )
        return results

    def sample_from_topology(self, topology: MassActionTopology) -> CRN:
        """Assign random rate constants to a given topology.

        Args:
            topology: A MassActionTopology defining the reaction structure.

        Returns:
            A fully specified CRN with randomly sampled rate constants.
        """
        rates = self._sample_rates(topology.n_reactions)
        return topology.to_crn(rates)

    def sample_initial_state(
        self,
        crn: CRN,
        mean_molecules: float = 10.0,
        spread: float = 2.0,
    ) -> torch.Tensor:
        """Sample a log-normally distributed initial state.

        Args:
            crn: The CRN whose species count determines the output shape.
            mean_molecules: Geometric mean of molecule counts.
            spread: Geometric standard deviation.

        Returns:
            (n_species,) tensor of non-negative rounded molecule counts.
        """
        log_counts = torch.randn(crn.n_species) * math.log(spread) + math.log(
            mean_molecules
        )
        return log_counts.exp().round().clamp(min=0.0)

    def _sample_rates(self, n: int) -> list[float]:
        """Sample n rate constants log-uniformly from rate_constant_range.

        Args:
            n: Number of rate constants to sample.

        Returns:
            List of n positive floats.
        """
        low, high = self._config.rate_constant_range
        return [
            math.exp(
                torch.rand(1).item() * (math.log(high) - math.log(low)) + math.log(low)
            )
            for _ in range(n)
        ]
