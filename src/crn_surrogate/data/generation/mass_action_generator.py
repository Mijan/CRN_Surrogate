"""Random mass-action CRN generator for multi-topology training.

Generates chemically plausible random CRNs with configurable structural
constraints (production, degradation, species participation).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.propensities import (
    ConstantRateParams,
    SerializablePropensity,
    constant_rate,
    mass_action,
)
from crn_surrogate.crn.reaction import Reaction

__all__ = ["MassActionCRNGenerator", "MassActionGeneratorConfig"]


@dataclass(frozen=True)
class MassActionGeneratorConfig:
    """Configuration for random mass-action CRN generation.

    Attributes:
        n_species_range: Inclusive (min, max) number of species.
        n_reactions_range: Inclusive (min, max) number of reactions.
        max_reactant_order: Maximum total reactant order
            (0=constant, 1=unimolecular, 2=bimolecular).
        max_product_count: Max total product molecules per reaction.
        rate_constant_range: Log-uniform sampling range for rate constants.
        require_production: At least one zero-order (constitutive) reaction.
        require_degradation: Every species has at least one net-negative reaction.
        max_attempts: Max retries before raising RuntimeError.
    """

    n_species_range: tuple[int, int] = (1, 3)
    n_reactions_range: tuple[int, int] = (2, 6)
    max_reactant_order: int = 2
    max_product_count: int = 2
    rate_constant_range: tuple[float, float] = (0.01, 10.0)
    require_production: bool = True
    require_degradation: bool = True
    max_attempts: int = 100


# Biologically motivated order weights: [P(order=0), P(order=1), P(order=2)]
_ORDER_WEIGHTS: list[float] = [0.15, 0.55, 0.30]


class MassActionCRNGenerator:
    """Generates random mass-action CRNs within configurable constraints.

    Each call to sample() produces a structurally valid CRN satisfying the
    configured requirements (production, degradation, species participation,
    no duplicate reactions). The generator uses torch RNG so results are
    reproducible under torch.manual_seed().
    """

    def __init__(self, config: MassActionGeneratorConfig) -> None:
        """Args:
        config: Generator configuration.
        """
        self._config = config

    def sample(self) -> CRN:
        """Sample a single valid CRN.

        Returns:
            A randomly generated mass-action CRN.

        Raises:
            RuntimeError: If constraints cannot be satisfied within max_attempts.
        """
        cfg = self._config
        for _ in range(cfg.max_attempts):
            n_species = int(
                torch.randint(cfg.n_species_range[0], cfg.n_species_range[1] + 1, (1,)).item()
            )
            n_reactions = int(
                torch.randint(cfg.n_reactions_range[0], cfg.n_reactions_range[1] + 1, (1,)).item()
            )
            try:
                reactions = self._build_reactions(n_species, n_reactions)
                reactions = self._repair_constraints(reactions, n_species)
                return CRN(
                    reactions=reactions,
                    species_names=tuple(f"S{i}" for i in range(n_species)),
                )
            except (ValueError, RuntimeError):
                continue

        raise RuntimeError(
            f"MassActionCRNGenerator failed to produce a valid CRN after "
            f"{cfg.max_attempts} attempts. Check config constraints."
        )

    def sample_batch(self, n: int) -> list[CRN]:
        """Sample a batch of CRNs.

        Args:
            n: Target number of CRNs to generate.

        Returns:
            List of generated CRNs. May be shorter than n if some attempts fail.

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
                f"Generator failed on {failures}/{n} attempts.", RuntimeWarning, stacklevel=2
            )
        return results

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
            spread: Geometric standard deviation (multiplicative spread).

        Returns:
            (n_species,) tensor of non-negative rounded molecule counts.
        """
        log_counts = (
            torch.randn(crn.n_species) * math.log(spread) + math.log(mean_molecules)
        )
        return log_counts.exp().round().clamp(min=0.0)

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

    def _sample_rate_constant(self) -> float:
        """Sample a rate constant log-uniformly from rate_constant_range."""
        low, high = self._config.rate_constant_range
        log_rate = (
            torch.rand(1).item() * (math.log(high) - math.log(low)) + math.log(low)
        )
        return math.exp(log_rate)

    def _sample_product_vec(
        self, n_species: int, reactant_vec: torch.Tensor
    ) -> torch.Tensor:
        """Sample a product stoichiometry with non-trivial net change.

        Args:
            n_species: Number of species.
            reactant_vec: (n_species,) reactant stoichiometry.

        Returns:
            (n_species,) product stoichiometry with net change != 0.
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
            Tuple of reactant_vec (n_species,), product_vec (n_species,), order int.
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
            # Order 2: dimerization or two distinct species
            if torch.rand(1).item() < 0.5 or n_species == 1:
                s = int(torch.randint(0, n_species, (1,)).item())
                reactant_vec[s] = 2.0
            else:
                idx = torch.randperm(n_species)[:2]
                reactant_vec[idx[0]] = 1.0
                reactant_vec[idx[1]] = 1.0
            product_vec = self._sample_product_vec(n_species, reactant_vec)

        return reactant_vec, product_vec, order

    def _build_reactions(self, n_species: int, n_reactions: int) -> list[Reaction]:
        """Build a list of reactions with deduplication.

        Args:
            n_species: Number of species.
            n_reactions: Target number of reactions.

        Returns:
            List of Reaction objects.
        """
        reactions: list[Reaction] = []
        seen_pairs: set[tuple] = set()

        for _ in range(n_reactions):
            for _ in range(10):  # retry on duplicates
                reactant_vec, product_vec, order = self._sample_reaction_vecs(n_species)
                pair = (
                    tuple(reactant_vec.tolist()),
                    tuple(product_vec.tolist()),
                )
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    break

            net_stoich = product_vec - reactant_vec
            rate = self._sample_rate_constant()

            if order == 0:
                propensity = constant_rate(k=rate)
            else:
                propensity = mass_action(
                    rate_constant=rate,
                    reactant_stoichiometry=reactant_vec,
                )

            reactions.append(Reaction(stoichiometry=net_stoich, propensity=propensity))

        return reactions

    def _has_production(self, reactions: list[Reaction]) -> bool:
        """Return True if any reaction has a constant-rate (zero-order) propensity."""
        for rxn in reactions:
            if isinstance(rxn.propensity, SerializablePropensity):
                if isinstance(rxn.propensity.params, ConstantRateParams):
                    return True
        return False

    def _degraded_species(self, reactions: list[Reaction], n_species: int) -> set[int]:
        """Return indices of species that have net-negative stoichiometry somewhere."""
        degraded: set[int] = set()
        for rxn in reactions:
            for s in range(n_species):
                if rxn.stoichiometry[s].item() < 0:
                    degraded.add(s)
        return degraded

    def _participating_species(
        self, reactions: list[Reaction], n_species: int
    ) -> set[int]:
        """Return indices of species that appear in any reaction's net stoichiometry."""
        participating: set[int] = set()
        for rxn in reactions:
            for s in range(n_species):
                if rxn.stoichiometry[s].item() != 0:
                    participating.add(s)
        return participating

    def _reaction_key(self, rxn: Reaction) -> tuple:
        """Compute a deduplication key for a reaction.

        Uses (net_stoichiometry, species_dependencies) as the key.
        Two reactions with the same net change and the same influencing species
        are considered duplicates.

        Args:
            rxn: Reaction to key.

        Returns:
            Hashable key tuple.
        """
        stoich_key = tuple(rxn.stoichiometry.tolist())
        if isinstance(rxn.propensity, SerializablePropensity):
            deps_key = tuple(sorted(rxn.propensity.species_dependencies))
        else:
            deps_key = ()
        return (stoich_key, deps_key)

    def _dedup_reactions(self, reactions: list[Reaction]) -> list[Reaction]:
        """Remove duplicate reactions, keeping the last occurrence of each key."""
        seen: dict[tuple, int] = {}
        for i, rxn in enumerate(reactions):
            seen[self._reaction_key(rxn)] = i
        return [reactions[i] for i in sorted(seen.values())]

    def _repair_constraints(
        self, reactions: list[Reaction], n_species: int
    ) -> list[Reaction]:
        """Apply post-hoc structural repairs to satisfy configured constraints.

        Args:
            reactions: Initial list of reactions to repair.
            n_species: Number of species.

        Returns:
            Repaired list of reactions (deduplicated after each repair step).
        """
        cfg = self._config
        reactions = list(reactions)

        # Ensure every species participates in at least one reaction
        participating = self._participating_species(reactions, n_species)
        for s in range(n_species):
            if s not in participating:
                reactant_vec = torch.zeros(n_species)
                reactant_vec[s] = 1.0
                net_stoich = -reactant_vec
                rate = self._sample_rate_constant()
                propensity = mass_action(
                    rate_constant=rate, reactant_stoichiometry=reactant_vec
                )
                replace_idx = int(torch.randint(0, len(reactions), (1,)).item())
                reactions[replace_idx] = Reaction(
                    stoichiometry=net_stoich, propensity=propensity
                )

        # Ensure at least one zero-order production reaction
        if cfg.require_production and not self._has_production(reactions):
            s = int(torch.randint(0, n_species, (1,)).item())
            rate = self._sample_rate_constant()
            net_stoich = torch.zeros(n_species)
            net_stoich[s] = 1.0
            replace_idx = int(torch.randint(0, len(reactions), (1,)).item())
            reactions[replace_idx] = Reaction(
                stoichiometry=net_stoich, propensity=constant_rate(k=rate)
            )

        # Ensure every species has at least one degradation path
        if cfg.require_degradation:
            degraded = self._degraded_species(reactions, n_species)
            for s in range(n_species):
                if s not in degraded:
                    reactant_vec = torch.zeros(n_species)
                    reactant_vec[s] = 1.0
                    net_stoich = -reactant_vec
                    rate = self._sample_rate_constant()
                    reactions.append(
                        Reaction(
                            stoichiometry=net_stoich,
                            propensity=mass_action(
                                rate_constant=rate,
                                reactant_stoichiometry=reactant_vec,
                            ),
                        )
                    )

        return self._dedup_reactions(reactions)
