"""Log-uniform and uniform parameter sampler for CRN motif factories."""

from __future__ import annotations

import math
import random

from crn_surrogate.data.generation.configs import SamplingConfig
from crn_surrogate.data.generation.motifs.base import MotifFactory, MotifParameterRanges


class ParameterSampler:
    """Draws random parameter dicts for a given MotifFactory.

    Rate parameters are sampled log-uniformly (equal probability density per
    decade). Hill coefficients are sampled uniformly. Initial state values are
    sampled as uniform integers.

    Motif-specific inter-parameter constraints are enforced via rejection
    sampling with a fixed retry budget.
    """

    def __init__(self, config: SamplingConfig) -> None:
        """Args:
        config: Sampling configuration including seed and default sample count.
        """
        self._config = config
        self._rng = random.Random(config.random_seed)

    def sample(
        self,
        factory: MotifFactory,
        n_samples: int | None = None,
    ) -> list[dict[str, float]]:
        """Sample n_samples parameter dicts for the given factory.

        Args:
            factory: Motif factory defining parameter ranges and constraints.
            n_samples: Number of samples to draw. Defaults to
                config.n_samples_per_motif if None.

        Returns:
            List of parameter dicts, each satisfying factory.check_constraints().
        """
        n = n_samples if n_samples is not None else self._config.n_samples_per_motif
        ranges = factory.parameter_ranges()
        results = []
        for _ in range(n):
            params = self._sample_one(factory, ranges)
            results.append(params)
        return results

    def _sample_one(
        self,
        factory: MotifFactory,
        ranges: MotifParameterRanges,
    ) -> dict[str, float]:
        """Sample one parameter dict, with rejection for motif-specific constraints.

        Args:
            factory: Motif factory used for constraint checking.
            ranges: Parameter range specification.

        Returns:
            A single valid parameter dict.

        Raises:
            RuntimeError: If a valid set of parameters cannot be found within the
                retry budget.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            params: dict[str, float] = {}
            for name, (lo, hi) in ranges.rate_ranges.items():
                log_val = self._rng.uniform(math.log(lo), math.log(hi))
                params[name] = math.exp(log_val)
            for name, (lo, hi) in ranges.hill_coefficient_ranges.items():
                params[name] = self._rng.uniform(lo, hi)
            for name, (lo, hi) in ranges.initial_state_ranges.items():
                params[name] = float(self._rng.randint(lo, hi))
            if factory.check_constraints(params):
                return params
        raise RuntimeError(
            f"Could not sample valid parameters for {factory.motif_type()} "
            f"after {max_attempts} attempts"
        )
