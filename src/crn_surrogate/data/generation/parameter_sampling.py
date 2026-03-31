"""Log-uniform and uniform parameter sampler for CRN motif factories."""

from __future__ import annotations

import math
import random
from typing import TypeVar

from crn_surrogate.data.generation.configs import SamplingConfig
from crn_surrogate.data.generation.motifs.base import (
    MotifFactory,
    ParameterRange,
    extract_parameter_ranges,
)

P = TypeVar("P")


class ParameterSampler:
    """Draws random typed parameter instances for a given MotifFactory.

    Rate parameters are sampled log-uniformly (equal probability density per
    decade). Hill coefficients and other non-rate parameters with
    log_uniform=False are sampled uniformly. Initial state values are sampled
    as uniform integers via sample_initial_states().

    Motif-specific inter-parameter constraints are enforced via rejection
    sampling with a fixed retry budget using factory.validate_params().
    """

    def __init__(self, config: SamplingConfig) -> None:
        """Args:
        config: Sampling configuration including seed.
        """
        self._config = config
        self._rng = random.Random(config.random_seed)

    def sample(
        self,
        factory: MotifFactory,
        n_samples: int,
    ) -> list:
        """Sample n_samples typed parameter instances for the given factory.

        For ComposedMotifFactory instances, delegates to sample_composed().
        For elementary factories, samples from param_field() metadata.

        Args:
            factory: Motif factory defining parameter ranges and constraints.
            n_samples: Number of samples to draw.

        Returns:
            List of params instances satisfying factory.validate_params().
        """
        from crn_surrogate.data.generation.composer import ComposedMotifFactory

        if isinstance(factory, ComposedMotifFactory):
            return self.sample_composed(factory, n_samples)
        return self._sample_elementary(factory, n_samples)

    def _sample_elementary(
        self,
        factory: MotifFactory[P],
        n_samples: int,
    ) -> list[P]:
        """Sample n_samples typed params from an elementary (non-composed) factory.

        Args:
            factory: Elementary motif factory.
            n_samples: Number of samples to draw.

        Returns:
            List of typed params instances.
        """
        ranges = extract_parameter_ranges(factory.params_type)
        results: list[P] = []
        for _ in range(n_samples):
            params = self._sample_one(factory, ranges)
            results.append(params)
        return results

    def sample_composed(
        self,
        factory: "ComposedMotifFactory",  # type: ignore[name-defined]
        n_samples: int,
    ) -> list:
        """Sample upstream and downstream params independently for a composed factory.

        Args:
            factory: Composed motif factory.
            n_samples: Number of samples to draw.

        Returns:
            List of ComposedParams with independently sampled sub-params.
        """
        from crn_surrogate.data.generation.composer import ComposedParams

        up_params = self._sample_elementary(factory._spec.upstream_factory, n_samples)
        down_params = self._sample_elementary(factory._spec.downstream_factory, n_samples)
        return [ComposedParams(up, down) for up, down in zip(up_params, down_params)]

    def sample_initial_states(
        self,
        factory: MotifFactory,
        n_samples: int,
    ) -> list[dict[str, int]]:
        """Sample initial states from factory.initial_state_ranges().

        Args:
            factory: Motif factory defining initial state ranges.
            n_samples: Number of samples to draw.

        Returns:
            List of dicts mapping species name to initial integer count.
        """
        ranges = factory.initial_state_ranges()
        results: list[dict[str, int]] = []
        for _ in range(n_samples):
            state = {
                name: self._rng.randint(r.low, r.high)
                for name, r in ranges.items()
            }
            results.append(state)
        return results

    def _sample_one(
        self,
        factory: MotifFactory[P],
        ranges: dict[str, ParameterRange],
    ) -> P:
        """Sample one typed params instance, with rejection for motif-specific constraints.

        Args:
            factory: Motif factory used for params construction and validation.
            ranges: Parameter range specification from extract_parameter_ranges().

        Returns:
            A single valid params instance.

        Raises:
            RuntimeError: If a valid set of parameters cannot be found within the
                retry budget.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            sampled: dict[str, float] = {}
            for name, r in ranges.items():
                if r.log_uniform:
                    log_val = self._rng.uniform(math.log(r.low), math.log(r.high))
                    sampled[name] = math.exp(log_val)
                else:
                    sampled[name] = self._rng.uniform(r.low, r.high)
            params = factory.params_from_dict(sampled)
            try:
                factory.validate_params(params)
                return params
            except ValueError:
                continue
        raise RuntimeError(
            f"Could not sample valid parameters for {factory.motif_type} "
            f"after {max_attempts} attempts"
        )
