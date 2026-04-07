"""Experiment config registry.

Maps CLI-friendly string names to config classes. Import and register
each experiment config here. Scripts use --config NAME to select.
"""

from __future__ import annotations

from experiments.configs.base import BaseExperimentConfig

_REGISTRY: dict[str, type[BaseExperimentConfig]] = {}


def register(name: str, config_cls: type[BaseExperimentConfig]) -> None:
    """Register an experiment config class under a string name.

    Args:
        name: CLI-friendly identifier (e.g. "mass_action_3s").
        config_cls: Config class to register.

    Raises:
        ValueError: If name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Config '{name}' is already registered.")
    _REGISTRY[name] = config_cls


def get_config(name: str) -> BaseExperimentConfig:
    """Instantiate a registered config by name.

    Args:
        name: Registered config name (e.g. "mass_action_3s").

    Returns:
        A fresh config instance with default values.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return _REGISTRY[name]()


def available_configs() -> list[str]:
    """Return sorted list of registered config names."""
    return sorted(_REGISTRY.keys())


# ── Register all experiments ──────────────────────────────────────────────────
from experiments.configs.mass_action_3s import (  # noqa: E402
    MassAction3sConfig,
    MassAction3sV3Config,
    MassAction3sV4Config,
    MassAction3sV5Config,
    MassAction3sV7Config,
    MassAction3sV8Config,
)

register("mass_action_3s", MassAction3sConfig)
register("mass_action_3s_v3", MassAction3sV3Config)
register("mass_action_3s_v4", MassAction3sV4Config)
register("mass_action_3s_v5", MassAction3sV5Config)
register("mass_action_3s_v7", MassAction3sV7Config)
register("mass_action_3s_v8", MassAction3sV8Config)
