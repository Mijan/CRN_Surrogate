"""Builder functions that translate Hydra config into library config objects."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from crn_surrogate.configs.model_config import EncoderConfig, ModelConfig, SDEConfig
from crn_surrogate.configs.solver_config import SolverConfig
from crn_surrogate.configs.training_config import (
    SchedulerType,
    TrainingConfig,
    TrainingMode,
)
from crn_surrogate.measurement.config import (
    MeasurementConfig,
    NoiseConfig,
    NoiseMode,
    NoiseSharing,
)


def build_encoder_config(cfg: DictConfig) -> EncoderConfig:
    """Build EncoderConfig from Hydra config.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        EncoderConfig instance.
    """
    m = cfg.model
    return EncoderConfig(
        d_model=m.d_model,
        n_layers=m.n_encoder_layers,
        use_attention=True,
        context_dropout=m.context_dropout,
    )


def build_sde_config(cfg: DictConfig) -> SDEConfig:
    """Build SDEConfig from Hydra config.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        SDEConfig instance.
    """
    m = cfg.model
    return SDEConfig(
        d_model=m.d_model,
        d_hidden=m.d_hidden,
        n_noise_channels=m.max_n_reactions,
        n_hidden_layers=m.n_sde_hidden_layers,
        clip_state=cfg.solver.clip_state,
        d_protocol=m.d_protocol,
        mlp_dropout=m.mlp_dropout,
    )


def _build_measurement_config(cfg: DictConfig) -> MeasurementConfig:
    """Build MeasurementConfig from Hydra config.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        MeasurementConfig instance.
    """
    meas = cfg.measurement
    mode = NoiseMode.LEARNED if meas.noise_mode == "learned" else NoiseMode.FIXED
    sharing = (
        NoiseSharing.SHARED
        if meas.noise_sharing == "shared"
        else NoiseSharing.PER_SPECIES
    )
    return MeasurementConfig(
        noise=NoiseConfig(mode=mode, sharing=sharing, init_value=meas.noise_init_eps),
        min_variance=meas.min_variance,
    )


def build_model_config(cfg: DictConfig) -> ModelConfig:
    """Build ModelConfig from Hydra config.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        ModelConfig instance.
    """
    return ModelConfig(
        encoder=build_encoder_config(cfg),
        sde=build_sde_config(cfg),
        measurement=_build_measurement_config(cfg),
    )


def build_training_config(cfg: DictConfig, *, use_wandb: bool = True) -> TrainingConfig:
    """Build TrainingConfig from Hydra config.

    Args:
        cfg: Fully resolved Hydra config.
        use_wandb: Whether to enable W&B logging.

    Returns:
        TrainingConfig instance.
    """
    t = cfg.training
    sched = (
        SchedulerType.COSINE
        if t.scheduler_type == "cosine"
        else SchedulerType.REDUCE_ON_PLATEAU
    )
    return TrainingConfig(
        lr=t.lr,
        max_epochs=t.max_epochs,
        batch_size=t.batch_size,
        n_ssa_samples=t.n_ssa_samples,
        dt=t.dt,
        val_every=t.val_every,
        grad_clip_norm=t.grad_clip_norm,
        scheduler_type=sched,
        training_mode=TrainingMode.TEACHER_FORCING,
        checkpoint_every=t.checkpoint_every,
        use_wandb=use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_run_name=f"{cfg.experiment_name}_train",
        num_workers=t.num_workers,
        shuffle_train=t.shuffle_train,
        gpu_memory_fraction=t.gpu_memory_fraction,
    )


def build_step_loss(cfg: DictConfig, device: torch.device):
    """Build the appropriate per-transition loss for the training regime.

    Deterministic (ODE) models get MSEStepLoss.
    Stochastic (SDE) models get NLLStepLoss with a MeasurementModel.

    Args:
        cfg: Fully resolved Hydra config.
        device: Target device for measurement model parameters.

    Returns:
        BatchedStepLoss instance (MSEStepLoss or NLLStepLoss).
    """
    from crn_surrogate.training.losses import MSEStepLoss, NLLStepLoss

    if cfg.solver.deterministic:
        return MSEStepLoss()

    from crn_surrogate.measurement.direct import DirectObservation

    meas_config = _build_measurement_config(cfg)
    measurement_model = DirectObservation.from_config(
        meas_config,
        n_species=cfg.model.max_n_reactions,
    ).to(device)
    return NLLStepLoss(
        measurement_model=measurement_model,
        min_variance=cfg.measurement.min_variance,
    )


def build_model(cfg: DictConfig, device: torch.device):
    """Instantiate NeuralDrift or NeuralSDE based on solver.deterministic.

    Args:
        cfg: Fully resolved Hydra config.
        device: Target device.

    Returns:
        NeuralDrift or NeuralSDE instance moved to device.
    """
    sde_config = build_sde_config(cfg)
    n_species = cfg.model.max_n_species
    if cfg.solver.deterministic:
        from crn_surrogate.simulator.neural_sde import NeuralDrift

        return NeuralDrift(sde_config, n_species).to(device)
    else:
        from crn_surrogate.simulator.neural_sde import NeuralSDE

        return NeuralSDE(sde_config, n_species).to(device)


def build_simulator(cfg: DictConfig):
    """Instantiate EulerODESolver or EulerMaruyamaSolver.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        EulerODESolver or EulerMaruyamaSolver instance.
    """
    from crn_surrogate.simulator.state_transform import get_state_transform

    solver_config = SolverConfig(clip_state=cfg.solver.clip_state)
    transform = get_state_transform(cfg.solver.use_log1p)
    if cfg.solver.deterministic:
        from crn_surrogate.simulator.ode_solver import EulerODESolver

        return EulerODESolver(solver_config, state_transform=transform)
    else:
        from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver

        return EulerMaruyamaSolver(solver_config, state_transform=transform)


def build_dataset_generator_config(cfg: DictConfig):
    """Build MassActionGeneratorConfig from Hydra dataset config.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        MassActionGeneratorConfig instance.
    """
    from crn_surrogate.data.generation.mass_action_generator import (
        MassActionGeneratorConfig,
        RandomTopologyConfig,
    )

    d = cfg.dataset
    return MassActionGeneratorConfig(
        topology=RandomTopologyConfig(
            n_species_range=tuple(d.topology.n_species_range),
            n_reactions_range=tuple(d.topology.n_reactions_range),
            max_reactant_order=d.topology.max_reactant_order,
            max_product_count=d.topology.max_product_count,
        ),
        rate_constant_range=tuple(d.rate_constant_range),
    )


def build_data_simulator(cfg: DictConfig):
    """Instantiate the appropriate DataSimulator for dataset generation.

    Args:
        cfg: Fully resolved Hydra config.

    Returns:
        ODESimulator if cfg.solver.deterministic, else FastSSASimulator or SSASimulator.
    """
    from crn_surrogate.simulation.data_simulator import (
        FastSSASimulator,
        ODESimulator,
        SSASimulator,
    )

    if cfg.solver.deterministic:
        return ODESimulator()

    timeout = cfg.generation.sim_timeout
    if cfg.generation.use_fast_ssa:
        try:
            return FastSSASimulator(timeout=timeout)
        except ImportError:
            print("Numba not available, falling back to standard SSA.")
    return SSASimulator(timeout=timeout)


def select_device(device_str: str) -> torch.device:
    """Resolve 'auto', 'cpu', 'cuda', 'mps' to a torch.device.

    Args:
        device_str: One of "auto", "cpu", "cuda", "mps".

    Returns:
        Selected torch.device.
    """
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
