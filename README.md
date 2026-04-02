# crn_surrogate

Neural surrogate simulator for Chemical Reaction Networks (CRNs).

Given a CRN defined by its stoichiometry, propensity kinetics, and an initial molecular state, the model learns to produce stochastic trajectories that approximate Gillespie SSA ground truth — without running the SSA at inference time.

## Architecture

```
CRN (symbolic)                     InputProtocol (experiment-level)
    │                                       │
    ▼  crn_to_tensor_repr()                 ▼
CRNTensorRepr  ────────────────┐    ProtocolEncoder  (DeepSets over PulseEvents)
    │                          │           │
    ▼                          │           ▼
BipartiteGNNEncoder            │    protocol_embedding  (d_protocol,)
  ├── SpeciesEmbedding          │           │
  │   (conc + identity +        │           │
  │    is_external flag)        │           │
  ├── ReactionEmbedding         │           │
  └── L × MessagePassingLayer   │           │
        ▼                       │           │
    CRNContext  ────────────────┴───────────┘
    (context_vector = mean-pool species + mean-pool reactions)
        │  concatenated with protocol_embedding → [ctx ; proto]
        ▼
CRNNeuralSDE  (drift f and diffusion g, each a ConditionedMLP)
  dX_int = f(X,t; [ctx;proto]) dt + g(X,t; [ctx;proto]) dW   ← internal species
  X_ext  = InputProtocol.evaluate(t)                          ← external species (clamped)
        │
        ▼
EulerMaruyamaSolver  →  predicted Trajectory
                                    ▼
              Gillespie SSA  →  ground-truth Trajectory
                                    ▼
                      GaussianTransitionNLL   (teacher forcing, masked to internal species)
                      CombinedTrajectoryLoss  (full rollout)
```

**Key design choices:**

- The encoder is CRN-agnostic at the tensor level. The symbolic-to-tensor conversion (`crn_to_tensor_repr`) is an explicit boundary; the encoder never imports the `CRN` class.
- Edge features encode stoichiometric structure: `NET_CHANGE`, `IS_STOICHIOMETRIC`, `IS_DEPENDENCY`. The `IS_DEPENDENCY` flag distinguishes catalytic species (zero net stoichiometry but non-zero propensity influence) from stoichiometric participants.
- The diffusion matrix has shape `(n_species, n_reactions)`, matching the Chemical Langevin Equation where each reaction drives one independent Wiener process.
- Both drift and diffusion networks are `ConditionedMLP` instances: FiLM modulation is applied at every hidden layer (not just the output), so the network can compute context-dependent intermediate features. Depth is controlled by `SDEConfig.n_hidden_layers`.
- External species (inputs) are separated from the CRN context. The `CRNContext` encodes CRN structure only; the `ProtocolEncoder` encodes the experimental protocol. The two are concatenated before conditioning the SDE, so the same trained model can be evaluated under any protocol without re-encoding the CRN.

## External inputs

External inputs model experimental protocols (microfluidic ligand pulses, optogenetic stimulation, chemical inducers) as pseudo-species whose dynamics are prescribed by a `PulseSchedule` rather than CRN kinetics. The same CRN trained once can be evaluated under any protocol at inference time.

### Data structures

**`PulseEvent`** — a single rectangular pulse active on `[t_start, t_end)`:

```python
from crn_surrogate.crn.inputs import PulseEvent

event = PulseEvent(t_start=10.0, t_end=20.0, amplitude=15.0)
```

**`PulseSchedule`** — a sorted, non-overlapping sequence of `PulseEvent`s for one input species. Evaluates in O(log n) via bisect:

```python
from crn_surrogate.crn.inputs import PulseSchedule

schedule = PulseSchedule(events=(event,), baseline=0.0)
schedule.evaluate(12.0)  # → 15.0  (inside pulse)
schedule.evaluate(25.0)  # → 0.0   (at baseline)
```

**`InputProtocol`** — maps global species indices to their `PulseSchedule`s. Represents the full experimental protocol for one simulation:

```python
from crn_surrogate.crn.inputs import InputProtocol

protocol = InputProtocol(schedules={1: schedule})
protocol.evaluate(12.0)  # → {1: 15.0}
protocol.breakpoints()   # → sorted list of all pulse start/end times
```

`EMPTY_PROTOCOL` is a singleton for CRNs with no external inputs. It is the default value of `TrajectoryItem.input_protocol`.

### Factory functions

| Factory | Returns | Description |
|---------|---------|-------------|
| `single_pulse(t_start, t_end, amplitude)` | `PulseSchedule` | One rectangular pulse |
| `repeated_pulse(period, duty_cycle, amplitude, n_pulses, t_start)` | `PulseSchedule` | Square wave with n repetitions |
| `step_sequence(times, amplitudes)` | `PulseSchedule` | Staircase of constant levels (times = transition points, amplitudes = level per interval) |
| `constant_input(amplitude, t_start, t_end)` | `PulseSchedule` | Constant input; `t_end=inf` means constant for the entire simulation |
| `random_protocol(t_max, n_pulses_range, duration_range, amplitude_range, ...)` | `PulseSchedule` | Random pulse train |
| `random_input_protocol(input_species_indices, t_max, ...)` | `InputProtocol` | Independent random protocols for multiple species |

`single_pulse`, `repeated_pulse`, `step_sequence`, `constant_input`, and `random_protocol` return `PulseSchedule`; wrap in `InputProtocol(schedules={species_idx: schedule})` to use with the neural pipeline. `random_input_protocol` returns a complete `InputProtocol` directly.

### CRN with external species

Declare external (input-controlled) species when constructing a `CRN`. External species must have zero net stoichiometry across all reactions — they can only appear as propensity dependencies:

```python
from crn_surrogate.crn.crn import CRN
from crn_surrogate.crn.reaction import Reaction
from crn_surrogate.crn.propensities import hill, mass_action
import torch

# Species: A (index 0, internal), I (index 1, external inducer)
crn = CRN(
    reactions=[
        Reaction(
            stoichiometry=torch.tensor([1, 0]),  # I has zero net change
            propensity=hill(v_max=5.0, k_m=10.0, hill_coefficient=2.0, species_index=1),
            name="A production",
        ),
        Reaction(
            stoichiometry=torch.tensor([-1, 0]),
            propensity=mass_action(0.2, torch.tensor([1.0, 0.0])),
            name="A degradation",
        ),
    ],
    species_names=["A", "I"],
    external_species=frozenset({1}),
)

crn.is_external   # → array([False, True])
```

`CRNTensorRepr.is_external` (a `(n_species,)` bool tensor) is derived automatically and passed to `SpeciesEmbedding`, where it adds a learned offset to the embeddings of external species.

The `BipartiteGraphBuilder` automatically excludes edges whose species endpoint is external from the bipartite graph, since external species do not participate in CRN kinetics.

### ProtocolEncoder

A DeepSets encoder that maps a batch of `InputProtocol` objects to fixed-size embedding vectors. The same model handles protocols with any number of events or species.

```python
from crn_surrogate.encoder.protocol_encoder import ProtocolEncoder
from crn_surrogate.configs.model_config import ProtocolEncoderConfig

protocol_cfg = ProtocolEncoderConfig(
    d_event=32,           # hidden dim of the per-event MLP
    d_protocol=64,        # output embedding dimension
    n_layers=2,           # hidden layers in the per-event MLP
    max_input_species=16, # embedding table size for input species
    species_embed_dim=8,  # per-species embedding dimension
)
encoder = ProtocolEncoder(protocol_cfg)

# Encode a batch of protocols → (batch, d_protocol)
emb = encoder([protocol_A, protocol_B, EMPTY_PROTOCOL])
# EMPTY_PROTOCOL always encodes to exactly zero (not the projection bias)
```

**Per-event feature vector:** for each `PulseEvent` with local species index `k` (0-indexed by sorted rank among species present in the protocol):
```
[species_embedding(k), t_start, t_end, amplitude, log(amplitude), duration, midpoint]
```

**Architecture:** Linear(raw_dim → d_event) → SiLU → [Linear(d_event → d_event) → SiLU] × (n_layers−1) → masked sum-pool → Linear(d_event → d_protocol).

The local species index mapping (global species 3, 7 → local 0, 1) is per-protocol and protocol-agnostic: the embedding learns "first input species", not a specific global identity. The GNN already encodes which global species connects where.

### Using protocol conditioning in the neural SDE

Set `SDEConfig.d_protocol > 0` (must match `ProtocolEncoderConfig.d_protocol`) to enable protocol conditioning. The protocol embedding is concatenated to the CRN context vector before each FiLM modulation step:

```python
from crn_surrogate.configs.model_config import SDEConfig, ProtocolEncoderConfig
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
from crn_surrogate.crn.inputs import InputProtocol, single_pulse
import torch

D_PROTOCOL = 64
sde_cfg = SDEConfig(d_model=32, d_protocol=D_PROTOCOL, n_noise_channels=crn.n_reactions)
protocol_cfg = ProtocolEncoderConfig(d_protocol=D_PROTOCOL)

sde = CRNNeuralSDE(sde_cfg, n_species=crn.n_species)
protocol_encoder = ProtocolEncoder(protocol_cfg)
solver = EulerMaruyamaSolver(sde_cfg)

# Encode CRN once; encode protocol once per experiment
crn_context = encoder(crn_repr, initial_state)
protocol = InputProtocol(schedules={1: single_pulse(10.0, 20.0, 15.0)})
protocol_emb = protocol_encoder([protocol])[0]  # (d_protocol,)

external_mask = torch.tensor([s in crn.external_species for s in range(crn.n_species)])

trajectory = solver.solve(
    sde=sde,
    initial_state=initial_state,
    crn_context=crn_context,
    t_span=torch.linspace(0.0, 30.0, 120),
    dt=0.1,
    protocol_embedding=protocol_emb,     # conditions drift/diffusion
    input_protocol=protocol,             # clamps external species at each step
    external_species_mask=external_mask, # required when input_protocol is set
)
```

At each integration step, the solver:
1. Clamps external species to `protocol.evaluate(t)` before computing drift/diffusion
2. Applies Euler-Maruyama to internal species only
3. Clips internal species to `>= 0` (if `clip_state=True`)
4. Overwrites external species with `protocol.evaluate(t + dt)` on the new state

When `d_protocol=0` (default), the SDE behaves identically to the pre-protocol version.

### Loss masking for external species

`GaussianTransitionNLL.compute` accepts an optional `mask` parameter restricting the species-dimension sum to internal species:

```python
from crn_surrogate.training.losses import GaussianTransitionNLL

loss_fn = GaussianTransitionNLL()
internal_mask = ~external_mask  # (n_species,) bool, True for internal

loss = loss_fn.compute(
    sde=sde,
    crn_context=crn_context,
    true_trajectory=ssa_traj,
    times=times,
    dt=0.1,
    mask=internal_mask,
    protocol_embedding=protocol_emb,
)
```

When `mask=None` all species contribute to the loss.

### Gillespie SSA with external inputs

Pass an `InputProtocol` to `GillespieSSA.simulate`. The simulator partitions the simulation at protocol breakpoints, applies species overrides at each interval boundary, and records state at every breakpoint so that `interpolate_to_grid` accurately reflects input transitions:

```python
from crn_surrogate.simulation.gillespie import GillespieSSA

ssa = GillespieSSA()
traj = ssa.simulate(
    crn=crn,
    initial_state=initial_state,
    t_max=30.0,
    input_protocol=protocol,
)
```

## Install

```bash
pip install -e .
```

Optional: install [Weights & Biases](https://wandb.ai) for experiment tracking:

```bash
pip install wandb && wandb login
```

## Quick start

```python
from crn_surrogate.crn.examples import birth_death
from crn_surrogate.encoder.tensor_repr import crn_to_tensor_repr
from crn_surrogate.configs.model_config import EncoderConfig, SDEConfig, ModelConfig
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder
from crn_surrogate.simulator.neural_sde import CRNNeuralSDE
from crn_surrogate.simulator.sde_solver import EulerMaruyamaSolver
import torch

crn = birth_death(k_birth=2.0, k_death=0.5)
crn_repr = crn_to_tensor_repr(crn)

model_config = ModelConfig(
    encoder=EncoderConfig(d_model=32, n_layers=2),
    sde=SDEConfig.from_crn(crn),        # n_noise_channels = n_reactions, n_hidden_layers = 2
)
encoder = BipartiteGNNEncoder(model_config.encoder)
sde = CRNNeuralSDE(model_config.sde, n_species=crn.n_species)
solver = EulerMaruyamaSolver(model_config.sde)

x0 = torch.tensor([5.0])
t_span = torch.linspace(0.0, 10.0, 50)
ctx = encoder(crn_repr, x0)
traj = solver.solve(sde, x0, ctx, t_span, dt=0.05)
# traj.states: (50, 1)
```

See `notebooks/` for worked examples:

- `notebooks/03a_encoder_comparison.ipynb` — sum vs. attentive message passing
- `notebooks/03b_loss_comparison.ipynb` — `GaussianTransitionNLL` vs. `CombinedTrajectoryLoss`
- `notebooks/05_external_inputs.ipynb` — pulse schedule visualization, Gillespie with external inputs
- `notebooks/06_neural_sde_with_inputs.ipynb` — encoder invariance, protocol encoding, full forward pass

## CRN definitions

Supported propensity types (all serializable to/from flat tensors):

| Class | Factory | Kinetics |
|-------|---------|----------|
| `ConstantRateParams` | `constant_rate(k)` | `a = k` |
| `MassActionParams` | `mass_action(k, reactant_stoich)` | `a = k * ∏ Xᵢ^Rᵢ` |
| `HillParams` | `hill(v_max, k_m, n, species_index)` | `a = V·Xₛⁿ / (Kⁿ + Xₛⁿ)` |
| `HillRepressionParams` | `hill_repression(k_max, k_half, n, species_index)` | `a = k_max / (1 + (Xₛ/K)ⁿ)` |
| `HillActivationRepressionParams` | `hill_activation_repression(k_max, k_act, n_act, act_idx, k_rep, n_rep, rep_idx)` | Sigmoid activation × sigmoid repression |
| `SubstrateInhibitionParams` | `substrate_inhibition(v_max, k_m, k_i, species_index)` | `a = V·Xₛ / (Kₘ + Xₛ + Xₛ²/Kᵢ)` |
| `EnzymeMichaelisMentenParams` | `enzyme_michaelis_menten(k_cat, k_m, enzyme_idx, substrate_idx)` | `a = k_cat·E·S / (Kₘ + S)` |

Built-in example CRNs: `birth_death`, `lotka_volterra`, `schlogl`, `toggle_switch`, `simple_mapk_cascade`.

Analytical reference functions for validation are available for `birth_death` and `lotka_volterra` via `birth_death_analytical(k_birth, k_death)` and `lotka_volterra_analytical(...)`, which return drift/diffusion callables and stationary moments.

Custom propensities must be implemented as callable classes with `.params` (returning a registered `Params` dataclass) and `.species_dependencies` (returning `frozenset[int]`). Raw lambdas are not serializable.

## Data generation

The `crn_surrogate.data.generation` package provides a full pipeline for generating curated training datasets from CRN motifs.

### Motif types

Eight elementary motifs are supported, each with a typed `Params` dataclass and a factory class:

| MotifType | Factory | Species | Reactions | Kinetics |
|-----------|---------|---------|-----------|----------|
| `BIRTH_DEATH` | `BirthDeathFactory` | 1 | 2 | Mass action + constant rate |
| `AUTO_CATALYSIS` | `AutoCatalysisFactory` | 1 | 3 | Autocatalytic + degradation |
| `NEGATIVE_AUTOREGULATION` | `NegativeAutoregulationFactory` | 2 | 4 | Hill repression self-regulation |
| `TOGGLE_SWITCH` | `ToggleSwitchFactory` | 2 | 4 | Mutual Hill repression |
| `ENZYMATIC_CATALYSIS` | `EnzymaticCatalysisFactory` | 3 | 4 | Michaelis-Menten |
| `INCOHERENT_FEEDFORWARD` | `IncoherentFeedforwardFactory` | 3 | 5 | Hill activation + repression |
| `REPRESSILATOR` | `RepressilatorFactory` | 3 | 6 | Cyclic Hill repression |
| `SUBSTRATE_INHIBITION` | `SubstrateInhibitionMotifFactory` | 2 | 4 | Substrate inhibition kinetics |

Motifs can be **composed** via `ComposedMotifFactory`, which merges two elementary CRNs by sharing a bridging species, expanding stoichiometry and remapping species indices.

### Quick generation example

```python
from crn_surrogate.data.generation import (
    DataGenerationPipeline,
    GenerationConfig,
    SamplingConfig,
    CurationConfig,
    default_tasks,
)

config = GenerationConfig(
    sampling=SamplingConfig(n_rejection_samples=5),
    curation=CurationConfig(),
    output_dir="data_cache/dataset",
    n_ssa_trajectories=32,
    simulation_time=20.0,
    n_timepoints=100,
)

pipeline = DataGenerationPipeline(config, tasks=default_tasks(target_per_motif=200))
summary = pipeline.run()
# Writes data_cache/dataset/dataset.pt and metadata.json
```

### Parameter sampling

`ParameterSampler` draws kinetic parameters from ranges co-located in each `Params` dataclass using `param_field(low, high, log_uniform=True)`. No separate range configuration is needed. Rejection sampling calls `factory.validate_params()` to discard invalid configurations.

### Curation

`ViabilityFilter` applies six criteria to the (M, T, n_species) SSA ensemble:

| Criterion | Description |
|-----------|-------------|
| NaN/Inf check | Rejects any trajectory containing non-finite values |
| Blowup check | Rejects if any species exceeds `blowup_threshold` |
| Stuck-at-zero | Rejects if all species are zero for the entire simulation |
| Low activity | Rejects if mean molecule count is below `min_mean_molecules` |
| Low CV | Rejects if coefficient of variation is below `min_cv` (no interesting dynamics) |
| Unbounded final | Rejects if mean final state exceeds `max_final_mean` |

### Generation tasks

```python
from crn_surrogate.data.generation import (
    GenerationTask,
    all_elementary_tasks,
    default_tasks,
    get_factory,
    MotifType,
)

# All 8 elementary motifs, 100 items each
tasks = all_elementary_tasks(target=100)

# Default mix: 8 elementary + composed variants
tasks = default_tasks(target_per_motif=200)

# Custom task
tasks = [
    GenerationTask(
        factory=get_factory(MotifType.REPRESSILATOR),
        target=500,
        label="repressilator",
    )
]
```

### Dataset format

The pipeline saves a `list[TrajectoryItem]` to `dataset.pt`. Each item contains:

| Field | Type | Description |
|-------|------|-------------|
| `crn_repr` | `CRNTensorRepr` | Tensor representation for the encoder |
| `initial_state` | `Tensor (n_species,)` | Initial molecule counts |
| `trajectories` | `Tensor (M, T, n_species)` | SSA ensemble on the time grid |
| `times` | `Tensor (T,)` | Shared time grid |
| `motif_label` | `str` | Motif type string (e.g. `"birth_death"`) |
| `cluster_id` | `int` | Integer cluster ID assigned by the pipeline |
| `params` | `dict` | Kinetic parameters used to generate this CRN |
| `input_protocol` | `InputProtocol` | Protocol applied during SSA simulation (defaults to `EMPTY_PROTOCOL`) |
| `internal_species_mask` | `Tensor (n_species,) \| None` | Bool mask; True for internal species. Precomputed from `crn.is_external`. `None` means all species are internal. |

Use `CRNTrajectoryDataset` and `CRNCollator` to load and batch:

```python
from crn_surrogate.data import CRNTrajectoryDataset, CRNCollator
from torch.utils.data import DataLoader

dataset = CRNTrajectoryDataset.from_file("data_cache/dataset/dataset.pt")
loader = DataLoader(dataset, batch_size=16, collate_fn=CRNCollator())
```

## Training

### Configuration

```python
from crn_surrogate.configs.training_config import TrainingConfig, TrainingMode, SchedulerType
from crn_surrogate.training.trainer import Trainer

train_config = TrainingConfig(
    lr=1e-3,
    max_epochs=200,
    batch_size=16,
    n_sde_samples=8,        # K parallel SDE rollouts per item (rollout modes only)
    n_ssa_samples=32,       # M ground-truth SSA trajectories per item
    dt=0.1,                 # Euler-Maruyama step size
    training_mode=TrainingMode.TEACHER_FORCING,
    scheduler_type=SchedulerType.COSINE,
    use_wandb=True,
    wandb_project="crn-surrogate",
    wandb_run_name="my-run",
)

trainer = Trainer(encoder, sde, model_config, train_config)
result = trainer.train(train_dataset, val_dataset)
```

`TrainingResult` fields:

| Field | Description |
|---|---|
| `train_losses` | Per-epoch mean training loss |
| `val_losses` | Rollout loss at each validation epoch |
| `val_nll_losses` | NLL loss at each validation epoch |
| `val_epochs` | Epoch indices where validation ran |
| `grad_norms` | Pre-clip gradient norm per epoch |
| `learning_rates` | LR at the end of each epoch |

### Training modes

| Mode | Loss used | Description |
|---|---|---|
| `TEACHER_FORCING` | `GaussianTransitionNLL` | Each SDE step starts from the true state. All M×(T-1) transitions batched into a single forward pass. Fast and stable. |
| `FULL_ROLLOUT` | `CombinedTrajectoryLoss` | Model generates full trajectories from t=0. Better long-horizon fidelity but noisier gradients. |
| `SCHEDULED_SAMPLING` | Both | Starts with teacher forcing; linearly transitions to full rollout between `scheduled_sampling_start_epoch` and `scheduled_sampling_end_epoch`. |

Validation always uses full rollout regardless of training mode, reporting both rollout loss and NLL.

### Loss functions

**`GaussianTransitionNLL`** (recommended for teacher forcing):

Under the Euler-Maruyama model, each one-step transition is Gaussian. The per-step NLL is:

```
NLL = ½ Σₛ [ (y_{s,t+1} − μₛ)² / σ²ₛ  +  log σ²ₛ ]
```

where `μₛ = yₛ + Fθ[s]·dt` and `σ²ₛ = ||Gθ[s,:]||²·dt`. All `M×(T-1)` transitions across M SSA trajectories are reshaped and passed through the drift/diffusion networks in a single batched call — no Python loop.

**`CombinedTrajectoryLoss`** (for rollout training):

Weighted sum of:
- `MeanMatchingLoss` (weight 1.0): MSE between `E_pred[X(t)]` and `E_true[X(t)]`
- `VarianceMatchingLoss` (weight 0.5): scaled MSE between `Var_pred[X(t)]` and `Var_true[X(t)]`

Both require `K ≥ 2` SDE samples and `M ≥ 2` SSA trajectories. The trainer raises `ValueError` rather than silently returning zero for insufficient samples.

### Checkpointing

Best validation-loss checkpoint is saved to `TrainingConfig.checkpoint_dir` as `best_epochN.pt` containing encoder and SDE state dicts.

### LR schedulers

- `SchedulerType.COSINE`: cosine annealing over `max_epochs`
- `SchedulerType.REDUCE_ON_PLATEAU`: halves LR after 5 stagnant validation epochs (default)

## Profiling & Weights & Biases

The `Trainer` automatically records per-batch phase timings using `PhaseTimer` and writes two CSV files to `TrainingConfig.log_dir` after each epoch:

| File | Contents |
|------|----------|
| `profiler_batches.csv` | One row per training batch: `forward_s`, `backward_s`, GPU memory |
| `profiler_epochs.csv` | One row per phase per epoch: mean / std / min / max / total seconds |

W&B logs `train_loss`, `val_loss`, `val_nll`, `grad_norm`, `lr`, and phase timings per epoch.

## Evaluation

The `crn_surrogate.evaluation` package provides four diagnostic utilities that work with any trained encoder + SDE pair.

### `ModelEvaluator` — generate rollouts

```python
from crn_surrogate.evaluation import ModelEvaluator

evaluator = ModelEvaluator(encoder, sde, model_config.sde)
sde_trajs = evaluator.rollout(crn_repr, initial_state, times, dt=0.1, n_rollouts=200)
# → (K, T, n_species)
```

### `TrajectoryComparator` — trajectory-level statistics

Compares SDE rollouts against SSA ground truth:

```python
from crn_surrogate.evaluation import TrajectoryComparator

comp = TrajectoryComparator(
    sde_trajs, ssa_trajs, times,
    analytical_mean=k_birth / k_death,
    analytical_var=k_birth / k_death,
)
fig = comp.plot_summary()          # 3-panel: mean±std, variance, sample paths
metrics = comp.metrics()
# keys: mean_mse, var_mse, final_mean, final_var, mean_sde_std, diffusion_collapsed
```

### `DynamicsVisualizer` — learned drift and diffusion

Compares learned `Fθ(x)` and `||Gθ[s,:]||₂` against analytical CLE reference functions:

```python
from crn_surrogate.evaluation import DynamicsVisualizer

vis = DynamicsVisualizer(encoder, sde, crn_repr, initial_state)
state_range = torch.linspace(0, 150, 100)

ax = vis.plot_drift(
    state_range,
    analytical_drift_fn=lambda x: k_birth - k_death * x,
)
ax = vis.plot_diffusion(
    state_range,
    analytical_diffusion_fn=lambda x: torch.sqrt(k_birth + k_death * x),
)
```

The CRN is encoded once at construction; all subsequent calls reuse the cached context. A single batched forward pass evaluates all N state points.

### `ResidualAnalyzer` — model diagnostic

Checks whether the Gaussian SDE assumption holds by computing standardized residuals:

```
zₛ = (y_{s,t+1} − μₛ) / σₛ
```

If the model is correct, `z ~ N(0, 1)`.

```python
from crn_surrogate.evaluation import ResidualAnalyzer

analyzer = ResidualAnalyzer(encoder, sde, crn_repr)
report = analyzer.compute_residuals(ssa_trajs, times, dt=0.1, initial_state=x0)

# report.mean:     (n_species,) — should be ≈ 0
# report.std:      (n_species,) — should be ≈ 1
# report.kurtosis: (n_species,) — should be ≈ 3

ax = analyzer.plot_histogram(report)   # histogram + N(0,1) overlay
ax = analyzer.plot_qq(report)          # QQ plot (Blom quantile formula)
```

## Encoder variants

`EncoderConfig.use_attention` selects the message-passing aggregation:

| `use_attention` | Layer | Aggregation |
|-----------------|-------|-------------|
| `False` (default) | `SumMessagePassingLayer` | Sum over incoming messages |
| `True` | `AttentiveMessagePassingLayer` | Attention-weighted sum (query/key projections, scatter softmax) |

## SDE network depth

`SDEConfig.n_hidden_layers` (default 2) controls the depth of both the drift and diffusion `ConditionedMLP` networks. Each hidden layer is followed by a FiLM conditioning step:

```python
SDEConfig.from_crn(crn, n_hidden_layers=4)  # deeper networks, FiLM at every hidden layer
```

## Run tests

```bash
pytest
pytest -m "not slow"   # skip GPU / timing benchmarks
```

Test suite covers: CRN construction and propensities, Gillespie SSA simulator, bipartite graph utilities, all encoder components (embeddings, message-passing layers, full encoder), `FiLMLayer`, `ConditionedMLP`, neural SDE (drift/diffusion shapes, non-negativity, gradient flow), configs, loss functions (NLL correctness vs reference loop, gradient flow, masking, speedup), dataset collation, trainer, profiler, data generation pipeline (motif factories, parameter sampling, curation, composer, pipeline integration), and an end-to-end integration test.

## Project structure

```
src/crn_surrogate/
├── configs/
│   ├── model_config.py       # EncoderConfig, SDEConfig, ProtocolEncoderConfig, ModelConfig
│   └── training_config.py    # TrainingConfig, TrainingMode, SchedulerType
├── crn/
│   ├── crn.py                # CRN class (stoichiometry + propensities + external_species)
│   ├── reaction.py           # Reaction, PropensityFn protocol
│   ├── propensities.py       # mass_action, hill, hill_repression,
│   │                         #   hill_activation_repression, substrate_inhibition,
│   │                         #   enzyme_michaelis_menten, constant_rate
│   ├── inputs.py             # PulseEvent, PulseSchedule, InputProtocol, EMPTY_PROTOCOL;
│   │                         #   single_pulse, repeated_pulse, step_sequence,
│   │                         #   constant_input, random_protocol, random_input_protocol
│   └── examples.py           # birth_death, lotka_volterra, schlogl, toggle_switch,
│                             #   simple_mapk_cascade, birth_death_analytical,
│                             #   lotka_volterra_analytical
├── simulation/
│   ├── gillespie.py          # GillespieSSA (breakpoint-aware, supports InputProtocol)
│   ├── interpolation.py      # interpolate_to_grid (zero-order hold)
│   └── trajectory.py         # Trajectory dataclass
├── encoder/
│   ├── tensor_repr.py        # crn_to_tensor_repr, tensor_repr_to_crn, CRNTensorRepr
│   ├── graph_utils.py        # EdgeFeature, EDGE_FEAT_DIM, BipartiteEdges,
│   │                         #   BipartiteGraphBuilder (filters external-species edges)
│   ├── embeddings.py         # SpeciesEmbedding (+ is_external flag), ReactionEmbedding
│   ├── message_passing.py    # SumMessagePassingLayer, AttentiveMessagePassingLayer
│   ├── bipartite_gnn.py      # BipartiteGNNEncoder, CRNContext
│   └── protocol_encoder.py   # ProtocolEncoder (DeepSets over PulseEvents)
├── simulator/
│   ├── film.py               # FiLMLayer (feature-wise linear modulation)
│   ├── conditioned_mlp.py    # ConditionedMLP (MLP with per-layer FiLM conditioning)
│   ├── neural_sde.py         # CRNNeuralSDE (+ protocol_embedding parameter)
│   └── sde_solver.py         # EulerMaruyamaSolver (+ external species clamping)
├── data/
│   ├── dataset.py            # TrajectoryItem (+ input_protocol, internal_species_mask),
│   │                         #   CRNTrajectoryDataset, CRNCollator
│   └── generation/
│       ├── configs.py        # SamplingConfig, CurationConfig, GenerationConfig
│       ├── motif_type.py     # MotifType enum (8 elementary + COMPOSED)
│       ├── motif_registry.py # get_factory() registry lookup
│       ├── task.py           # GenerationTask, all_elementary_tasks, default_tasks
│       ├── parameter_sampling.py  # ParameterSampler (log-uniform + rejection sampling)
│       ├── curation.py       # ViabilityFilter, CurationResult (6 criteria)
│       ├── composer.py       # CRNComposer, ComposedMotifFactory, CompositionSpec
│       ├── pipeline.py       # DataGenerationPipeline, DatasetSummary, MotifResult
│       └── motifs/
│           ├── base.py                      # MotifParams, MotifFactory, param_field
│           ├── birth_death.py               # BirthDeathFactory, BirthDeathParams
│           ├── auto_catalysis.py            # AutoCatalysisFactory, AutoCatalysisParams
│           ├── negative_autoregulation.py   # NegativeAutoregulationFactory
│           ├── toggle_switch.py             # ToggleSwitchFactory
│           ├── enzymatic_catalysis.py       # EnzymaticCatalysisFactory
│           ├── feedforward_loop.py          # IncoherentFeedforwardFactory
│           ├── repressilator.py             # RepressilatorFactory
│           └── substrate_inhibition_motif.py # SubstrateInhibitionMotifFactory
├── training/
│   ├── losses.py             # GaussianTransitionNLL (+ mask, protocol_embedding),
│   │                         #   MeanMatchingLoss, VarianceMatchingLoss,
│   │                         #   CombinedTrajectoryLoss
│   ├── trainer.py            # Trainer, TrainingResult
│   └── profiler.py           # PhaseTimer, ProfileLogger, WandbLogger
└── evaluation/
    ├── rollout.py            # ModelEvaluator
    ├── dynamics.py           # DynamicsVisualizer, DynamicsProfile
    ├── residuals.py          # ResidualAnalyzer, ResidualReport
    └── trajectory.py         # TrajectoryComparator
tests/
notebooks/
├── 01_simulation.ipynb            # Gillespie SSA walkthrough
├── 02_training_data.ipynb         # Data generation and curation
├── 03a_encoder_comparison.ipynb   # Sum vs. attentive message passing
├── 03b_loss_comparison.ipynb      # GaussianTransitionNLL vs. CombinedTrajectoryLoss
├── 04_data_generation.ipynb       # Full pipeline demo
├── 05_external_inputs.ipynb       # PulseSchedule visualization, Gillespie with inputs
└── 06_neural_sde_with_inputs.ipynb  # Protocol encoding, neural SDE forward pass demo
```

## License

MIT License © 2026 Jan Mikelson
