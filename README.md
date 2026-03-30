# crn_surrogate

Neural surrogate simulator for Chemical Reaction Networks (CRNs).

Given a CRN defined by its stoichiometry, propensity kinetics, and an initial molecular state, the model learns to produce stochastic trajectories that approximate Gillespie SSA ground truth — without running the SSA at inference time.

## Architecture

```
CRN (symbolic)
    │
    ▼  crn_to_tensor_repr()
CRNTensorRepr  ──────────────────────────────────────────────────────────────────┐
    │                                                                             │
    ▼                                                                             │
BipartiteGNNEncoder                                                               │
  ├── SpeciesEmbedding    (concentration + learnable identity)                    │
  ├── ReactionEmbedding   (propensity type embed + parameter projection)          │
  └── L × MessagePassingLayer  (sum or attention-weighted)                        │
        alternates rxn→species / species→rxn messages                            │
        ▼                                                                         │
    CRNContext  (species_embeddings, reaction_embeddings, context_vector)         │
        │                                                                         │
        ▼                                                                         │
CRNNeuralSDE  (drift f and diffusion g, each a ConditionedMLP)                   │
  dX = f(X,t; ctx) dt + g(X,t; ctx) dW                                          │
        │                                                                         │
        ▼                                                                         │
EulerMaruyamaSolver  →  predicted Trajectory                                     │
                                                                                  │
                      Gillespie SSA  →  ground-truth Trajectory  ←───────────────┘
                                            ▼
                              GaussianTransitionNLL   (teacher forcing)
                              CombinedTrajectoryLoss  (full rollout)
```

**Key design choices:**

- The encoder is CRN-agnostic at the tensor level. The symbolic-to-tensor conversion (`crn_to_tensor_repr`) is an explicit boundary; the encoder never imports the `CRN` class.
- Edge features encode stoichiometric structure: `NET_CHANGE`, `IS_STOICHIOMETRIC`, `IS_DEPENDENCY`. The `IS_DEPENDENCY` flag distinguishes catalytic species (zero net stoichiometry but non-zero propensity influence) from stoichiometric participants.
- The diffusion matrix has shape `(n_species, n_reactions)`, matching the Chemical Langevin Equation where each reaction drives one independent Wiener process.
- Both drift and diffusion networks are `ConditionedMLP` instances: FiLM modulation is applied at every hidden layer (not just the output), so the network can compute context-dependent intermediate features. Depth is controlled by `SDEConfig.n_hidden_layers`.

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

## CRN definitions

Supported propensity types (all serializable to/from flat tensors):

| Class | Factory | Kinetics |
|-------|---------|----------|
| `ConstantRateParams` | `constant_rate(k)` | `a = k` |
| `MassActionParams` | `mass_action(k, reactant_stoich)` | `a = k * ∏ Xᵢ^Rᵢ` |
| `HillParams` | `hill(v_max, k_m, n, species_index)` | `a = V·Xₛⁿ / (Kⁿ + Xₛⁿ)` |
| `EnzymeMichaelisMentenParams` | `enzyme_michaelis_menten(k_cat, k_m, enzyme_idx, substrate_idx)` | `a = k_cat·E·S / (Kₘ + S)` |

Built-in example CRNs: `birth_death`, `lotka_volterra`, `schlogl`, `toggle_switch`, `simple_mapk_cascade`.

Analytical reference functions for validation are available for `birth_death` and `lotka_volterra` via `birth_death_analytical(k_birth, k_death)` and `lotka_volterra_analytical(...)`, which return drift/diffusion callables and stationary moments.

Custom propensities must be implemented as callable classes with `.params` (returning a registered `Params` dataclass) and `.species_dependencies` (returning `frozenset[int]`). Raw lambdas are not serializable.

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

Test suite covers: CRN construction and propensities, Gillespie SSA simulator, bipartite graph utilities, all encoder components (embeddings, message-passing layers, full encoder), `FiLMLayer`, `ConditionedMLP`, neural SDE (drift/diffusion shapes, non-negativity, gradient flow), configs, loss functions (NLL correctness vs reference loop, gradient flow, masking, speedup), dataset collation, trainer, profiler, and an end-to-end integration test.

## Project structure

```
src/crn_surrogate/
├── configs/
│   ├── model_config.py       # EncoderConfig, SDEConfig, ModelConfig
│   └── training_config.py    # TrainingConfig, TrainingMode, SchedulerType
├── crn/
│   ├── crn.py                # CRN class (stoichiometry + propensities)
│   ├── reaction.py           # Reaction, PropensityFn protocol
│   ├── propensities.py       # mass_action, hill, enzyme_michaelis_menten, constant_rate
│   └── examples.py           # birth_death, lotka_volterra, schlogl, toggle_switch,
│                             #   simple_mapk_cascade, birth_death_analytical,
│                             #   lotka_volterra_analytical
├── simulation/
│   ├── gillespie.py          # GillespieSSA (CRN-agnostic)
│   ├── interpolation.py      # interpolate_to_grid (zero-order hold)
│   └── trajectory.py         # Trajectory dataclass
├── encoder/
│   ├── tensor_repr.py        # crn_to_tensor_repr, tensor_repr_to_crn, CRNTensorRepr
│   ├── graph_utils.py        # EdgeFeature, EDGE_FEAT_DIM, BipartiteEdges, build_bipartite_edges
│   ├── embeddings.py         # SpeciesEmbedding, ReactionEmbedding
│   ├── message_passing.py    # SumMessagePassingLayer, AttentiveMessagePassingLayer
│   └── bipartite_gnn.py      # BipartiteGNNEncoder, CRNContext
├── simulator/
│   ├── film.py               # FiLMLayer (feature-wise linear modulation)
│   ├── conditioned_mlp.py    # ConditionedMLP (MLP with per-layer FiLM conditioning)
│   ├── neural_sde.py         # CRNNeuralSDE
│   └── sde_solver.py         # EulerMaruyamaSolver
├── data/
│   └── dataset.py            # TrajectoryItem, CRNTrajectoryDataset, CRNCollator
├── training/
│   ├── losses.py             # GaussianTransitionNLL, MeanMatchingLoss,
│   │                         #   VarianceMatchingLoss, CombinedTrajectoryLoss
│   ├── trainer.py            # Trainer, TrainingResult
│   └── profiler.py           # PhaseTimer, ProfileLogger, WandbLogger
└── evaluation/
    ├── rollout.py            # ModelEvaluator
    ├── dynamics.py           # DynamicsVisualizer, DynamicsProfile
    ├── residuals.py          # ResidualAnalyzer, ResidualReport
    └── trajectory.py         # TrajectoryComparator
tests/
notebooks/
├── 03a_encoder_comparison.ipynb   # sum vs. attentive message passing
├── 03b_loss_comparison.ipynb      # GaussianTransitionNLL vs. CombinedTrajectoryLoss
└── archive/
```

## License

MIT License © 2026 Jan Mikelson
