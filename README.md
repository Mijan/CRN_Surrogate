# crn_surrogate

Neural surrogate simulator for Chemical Reaction Networks (CRNs).

Given a CRN defined by its stoichiometry matrix and propensity parameters plus an initial molecular state, the model produces stochastic trajectories that approximate Gillespie SSA ground truth.

## Architecture

1. **CRN Encoder** — bipartite GNN over the species-reaction graph defined by the stoichiometry matrix; produces contextualized species and reaction embeddings.
2. **Neural SDE Surrogate** — drift and diffusion coefficients conditioned on the CRN encoder output via FiLM modulation; solved forward in time with Euler-Maruyama.

## Install

```bash
pip install -e .
```

## Quick start

See `notebooks/` for worked examples:

- `01_simulation.ipynb` — run Gillespie SSA and Neural SDE side-by-side
- `02_training_data.ipynb` — build a `CRNTrajectoryDataset`
- `03_training.ipynb` — full training loop with optional W&B logging

## Run tests

```bash
pytest
```

## Profiling & Weights & Biases

The `Trainer` automatically records per-batch phase timings (`forward`, `backward`) using `PhaseTimer` and writes two CSV files to `TrainingConfig.log_dir` after each epoch:

| File | Contents |
|------|----------|
| `profiler_batches.csv` | One row per training batch with `forward_s`, `backward_s`, GPU memory |
| `profiler_epochs.csv` | One row per phase per epoch: mean / std / min / max / total seconds |

To enable Weights & Biases logging, set `use_wandb=True` in `TrainingConfig`:

```python
from crn_surrogate.configs.training_config import TrainingConfig

config = TrainingConfig(
    use_wandb=True,
    wandb_project="my-project",   # optional, default "crn-surrogate"
    wandb_run_name="run-01",      # optional
)
```

This requires `wandb` to be installed (`pip install wandb`) and a prior `wandb login`.

## Project structure

```
src/crn_surrogate/
├── configs/          # Frozen dataclass configurations (ModelConfig, TrainingConfig)
├── data/             # CRN definitions, propensities, Gillespie SSA, dataset
├── encoder/          # Bipartite GNN encoder
├── simulator/        # Neural SDE + Euler-Maruyama solver
└── training/         # Loss functions, training loop, profiler
tests/
notebooks/
```

## License

MIT License © 2026 Jan Mikelson
