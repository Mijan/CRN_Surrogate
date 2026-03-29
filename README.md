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

```bash
python main.py
```

## Run tests

```bash
pytest
```

## Project structure

```
crn_surrogate/
├── configs/          # Frozen dataclass configurations
├── data/             # CRN definitions, propensities, Gillespie SSA, dataset
├── encoder/          # Bipartite GNN encoder
├── simulator/        # Neural SDE + Euler-Maruyama solver
└── training/         # Loss functions and training loop
tests/
main.py
```

## License

MIT License © 2026 Jan Mikelson
