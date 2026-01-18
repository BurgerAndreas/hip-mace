# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIP-MACE is an implementation of MACE (Machine-learning Atomic Cluster Expansion) as a Hessian Interatomic Potential (HIP). HIPs predict the Hessian (second derivative of energy with respect to atom positions) in addition to forces and energy. This extends the base MACE architecture with learnable Hessian prediction using equivariant neural networks.

Paper: https://arxiv.org/abs/2509.21624

## Development Guidelines

- **Keep changes minimal**: Only modify what is necessary for the task. Avoid refactoring or "improving" unrelated code.
- **Use the uv virtual environment**: Always run commands with `uv run` or activate the venv first with `source .venv/bin/activate`.

## Commands

```bash
# Setup environment
uv sync

# Run tests
uv run tests/test_hip.py
uv run tests/test_equivariance.py

# Train with HIP (Hessian prediction)
uv run scripts/run_train.py --config=configs/horm_100.yaml --hip=true

# Train without HIP (energy and forces only)
uv run scripts/run_train.py --config=configs/horm_100.yaml

# Evaluate Hessian predictions
uv run scripts/eval_horm.py

# Data preparation
uv run scripts/download_horm_data_kaggle.py
uv run scripts/convert_lmdb_to_h5.py --in_file "data/sample_100.lmdb"
```

## Architecture

### Core Modules

- **`mace/modules/models.py`**: Main MACE model with HIP integration. `ScaleShiftMACE` is the production model. Forward pass calls `_predict_hessian_hip()` when `predict_hessian=True`.

- **`mace/modules/hip.py`**: Hessian-specific functionality:
  - `add_hessian_graph_batch()`: Builds extended graph for Hessian (uses larger cutoff than main model)
  - `blocks3x3_to_hessian()`: Assembles Hessian from 3x3 blocks using vectorized `index_add_`
  - `irreps_to_cartesian_matrix()`: Converts irreducible representations to Cartesian 3x3 matrices

- **`mace/modules/blocks.py`**: Interaction blocks, readout blocks, radial embeddings

- **`mace/modules/loss.py`**: Loss functions including `mean_squared_error_hessian` and `mean_absolute_error_hessian`

- **`mace/tools/train.py`**: Training loop. Controls HIP via `predict_hessian` parameter in forward call.

### HIP Design

HIP extends MACE by adding:
1. **Hessian output head** with L=0,1,2 irreps (even parity, unlike forces which use odd parity)
2. **Separate graph construction** for Hessian with configurable `hessian_r_max` (typically larger than main `r_max`)
3. **Equivariant tensor products** combining node features with spherical harmonics for edge contributions
4. **Flattened Hessian assembly** using precomputed indices for GPU-efficient batched construction

Key HIP parameters in model config:
- `hip`: Enable Hessian prediction
- `hessian_feature_dim`: Channels per L-component
- `hessian_r_max`: Cutoff radius for Hessian graph
- `hessian_edge_lmax`: Maximum spherical harmonic degree
- `hessian_use_last_layer_only`: Use only final interaction layer (vs. aggregating all)
- `hessian_use_directional_encoding`, `hessian_separate_radial_network`, `hessian_use_edge_gates`: Advanced options

### Data Flow

1. **Input**: `AtomicData` (extends torch_geometric) with positions, forces, energy, hessian
2. **Processing**: Message passing through interaction blocks → node features
3. **HIP output**: Node features → tensor products with spherical harmonics → 3x3 blocks → assembled Hessian

### Configuration System

YAML configs in `configs/` with CLI override via `configargparse`. Key sections:
- Model: `hidden_irreps`, `r_max`, `max_ell`, `num_interactions`
- HIP: `hip`, `hessian_key`, `hessian_weight`
- Data: `train_file`, `valid_file` (supports `.h5`, `.lmdb`, `.xyz`)
- Loss: `loss`, `energy_weight`, `forces_weight`, `hessian_weight`

## Key Files for HIP Development

| File | Purpose |
|------|---------|
| `mace/modules/hip.py` | Hessian assembly & graph construction |
| `mace/modules/models.py` | MACE forward pass + `_predict_hessian_hip()` |
| `mace/modules/loss.py` | Hessian loss functions |
| `mace/tools/train.py` | Training loop with `predict_hessian` toggle |
| `mace/tools/model_script_utils.py` | Model configuration builder |
| `mace/data/atomic_data.py` | Data structures including Hessian |
| `tests/test_hip.py` | HIP functionality tests |
| `tests/test_equivariance.py` | E(3) equivariance verification |

## Data Keys

Reference properties use keys defined in `mace/tools/default_keys.py`:
- `energy_key`: Energy reference (default: "energy")
- `forces_key`: Forces reference (default: "forces")
- `hessian_key`: Hessian reference (default: "hessian")

## Loss Function

When `hip=True`, the loss automatically includes Hessian MAE weighted by `hessian_weight`. The loss is DDP-aware with proper global averaging.
