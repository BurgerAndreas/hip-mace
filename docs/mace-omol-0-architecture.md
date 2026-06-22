# MACE-OMOL-0 Architecture

Downloaded and inspected **MACE-OMOL-0 extra-large** (`MACE-omol-0-extra-large-1024.model`, 403 MB).

**Checkpoint path:** `/home/anburger/.cache/mace/MACE-omol-0-extra-large-1024.model`

**Download URL:** https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model

## Summary

| Property | Value |
|---|---|
| **Model class** | `ScaleShiftMACE` |
| **Total parameters** | **52,365,482** (~52.4M) |
| **Checkpoint size** | 403 MB |
| **Elements** | 83 (Z=1–83, H–Bi) |
| **Head** | `omol` |

## Architecture

| Hyperparameter | Value |
|---|---|
| **Message-passing layers** | 3 (`num_interactions=3`) |
| **Interaction block** | `RealAgnosticResidualNonLinearInteractionBlock` (all 3 layers) |
| **Hidden irreps** | `1024x0e + 1024x1o + 1024x2e` |
| **Channels (scalar 0e)** | 1024 |
| **max_L (hidden)** | 2 |
| **Node feature dim** | 9216 per atom |
| **Edge irreps** | `128x0e + 128x1o + 128x2e` (1152-dim per edge) |
| **Spherical harmonics** | `max_ell=3` |
| **Body-order correlation** | 2 |
| **Cutoff radius** | 6.0 Å |
| **Radial basis** | 8 Bessel + 5 polynomial cutoff |
| **Radial MLP** | `[128, 128, 128]` |
| **Readout** | `NonLinearBiasReadoutBlock` with `MLP_irreps=16x0e`, gate=SiLU |
| **avg_num_neighbors** | 30.0 |
| **apply_cutoff** | False |
| **use_agnostic_product** | True |
| **use_last_readout_only** | True |
| **use_embedding_readout** | True |

## Charge / Spin Embeddings (OMOL-specific)

Graph-level categorical embeddings fused into node features:

- **total_charge**: 201 classes → 1024-dim embedding
- **total_spin**: 101 classes → 1024-dim embedding
- Projected via `Linear(2048→1024)` + SiLU

`joint_embedding` accounts for **2.4M params** of the total.

## Parameter Breakdown by Module

| Module | Parameters | Share |
|---|---|---|
| `interactions` (3 layers) | 42,470,281 | ~81% |
| `products` (3 layers) | 7,386,112 | ~14% |
| `joint_embedding` | 2,406,400 | ~4.6% |
| `node_embedding` | 84,992 | |
| `readouts` | 16,673 | |
| `embedding_readout` | 1,024 | |

Per-layer interaction params: 18.95M → 12.81M → 10.71M (first layer is largest due to edge→hidden projection).

## Elements

Atomic numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83

## Reproduce

```bash
cd /lustre/fsw/portfolios/nvr/users/anburger/hip-mace && uv run python -c "
import torch; _o=torch.load; torch.load=lambda *a,**kw: _o(*a,**{**kw,'weights_only':False})
from mace.calculators.foundations_models import mace_omol
from mace.tools.scripts_utils import extract_config_mace_model
model = mace_omol(return_raw_model=True, device='cpu')
print(f'params: {sum(p.numel() for p in model.parameters()):,}')
for k,v in extract_config_mace_model(model).items():
    if k in ('hidden_irreps','num_interactions','r_max','max_ell','embedding_specs','heads'): print(k,':',v)
"
```

The `torch.load` patch is needed because PyTorch 2.6+ defaults `weights_only=True`, which breaks e3nn's internal constants load.

## Notes

- The README lists **89 elements covered** in the OMOL dataset; this checkpoint trains on **83 element types** (H through Bi).
- Training: OMOL dataset, ωB97M-VV10 DFT.
- No HIP/Hessian head, no pair repulsion, no CuEq acceleration config.
