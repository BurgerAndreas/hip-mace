# Guide: Adding HIP Hessian Prediction to an Equivariant MLIP

This guide explains how to modify a standard equivariant Machine Learning Interatomic Potential (MLIP) that uses spherical harmonics to predict Hessians using the HIP (Hessian Interatomic Potential) approach. We use MACE as the reference implementation.

## Prerequisites

Your MLIP should have:
- E(3)-equivariant message passing with spherical harmonics
- Node features with irreducible representations (irreps)
- Tensor products for combining node and edge features

## Overview of Changes

| Component | Standard MLIP | With HIP |
|-----------|--------------|----------|
| Graph | Single graph with cutoff `r_max` | Additional Hessian graph with larger `hessian_r_max` |
| Output | Energy (scalar), Forces (vector) | + Hessian (rank-2 tensor) |
| Edge features | Aggregated to nodes | Also extracted per-edge before aggregation |
| Output irreps | `0e` (energy), `1o` (forces) | + `0e + 1e + 2e` (Hessian) |

## Step-by-Step Implementation

### Step 1: Understand the Hessian Structure

The Hessian matrix H has shape `(3N, 3N)` where N is the number of atoms:

```
H = ∂²E/∂r∂r = | H_11  H_12  ...  H_1N |
               | H_21  H_22  ...  H_2N |
               | ...   ...   ...  ...  |
               | H_N1  H_N2  ...  H_NN |
```

Each block `H_ij` is a `(3, 3)` matrix. HIP predicts:
- **Diagonal blocks** `H_ii`: From node features (self-interaction)
- **Off-diagonal blocks** `H_ij`: From edge features (pair interaction)

The Hessian is symmetric: `H_ij = H_ji^T`

### Step 2: Determine Required Irreps for Hessian

The Hessian is a **rank-2 symmetric tensor** with **even parity** under inversion.

A 3×3 matrix decomposes into irreducible representations:
- `0e`: Trace (1 component) - isotropic part
- `1e`: Antisymmetric part (3 components) - but zero for symmetric Hessian
- `2e`: Symmetric traceless part (5 components) - anisotropic part

For a symmetric matrix, we need: **`1x0e + 1x1e + 1x2e`** = 9 components

> **Why even parity?** The Hessian `∂²E/∂r_i∂r_j` involves two position derivatives. Under inversion, each `∂/∂r` flips sign, so `∂²/∂r∂r` is even. This differs from forces (`1o`) which are odd.

### Step 3: Ensure Your Model Produces Even-Parity Features

Standard MLIPs often have node features like `Cx0e + Cx1o + Cx2e` (scalars, vectors, matrices).

To produce `0e + 1e + 2e` output, you need tensor products that generate even L=1:

```python
# Tensor product paths to 1e (even vector):
# 1o ⊗ 1o → 0e + 1e + 2e  ✓ (odd × odd = even)
# 0e ⊗ 1e → 1e            ✓ (even × even = even)
# 2e ⊗ 1o → 1e + 2e + 3e  ✗ (even × odd = odd, gives 1o not 1e)
```

**Key insight:** Your node features must contain `1o` (odd vectors) and your edge spherical harmonics contain `1o`. Their tensor product produces the required even-parity outputs.

Check your `hidden_irreps` includes components that can produce `1e`:
```python
# Example: hidden_irreps = "128x0e + 64x1o + 32x2e"
# 1o ⊗ 1o (from edge Y_lm) → 0e + 1e + 2e ✓
```

### Step 4: Build a Separate Hessian Graph

The Hessian requires interactions at longer range than forces. Create a separate graph:

```python
def add_hessian_graph(data, hessian_r_max):
    """Build graph with larger cutoff for Hessian prediction."""
    # Generate edges with larger cutoff
    edge_index_hessian, edge_vectors, edge_lengths = compute_neighbors(
        positions=data["positions"],
        cutoff=hessian_r_max,  # e.g., 8-16 Å vs 5-6 Å for forces
        pbc=data.get("pbc"),
        cell=data.get("cell"),
    )

    # Compute spherical harmonics for Hessian edges
    edge_attrs_hessian = spherical_harmonics(edge_vectors)

    # Compute radial embeddings for Hessian edges
    edge_feats_hessian = radial_embedding(edge_lengths)

    data["edge_index_hessian"] = edge_index_hessian
    data["edge_attrs_hessian"] = edge_attrs_hessian
    data["edge_feats_hessian"] = edge_feats_hessian

    return data
```

### Step 5: Create Hessian Output Heads

Add projection layers to map node/edge features to Hessian irreps:

```python
from e3nn import o3

class HessianHead(torch.nn.Module):
    def __init__(self, hidden_irreps, hessian_feature_dim=32):
        super().__init__()

        # Intermediate irreps with multiple channels
        hessian_out_irreps = o3.Irreps(
            f"{hessian_feature_dim}x0e + {hessian_feature_dim}x1e + {hessian_feature_dim}x2e"
        )

        # Project node features for diagonal blocks
        self.node_proj = o3.Linear(hidden_irreps, hessian_out_irreps)

        # Final projection to single channel per L
        self.node_out = o3.Linear(hessian_out_irreps, o3.Irreps("1x0e + 1x1e + 1x2e"))

        # Project edge features for off-diagonal blocks
        self.edge_proj = o3.Linear(hidden_irreps, hessian_out_irreps)
        self.edge_out = o3.Linear(hessian_out_irreps, o3.Irreps("1x0e + 1x1e + 1x2e"))
```

### Step 6: Extract Per-Edge Messages (Critical Step)

In standard message passing, edge messages are aggregated (summed) to nodes:

```python
# Standard MACE interaction
def forward(self, node_feats, edge_attrs, edge_feats, edge_index):
    # Compute per-edge messages
    tp_weights = self.radial_mlp(edge_feats)
    messages = self.tensor_product(node_feats[edge_index[0]], edge_attrs, tp_weights)
    # [n_edges, irreps]

    # Aggregate to nodes (LOSES per-edge information!)
    node_messages = scatter_sum(messages, edge_index[1], dim=0)
    # [n_nodes, irreps]

    return node_messages
```

**For HIP, you need the per-edge messages BEFORE aggregation:**

```python
def forward(self, node_feats, edge_attrs, edge_feats, edge_index, return_raw_messages=False):
    tp_weights = self.radial_mlp(edge_feats)
    messages = self.tensor_product(node_feats[edge_index[0]], edge_attrs, tp_weights)
    # [n_edges, irreps]

    # Return raw messages for Hessian prediction
    if return_raw_messages:
        return messages  # [n_edges, irreps] - per-edge features!

    # Standard path: aggregate to nodes
    node_messages = scatter_sum(messages, edge_index[1], dim=0)
    return node_messages
```

### Step 7: Convert Irreps to Cartesian 3×3 Matrices

Use Wigner 3j symbols to convert `0e + 1e + 2e` (9 components) to 3×3 matrices:

```python
from e3nn import o3
import einops

def irreps_to_cartesian_matrix(irreps):
    """
    Convert [N, 9] irreps (1x0e + 1x1e + 1x2e) to [N, 3, 3] Cartesian matrices.

    The 3×3 matrix is reconstructed by contracting irreps with Wigner 3j symbols
    that couple two L=1 (Cartesian) indices with each L component.
    """
    dtype, device = irreps.dtype, irreps.device

    # Wigner 3j symbols: (1, 1, L) for L = 0, 1, 2
    # These encode how two Cartesian indices combine into each L
    w0 = o3.wigner_3j(1, 1, 0, dtype=dtype, device=device)  # [3, 3, 1]
    w1 = o3.wigner_3j(1, 1, 1, dtype=dtype, device=device)  # [3, 3, 3]
    w2 = o3.wigner_3j(1, 1, 2, dtype=dtype, device=device)  # [3, 3, 5]

    # Contract: sum over m components, output [batch, 3, 3]
    matrix = (
        einops.einsum(w0, irreps[..., 0:1], "i j m, b m -> b i j") +
        einops.einsum(w1, irreps[..., 1:4], "i j m, b m -> b i j") +
        einops.einsum(w2, irreps[..., 4:9], "i j m, b m -> b i j")
    )
    return matrix  # [N, 3, 3]
```

### Step 8: Assemble the Full Hessian Matrix

Combine diagonal (node) and off-diagonal (edge) blocks into the full Hessian:

```python
def assemble_hessian(edge_index, node_blocks, edge_blocks, num_atoms):
    """
    Assemble full Hessian from 3×3 blocks.

    Args:
        edge_index: [2, E] - edge connectivity
        node_blocks: [N, 3, 3] - diagonal blocks
        edge_blocks: [E, 3, 3] - off-diagonal blocks
        num_atoms: N

    Returns:
        hessian: [3N, 3N] or flattened [9N²]
    """
    device = node_blocks.device
    dtype = node_blocks.dtype

    # Initialize Hessian
    hessian = torch.zeros(num_atoms * 3, num_atoms * 3, device=device, dtype=dtype)

    # Add diagonal blocks (with symmetrization)
    for i in range(num_atoms):
        block = node_blocks[i]
        hessian[i*3:(i+1)*3, i*3:(i+1)*3] = block + block.T

    # Add off-diagonal blocks
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e], edge_index[1, e]
        block = edge_blocks[e]
        # H[i,j] += block, H[j,i] += block.T (symmetry)
        hessian[i*3:(i+1)*3, j*3:(j+1)*3] += block
        hessian[j*3:(j+1)*3, i*3:(i+1)*3] += block.T

    return hessian
```

**For efficiency with batching, use `index_add_` with precomputed indices:**

```python
def assemble_hessian_fast(edge_index, node_blocks, edge_blocks, precomputed_indices):
    """Efficient batched assembly using index_add_."""
    total_entries = (num_atoms * 3) ** 2
    hessian_flat = torch.zeros(total_entries, device=device, dtype=dtype)

    # Off-diagonal: scatter edge blocks
    edge_flat = edge_blocks.reshape(-1)  # [E*9]
    edge_flat_T = edge_blocks.transpose(-2, -1).reshape(-1)  # [E*9] transposed

    hessian_flat.index_add_(0, precomputed_indices["msg_ij"], edge_flat)
    hessian_flat.index_add_(0, precomputed_indices["msg_ji"], edge_flat_T)

    # Diagonal: scatter node blocks
    node_flat = node_blocks.reshape(-1)  # [N*9]
    node_flat_T = node_blocks.transpose(-2, -1).reshape(-1)

    hessian_flat.index_add_(0, precomputed_indices["diag_ij"], node_flat)
    hessian_flat.index_add_(0, precomputed_indices["diag_ji"], node_flat_T)

    return hessian_flat
```

### Step 9: Integrate into Forward Pass

Modify your model's forward pass to optionally predict the Hessian:

```python
class MACEWithHIP(MACE):
    def __init__(self, ..., hip=False, hessian_r_max=8.0):
        super().__init__(...)
        self.hip = hip
        self.hessian_r_max = hessian_r_max

        if hip:
            self.hessian_head = HessianHead(hidden_irreps)
            self.hessian_interaction = InteractionBlock(...)  # Dedicated block

    def forward(self, data, predict_hessian=False, ...):
        # Standard forward pass
        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(node_feats, edge_attrs, edge_feats, edge_index)
            node_feats = product(node_feats, sc, node_attrs)
            node_feats_list.append(node_feats)

        # Energy and forces (standard)
        energy = self.readout(node_feats_list[-1])
        forces = -torch.autograd.grad(energy, positions)[0]

        # Hessian prediction (HIP)
        hessian = None
        if predict_hessian and self.hip:
            hessian = self._predict_hessian(data, node_feats_list)

        return {"energy": energy, "forces": forces, "hessian": hessian}

    def _predict_hessian(self, data, node_feats_list):
        # 1. Build Hessian graph
        data = add_hessian_graph(data, self.hessian_r_max)

        # 2. Compute spherical harmonics and radial features for Hessian edges
        edge_attrs_hess = self.spherical_harmonics(data["edge_vectors_hessian"])
        edge_feats_hess = self.radial_embedding(data["edge_lengths_hessian"])

        # 3. Get node features (e.g., from last layer)
        node_feats = node_feats_list[-1]

        # 4. Compute diagonal features (per-node)
        diag_irreps = self.hessian_head.node_proj(node_feats)
        diag_irreps = self.hessian_head.node_out(diag_irreps)  # [N, 9]

        # 5. Compute off-diagonal features (per-edge) via message passing
        raw_messages = self.hessian_interaction(
            node_feats=node_feats,
            edge_attrs=edge_attrs_hess,
            edge_feats=edge_feats_hess,
            edge_index=data["edge_index_hessian"],
            return_raw_messages=True,  # Get per-edge features!
        )
        offdiag_irreps = self.hessian_head.edge_proj(raw_messages)
        offdiag_irreps = self.hessian_head.edge_out(offdiag_irreps)  # [E, 9]

        # 6. Convert irreps to 3×3 matrices
        diag_blocks = irreps_to_cartesian_matrix(diag_irreps)      # [N, 3, 3]
        offdiag_blocks = irreps_to_cartesian_matrix(offdiag_irreps)  # [E, 3, 3]

        # 7. Assemble full Hessian
        hessian = assemble_hessian(
            data["edge_index_hessian"], diag_blocks, offdiag_blocks, num_atoms
        )

        return hessian
```

### Step 10: Add Hessian Loss Function

Train with a Hessian loss (typically MAE or MSE):

```python
def hessian_loss(pred_hessian, target_hessian, batch_ptr):
    """
    Compute Hessian loss with proper batching.

    Args:
        pred_hessian: Flattened predicted Hessians [(3N_1)² + (3N_2)² + ...]
        target_hessian: Flattened target Hessians
        batch_ptr: Pointer to separate samples [0, (3N_1)², (3N_1)² + (3N_2)², ...]
    """
    # MAE loss
    loss = torch.abs(pred_hessian - target_hessian).mean()

    # Or MSE loss
    # loss = ((pred_hessian - target_hessian) ** 2).mean()

    return loss

# In training loop
loss = (
    energy_weight * energy_loss +
    forces_weight * forces_loss +
    hessian_weight * hessian_loss  # Add Hessian term
)
```

## Summary Checklist

- [ ] **Irreps check**: Ensure `hidden_irreps` can produce `0e + 1e + 2e` via tensor products
- [ ] **Separate graph**: Build Hessian graph with larger cutoff
- [ ] **Extract raw messages**: Modify interaction blocks to return per-edge features before aggregation
- [ ] **Projection layers**: Add `o3.Linear` layers to project to `1x0e + 1x1e + 1x2e`
- [ ] **Irreps→Cartesian**: Implement conversion using Wigner 3j symbols
- [ ] **Hessian assembly**: Combine diagonal and off-diagonal blocks with symmetry
- [ ] **Loss function**: Add Hessian MAE/MSE to training objective
- [ ] **Batching**: Handle variable-size Hessians across batch (use flattened representation)

## Common Pitfalls

1. **Wrong parity**: Using `1o` instead of `1e` for Hessian - the Hessian is even-parity!

2. **Missing tensor product paths**: If your model only has `0e + 1o`, you can't produce `1e`. Add `2e` to hidden_irreps or use edge spherical harmonics with `1o`.

3. **Aggregating before extraction**: If you only have aggregated node messages, you lose the per-edge information needed for off-diagonal blocks.

4. **Cutoff too small**: Hessian elements can be non-zero at longer range than forces. Use `hessian_r_max > r_max`.

5. **Forgetting symmetry**: The Hessian is symmetric. Either predict only upper triangle, or symmetrize: `H_ij = (H_ij + H_ji^T) / 2`.

6. **Batch indexing**: Each sample has a different size Hessian `(3N_i)²`. Use pointers or pad carefully.

## References

- HIP-MACE Paper: [arXiv:2509.21624](https://arxiv.org/abs/2509.21624)
- MACE: [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)
- e3nn library: [e3nn.org](https://e3nn.org)
