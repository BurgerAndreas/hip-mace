# HIP Hessian Prediction: Message Passing Implementation

The HIP (Hessian Interatomic Potential) message passing implementation predicts the Hessian matrix (second derivative of energy w.r.t. atomic positions) using an equivariant architecture.

## Overview

The Hessian is a symmetric `(3N × 3N)` matrix where N is the number of atoms. HIP decomposes it into:
1. **Diagonal blocks** (3×3 per atom) - from node features
2. **Off-diagonal blocks** (3×3 per atom pair) - from edge features via message passing

## Key Components

### 1. Separate Hessian Graph

**File:** `mace/modules/hip.py:374-576`

A separate graph is built for Hessian prediction with a potentially larger cutoff (`hessian_r_max`), since second derivatives require longer-range interactions:

```python
data = add_hessian_graph_batch(data, cutoff=self.hessian_r_max)
```

This creates:
- `edge_index_hessian`: Edge connectivity for Hessian
- `edge_distance_hessian`: Edge lengths
- `edge_distance_vec_hessian`: Edge vectors
- Precomputed indices for efficient Hessian assembly (`message_idx_ij`, `message_idx_ji`, `diag_ij`, `diag_ji`)

### 2. Dedicated Interaction Block

**File:** `mace/modules/models.py:455-490`

A separate MACE-style interaction block is created specifically for extracting edge features:

```python
self.hessian_interaction = interaction_cls(
    node_attrs_irreps=node_attr_irreps,
    node_feats_irreps=hidden_irreps,
    edge_attrs_irreps=sh_irreps_hessian,  # Spherical harmonics for Hessian graph
    edge_feats_irreps=hessian_edge_feats_irreps,  # Radial embeddings
    target_irreps=hidden_irreps,
    hidden_irreps=hidden_irreps,
    avg_num_neighbors=avg_num_neighbors,
    radial_MLP=hessian_radial_MLP_for_interaction,
    ...
)
```

### 3. Extract Raw Messages Before Aggregation

**File:** `mace/modules/blocks.py:626-627, 733-734, etc.`

The key insight is extracting **per-edge messages** before they're aggregated to nodes. In standard MACE, messages are computed and then scattered/summed to target nodes. For HIP, we intercept the messages before aggregation:

```python
mji = self.conv_tp(
    node_feats[edge_index[0]], edge_attrs, tp_weights
)  # [n_edges, irreps]
if return_raw_messages:
    return mji  # Return BEFORE scatter_sum aggregation
```

This is invoked via `_extract_raw_messages_from_interaction()` at `models.py:675-701`:

```python
def _extract_raw_messages_from_interaction(
    self,
    interaction: InteractionBlock,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
    cutoff: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract per-edge messages from an interaction block before aggregation."""
    result = interaction(
        node_attrs=node_attrs,
        node_feats=node_feats,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=edge_index,
        cutoff=cutoff,
        return_raw_messages=True,
    )
    if isinstance(result, tuple):
        return result[0]
    return result
```

### 4. Hessian Feature Generation

**File:** `mace/modules/models.py:785-809`

For message passing mode, the off-diagonal (edge) features are computed as:

```python
# Get per-edge messages from the Hessian interaction block
raw_messages = self._extract_raw_messages_from_interaction(
    interaction=self.hessian_interaction,
    node_feats=node_feats,
    edge_attrs=edge_attrs_hessian,
    edge_feats=edge_feats_hessian,
    edge_index=edge_index_hessian,
)
# [n_edges, irreps_mid]

# Apply the interaction's linear layer to transform irreps_mid -> hidden_irreps
raw_messages = self.hessian_interaction.linear(raw_messages)

# Project to hessian output irreps (0e + 1e + 2e)
off_diag_feats = self.hessian_message_proj(raw_messages)
```

Diagonal (node) features are computed by projecting node features directly:

```python
diag_feats = self.hessian_proj_nodes_layerwise(node_feats)
```

### 5. Irreps to Cartesian Conversion

**File:** `mace/modules/hip.py:30-44`

The output irreps `1x0e + 1x1e + 1x2e` (9 components total: 1 + 3 + 5) are converted to 3×3 Cartesian matrices using Wigner 3j symbols:

```python
def irreps_to_cartesian_matrix(irreps: torch.Tensor) -> torch.Tensor:
    """
    irreps: torch.Tensor [N, 9] or [E, 9]
    Returns: torch.Tensor [N, 3, 3] or [E, 3, 3]
    """
    w0 = o3.wigner_3j(1, 1, 0)  # L=0 component (1 element)
    w1 = o3.wigner_3j(1, 1, 1)  # L=1 component (3 elements)
    w2 = o3.wigner_3j(1, 1, 2)  # L=2 component (5 elements)
    return (
        einsum(w0, irreps[..., :1], "m1 m2 m3, b m3 -> b m1 m2")
        + einsum(w1, irreps[..., 1:4], "m1 m2 m3, b m3 -> b m1 m2")
        + einsum(w2, irreps[..., 4:9], "m1 m2 m3, b m3 -> b m1 m2")
    )
```

### 6. Hessian Assembly

**File:** `mace/modules/hip.py:316-342`

The 3×3 blocks are assembled into the full flattened Hessian using efficient `index_add_` operations:

```python
def blocks3x3_to_hessian(edge_index, data, l012_edge_features, l012_node_features):
    # Off-diagonal blocks: scatter edge blocks to H[i,j] and H[j,i] (transposed)
    hessian = _indexadd_offdiagonal_to_flat_hessian(edge_index, l012_edge_features, data)
    # Diagonal blocks: add node blocks to H[i,i]
    hessian = _indexadd_diagonal_to_flat_hessian(hessian, l012_node_features, data)
    return hessian
```

The off-diagonal assembly (`hip.py:183-212`):
```python
def _indexadd_offdiagonal_to_flat_hessian(edge_index, messages, data):
    hessian1d = torch.zeros(total_entries, device=messages.device, dtype=messages.dtype)
    messageflat = messages.reshape(-1)
    # Transpose for j->i contribution
    messages_3x3_T = messages.view(E, 3, 3).transpose(-2, -1).reshape(-1)
    # Add both contributions
    hessian1d.index_add_(0, data["message_idx_ij"], messageflat)        # i->j direct
    hessian1d.index_add_(0, data["message_idx_ji"], messages_3x3_T)     # j->i transposed
    return hessian1d
```

## Why Even Parity (0e, 1e, 2e)?

The Hessian is a rank-2 tensor that transforms with **even parity** under inversion (unlike forces which have odd parity `1o`).

The tensor product of two odd-parity vectors:
- Node features containing `1o` (odd vector)
- Edge spherical harmonics `1o` (odd vector)

Produces: `1o ⊗ 1o → 0e + 1e + 2e` (all even parity)

This ensures the predicted Hessian transforms correctly under E(3) transformations.

## Layer Aggregation

If using multiple interaction layers (not just the last), features are aggregated:

```python
if self.hessian_aggregation == "learnable" and len(diag_feats_list) > 1:
    # Learnable softmax weights over layers
    weights = torch.softmax(self.hessian_layer_weights_param, dim=0)
    diag_out = (diag_stacked * weights).sum(dim=0)
    off_diag_out = (off_diag_stacked * weights).sum(dim=0)
else:
    # Simple mean aggregation
    diag_out = diag_stacked.mean(dim=0)
    off_diag_out = off_diag_stacked.mean(dim=0)
```

## Data Flow Summary

```
Node Features (from main MACE interaction layers)
          ↓
┌─────────┴─────────┐
↓                   ↓
Diagonal Features   Hessian Interaction Block (on Hessian graph)
(per-node)          ↓
     ↓              Raw Messages (per-edge, BEFORE scatter_sum)
     ↓              ↓
hessian_proj_nodes  hessian_interaction.linear + hessian_message_proj
     ↓              ↓
[N, 9] irreps       [E, 9] irreps
     ↓              ↓
     └──── irreps_to_cartesian_matrix ────┘
           ↓              ↓
       [N, 3, 3]      [E, 3, 3]
           └──────┬───────┘
                  ↓
          blocks3x3_to_hessian (index_add assembly)
                  ↓
          Flattened Hessian [(3N)²]
```

## Configuration Parameters

Key HIP parameters in model config:

| Parameter | Description |
|-----------|-------------|
| `hip` | Enable Hessian prediction |
| `hessian_feature_dim` | Channels per L-component (0, 1, 2) |
| `hessian_r_max` | Cutoff radius for Hessian graph (typically larger than main `r_max`) |
| `hessian_edge_lmax` | Maximum spherical harmonic degree for edge features |
| `hessian_use_last_layer_only` | Use only final interaction layer vs. aggregating all |
| `hessian_edge_feature_method` | `"message_passing"` or `"edge_tp"` |
| `hessian_aggregation` | `"mean"` or `"learnable"` for multi-layer aggregation |
| `hessian_use_radial` | Include radial embeddings in edge features |
| `hessian_use_both_nodes` | Use both h_i and h_j (vs. only h_j) |
| `hessian_separate_radial_network` | Use dedicated radial MLP for Hessian |
| `hessian_use_edge_gates` | Add equivariant gating on off-diagonal features |
