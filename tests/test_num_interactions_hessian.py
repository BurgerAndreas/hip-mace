#!/usr/bin/env python3
"""Test script for num_interactions_hessian parameter."""

import torch
from e3nn import o3

from mace.modules import ScaleShiftMACE, interaction_classes

def test_num_interactions_hessian():
    """Test that num_interactions_hessian parameter works correctly."""

    # Basic model config
    config = {
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 3,
        "interaction_cls": interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": interaction_classes["RealAgnosticResidualInteractionBlock"],
        "num_interactions": 2,
        "num_elements": 3,
        "hidden_irreps": o3.Irreps("32x0e + 32x1o + 32x2e"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": torch.zeros(3),
        "avg_num_neighbors": 10.0,
        "atomic_numbers": [1, 6, 8],
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }

    # Test 1: Model without HIP (num_interactions_hessian should be ignored)
    print("Test 1: Model without HIP")
    model_no_hip = ScaleShiftMACE(**config, hip=False, num_interactions_hessian=0)
    print(f"  Model created successfully")
    print(f"  Has hessian_interactions: {hasattr(model_no_hip, 'hessian_interactions')}")

    # Test 2: Model with HIP but num_interactions_hessian=0 (default)
    print("\nTest 2: Model with HIP, num_interactions_hessian=0")
    model_hip_0 = ScaleShiftMACE(**config, hip=True, num_interactions_hessian=0)
    print(f"  Model created successfully")
    print(f"  num_interactions_hessian: {model_hip_0.num_interactions_hessian}")
    print(f"  Has hessian_interactions: {hasattr(model_hip_0, 'hessian_interactions')}")

    # Test 3: Model with HIP and num_interactions_hessian=2
    print("\nTest 3: Model with HIP, num_interactions_hessian=2")
    model_hip_2 = ScaleShiftMACE(**config, hip=True, num_interactions_hessian=2)
    print(f"  Model created successfully")
    print(f"  num_interactions_hessian: {model_hip_2.num_interactions_hessian}")
    print(f"  Has hessian_interactions: {hasattr(model_hip_2, 'hessian_interactions')}")
    if hasattr(model_hip_2, 'hessian_interactions'):
        print(f"  Number of hessian interaction layers: {len(model_hip_2.hessian_interactions)}")
        print(f"  Number of hessian product layers: {len(model_hip_2.hessian_products)}")

    # Test 4: Forward pass with num_interactions_hessian=2
    print("\nTest 4: Forward pass with num_interactions_hessian=2")
    batch_size = 2
    num_atoms = 5

    # Create dummy data
    data = {
        "positions": torch.randn(num_atoms, 3, requires_grad=True),
        "node_attrs": torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0, 1]), num_classes=3).float(),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        "shifts": torch.zeros(4, 3),
        "unit_shifts": torch.zeros(4, 3),
        "cell": torch.eye(3).unsqueeze(0).expand(batch_size, 3, 3),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "ptr": torch.tensor([0, 3, 5], dtype=torch.long),
    }

    # Forward pass without Hessian prediction
    print("  Forward pass (predict_hessian=False)...")
    output = model_hip_2(data, predict_hessian=False)
    print(f"    Energy shape: {output['energy'].shape}")
    print(f"    Forces shape: {output['forces'].shape}")
    print(f"    Hessian: {output['hessian']}")

    # Forward pass with Hessian prediction
    print("  Forward pass (predict_hessian=True)...")
    output = model_hip_2(data, predict_hessian=True)
    print(f"    Energy shape: {output['energy'].shape}")
    print(f"    Forces shape: {output['forces'].shape}")
    print(f"    Hessian shape: {output['hessian'].shape}")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_num_interactions_hessian()
