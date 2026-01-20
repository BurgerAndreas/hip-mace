#!/usr/bin/env python3
"""
Quick verification script for AMP implementation
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from mace.tools.checkpoint import CheckpointBuilder, CheckpointState


def test_basic_amp():
    """Test basic AMP functionality"""
    print("Testing basic AMP functionality...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    x = torch.randn(5, 10, device=device)
    target = torch.randn(5, 10, device=device)

    # Test bfloat16 autocast
    with autocast(enabled=True, dtype=torch.bfloat16):
        y = model(x)
        loss = ((y - target) ** 2).mean()

    print(f"✓ Autocast (bfloat16) works - output dtype: {y.dtype}")

    # Test float16 with scaler
    if torch.cuda.is_available():
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        with autocast(enabled=True, dtype=torch.float16):
            y = model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"✓ GradScaler (float16) works - scale: {scaler.get_scale()}")
    else:
        print("⊘ Skipping GradScaler test (CUDA not available)")


def test_checkpoint_scaler():
    """Test checkpoint with scaler"""
    print("\nTesting checkpoint with scaler...")

    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Test without scaler
    state1 = CheckpointState(model, optimizer, scheduler, None)
    checkpoint1 = CheckpointBuilder.create_checkpoint(state1)
    assert "scaler" not in checkpoint1
    print("✓ Checkpoint without scaler works")

    # Test with scaler
    if torch.cuda.is_available():
        scaler = GradScaler()
        scaler._scale = torch.tensor(2048.0)

        state2 = CheckpointState(model, optimizer, scheduler, scaler)
        checkpoint2 = CheckpointBuilder.create_checkpoint(state2)
        assert "scaler" in checkpoint2
        print("✓ Checkpoint with scaler works")

        # Test loading
        new_scaler = GradScaler()
        state3 = CheckpointState(model, optimizer, scheduler, new_scaler)
        CheckpointBuilder.load_checkpoint(state3, checkpoint2, strict=False)
        assert new_scaler.get_scale() == scaler.get_scale()
        print("✓ Checkpoint scaler load/save works")
    else:
        print("⊘ Skipping scaler checkpoint test (CUDA not available)")


def test_dtype_dict():
    """Test dtype dictionary has all required dtypes"""
    print("\nTesting dtype dictionary...")

    from mace.tools.torch_tools import dtype_dict

    required_dtypes = ["float32", "float64", "bfloat16", "float16"]
    for dtype_name in required_dtypes:
        assert dtype_name in dtype_dict, f"Missing {dtype_name} in dtype_dict"

    print(f"✓ dtype_dict has all required dtypes: {list(dtype_dict.keys())}")


def test_bfloat16_check():
    """Test bfloat16 support checking"""
    print("\nTesting bfloat16 support check...")

    from mace.tools.torch_tools import check_bfloat16_support

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supports_bf16 = check_bfloat16_support(device)

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        expected = capability[0] >= 8
        assert supports_bf16 == expected
        print(f"✓ bfloat16 support check works - GPU {capability}, supports_bf16={supports_bf16}")
    else:
        print(f"✓ bfloat16 support check works - CPU, supports_bf16={supports_bf16}")


if __name__ == "__main__":
    print("=" * 60)
    print("AMP Implementation Verification")
    print("=" * 60)

    try:
        test_basic_amp()
        test_checkpoint_scaler()
        test_dtype_dict()
        test_bfloat16_check()

        print("\n" + "=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
