"""
Tests for Automatic Mixed Precision (AMP) training
"""

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from mace.tools.checkpoint import CheckpointBuilder, CheckpointState


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model():
    """Create a simple model for testing"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer"""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def simple_scheduler(simple_optimizer):
    """Create a simple learning rate scheduler"""
    return torch.optim.lr_scheduler.ExponentialLR(simple_optimizer, gamma=0.95)


class TestAutoCast:
    """Test autocast functionality"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_bfloat16(self, simple_model, device):
        """Test that autocast produces bfloat16 tensors for matmul operations"""
        simple_model.to(device)
        x = torch.randn(5, 10, device=device)

        with autocast(enabled=True, dtype=torch.bfloat16):
            # Linear layers use matmul which should be in bfloat16
            y = simple_model(x)

        # Output should be in bfloat16 (or float32 for some ops)
        assert y.dtype in [torch.bfloat16, torch.float32]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_float16(self, simple_model, device):
        """Test that autocast produces float16 tensors for matmul operations"""
        simple_model.to(device)
        x = torch.randn(5, 10, device=device)

        with autocast(enabled=True, dtype=torch.float16):
            y = simple_model(x)

        # Output should be in float16 (or float32 for some ops)
        assert y.dtype in [torch.float16, torch.float32]

    def test_autocast_disabled(self, simple_model, device):
        """Test that autocast disabled keeps float32"""
        simple_model.to(device)
        simple_model = simple_model.float()  # Ensure float32
        x = torch.randn(5, 10, device=device, dtype=torch.float32)

        with autocast(enabled=False, dtype=torch.float32):
            y = simple_model(x)

        assert y.dtype == torch.float32


class TestGradScaler:
    """Test gradient scaler for fp16 training"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scaler_scale_backward(self, simple_model, simple_optimizer, device):
        """Test that GradScaler scales gradients correctly"""
        simple_model.to(device)
        scaler = GradScaler()
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        simple_optimizer.zero_grad()
        with autocast(enabled=True, dtype=torch.float16):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        # Scale loss and backward
        scaler.scale(loss).backward()

        # Check that gradients exist and are not NaN
        for param in simple_model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scaler_step_and_update(self, simple_model, simple_optimizer, device):
        """Test that GradScaler step and update work correctly"""
        simple_model.to(device)
        scaler = GradScaler()
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        simple_optimizer.zero_grad()
        with autocast(enabled=True, dtype=torch.float16):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(simple_optimizer)
        scaler.update()

        # Check that scaler scale is reasonable
        assert scaler.get_scale() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scaler_unscale_before_clip(self, simple_model, simple_optimizer, device):
        """Test that gradients can be unscaled before clipping"""
        simple_model.to(device)
        scaler = GradScaler()
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        simple_optimizer.zero_grad()
        with autocast(enabled=True, dtype=torch.float16):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(simple_optimizer)

        # Clip gradients
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(
            simple_model.parameters(), max_norm=max_norm
        )

        # Check that total norm is computed correctly
        assert total_norm >= 0
        assert not torch.isnan(total_norm)


class TestCheckpointScaler:
    """Test checkpoint saving/loading with GradScaler"""

    def test_checkpoint_save_with_scaler(
        self, simple_model, simple_optimizer, simple_scheduler
    ):
        """Test that checkpoint saves scaler state"""
        scaler = GradScaler()
        state = CheckpointState(simple_model, simple_optimizer, simple_scheduler, scaler)
        checkpoint = CheckpointBuilder.create_checkpoint(state)

        assert "scaler" in checkpoint
        assert "scale" in checkpoint["scaler"]

    def test_checkpoint_save_without_scaler(
        self, simple_model, simple_optimizer, simple_scheduler
    ):
        """Test that checkpoint without scaler doesn't have scaler key"""
        state = CheckpointState(simple_model, simple_optimizer, simple_scheduler, None)
        checkpoint = CheckpointBuilder.create_checkpoint(state)

        assert "scaler" not in checkpoint

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_load_with_scaler(
        self, simple_model, simple_optimizer, simple_scheduler, device
    ):
        """Test that checkpoint loads scaler state correctly"""
        simple_model.to(device)

        # Create and save checkpoint with scaler
        scaler1 = GradScaler()
        scaler1._scale = torch.tensor(2048.0)  # Set a specific scale
        state1 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, scaler1)
        checkpoint = CheckpointBuilder.create_checkpoint(state1)

        # Load checkpoint into new scaler
        scaler2 = GradScaler()
        state2 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, scaler2)
        CheckpointBuilder.load_checkpoint(state2, checkpoint, strict=False)

        # Check that scaler state was loaded
        assert scaler2.get_scale() == scaler1.get_scale()

    def test_checkpoint_load_mismatch_scaler_in_checkpoint_but_not_state(
        self, simple_model, simple_optimizer, simple_scheduler
    ):
        """Test graceful handling when checkpoint has scaler but state doesn't"""
        # Create checkpoint with scaler
        scaler = GradScaler()
        state1 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, scaler)
        checkpoint = CheckpointBuilder.create_checkpoint(state1)

        # Load into state without scaler (should warn but not fail)
        state2 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, None)
        CheckpointBuilder.load_checkpoint(state2, checkpoint, strict=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_load_mismatch_scaler_in_state_but_not_checkpoint(
        self, simple_model, simple_optimizer, simple_scheduler
    ):
        """Test graceful handling when state has scaler but checkpoint doesn't"""
        # Create checkpoint without scaler
        state1 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, None)
        checkpoint = CheckpointBuilder.create_checkpoint(state1)

        # Load into state with scaler (should log info but not fail)
        scaler = GradScaler()
        state2 = CheckpointState(simple_model, simple_optimizer, simple_scheduler, scaler)
        CheckpointBuilder.load_checkpoint(state2, checkpoint, strict=False)


class TestBFloat16VsFloat16:
    """Test differences between bfloat16 and float16"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bfloat16_no_scaling_needed(self, simple_model, device):
        """Test that bfloat16 doesn't need gradient scaling"""
        simple_model.to(device)
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        # Train step with bfloat16 (no scaler)
        with autocast(enabled=True, dtype=torch.bfloat16):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        loss.backward()

        # Check that gradients are finite without scaling
        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_benefits_from_scaling(self, simple_model, simple_optimizer, device):
        """Test that float16 training benefits from gradient scaling"""
        simple_model.to(device)
        scaler = GradScaler()
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        # Train step with float16 and scaler
        simple_optimizer.zero_grad()
        with autocast(enabled=True, dtype=torch.float16):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(simple_optimizer)
        scaler.update()

        # Check that training proceeded without NaN/Inf
        for param in simple_model.parameters():
            assert torch.isfinite(param).all()


class TestAMPIntegration:
    """Integration tests for AMP in training pipeline"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_step_with_amp(self, simple_model, simple_optimizer, device):
        """Test a complete training step with AMP enabled"""
        simple_model.to(device)
        scaler = GradScaler()
        amp_enabled = True
        amp_dtype = torch.float16

        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 10, device=device)

        # Simulate take_step function
        simple_optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            y = simple_model(x)
            loss = ((y - target) ** 2).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping with scaler
        max_grad_norm = 1.0
        if scaler is not None:
            scaler.unscale_(simple_optimizer)
        torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=max_grad_norm)

        if scaler is not None:
            scaler.step(simple_optimizer)
            scaler.update()
        else:
            simple_optimizer.step()

        # Check that training proceeded successfully
        assert not torch.isnan(loss)
        for param in simple_model.parameters():
            assert torch.isfinite(param).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_evaluation_with_amp(self, simple_model, device):
        """Test evaluation with AMP enabled"""
        simple_model.to(device)
        simple_model.eval()
        amp_enabled = True
        amp_dtype = torch.bfloat16

        x = torch.randn(5, 10, device=device)

        with torch.no_grad():
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                y = simple_model(x)

        # Check output
        assert y.shape == (5, 10)
        assert torch.isfinite(y).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
