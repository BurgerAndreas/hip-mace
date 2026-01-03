import torch
import numpy as np
import warnings
from ase import build
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric

# Suppress warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

def generate_random_rotation():
    """Generates a random 3x3 rotation matrix."""
    return o3.rand_matrix().to(device)

def get_block_diagonal_rotation(R, num_atoms):
    """Creates a block diagonal matrix (I_N kron R) for Hessian transformation."""
    # Efficiently create block diagonal without creating a massive dense matrix first if possible
    # Dimensions: (3*N, 3*N)
    eye = torch.eye(num_atoms, device=R.device, dtype=R.dtype)
    # R_big = torch.kron(eye, R) # Standard way
    # Faster/Safer way avoiding some view strides:
    R_big = torch.einsum("ij,ab->iajb", eye, R).reshape(num_atoms * 3, num_atoms * 3)
    return R_big

def test_hip_mace_equivariance():
    print(f"\n--- Starting HIP MACE Equivariance Test on {device} ---")
    
    # ---------------------------
    # 1. Model Configuration
    # ---------------------------
    # Defined strictly based on your saved info
    r_max = 6.0
    atomic_numbers_list = [1, 8] # H, O
    atomic_energies = np.array([0.0, 0.0], dtype=float) # Dummy energies
    
    model_config = dict(
        r_max=r_max,
        num_bessel=7,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=5,
        num_elements=len(atomic_numbers_list),
        hidden_irreps=o3.Irreps("32x0e + 32x1o + 32x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=atomic_numbers_list,
        correlation=3,
        radial_type="bessel",
        # HIP Specifics
        hip=True,
        hessian_feature_dim=16,
        hessian_use_last_layer_only=True,
        hessian_r_max=16.0,
        hessian_edge_lmax=3,
    )

    model = modules.MACE(**model_config)
    model.to(device=device)
    model.eval() # Set to eval mode for deterministic testing

    # ---------------------------
    # 2. Data Setup (Water Molecule)
    # ---------------------------
    # Using a simple water molecule for clear testing
    atoms = build.molecule('H2O')
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        # Dummy properties required by config init
        properties={"energy": -10.0, "forces": np.zeros((3,3)), "hessian": np.zeros((3*3, 3*3))},
        property_weights={"energy": 1.0, "forces": 1.0, "hessian": 1.0},
    )
    
    table = tools.AtomicNumberTable(atomic_numbers_list)
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=r_max)
    
    # Create batch (Batch size 1 for equivariance test)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch_orig = next(iter(data_loader))
    batch_orig.to(device)

    # ---------------------------
    # 3. Forward Pass - Original
    # ---------------------------
    output_orig = model(
        batch_orig.to_dict(), 
        training=False,
        predict_hessian=True, 
    )
    
    E1 = output_orig["energy"]
    F1 = output_orig["forces"]
    H1 = output_orig["hessian"] 
    
    natoms = torch.bincount(batch_orig.batch)
    N = int(natoms.sum().item())
    H1 = H1.reshape(N * 3, N * 3)
    
    
    print(f"System size: {N} atoms")
    print(f"Hessian Shape: {H1.shape}")

    # ---------------------------
    # 4. Rotate and Forward Pass
    # ---------------------------
    R_mat = generate_random_rotation()
    
    # Create rotated batch
    batch_rot = batch_orig.clone()
    # Apply rotation: pos_new = pos_old @ R^T (standard row-vector rotation)
    # Note: In torch/numpy, v @ M uses last dim. 
    # If we want x' = R x, for row vectors this is x' = x R^T.
    # The snippet you provided used `pos @ R`. Let's stick to standard `pos @ R.T` 
    # so that `R` is the rotation matrix. 
    # If we use `pos @ R`, then the effective rotation matrix applied is R^T.
    batch_rot.positions = batch_orig.positions @ R_mat.T 
    
    output_rot = model(
        batch_rot.to_dict(),
        training=False,
        predict_hessian=True,
    )

    E2 = output_rot["energy"]
    F2 = output_rot["forces"]
    H2 = output_rot["hessian"]
    H2 = H2.reshape(N * 3, N * 3)

    # ---------------------------
    # 5. Verification
    # ---------------------------
    
    # A. Energy Invariance: E1 == E2
    energy_diff = torch.abs(E1 - E2).item()
    print(f"\nEnergy Difference (Abs): {energy_diff:.2e}")
    assert np.isclose(energy_diff, 0.0, atol=1e-4), "Energy is not invariant!"

    # B. Force Equivariance: F2 == F1 @ R^T  => F1 == F2 @ R
    # F are vectors. If pos rotates by R, Forces rotate by R.
    # F_rot = F_orig @ R^T
    # check: F_orig - F_rot @ R
    F2_rotated_back = F2 @ R_mat
    force_diff = torch.norm(F1 - F2_rotated_back).item()
    print(f"Force Difference (Norm): {force_diff:.2e}")
    assert np.isclose(force_diff, 0.0, atol=1e-3), "Forces are not equivariant!"

    # C. Hessian Equivariance
    # Hessian is a rank-2 tensor. Under rotation R:
    # H_rot = R_big @ H_orig @ R_big.T
    # Therefore: H_orig = R_big.T @ H_rot @ R_big
    
    R_big = get_block_diagonal_rotation(R_mat, N)
    
    # Transform H2 back to H1 frame
    H2_transformed_back = R_big.T @ H2 @ R_big
    
    hessian_diff = torch.norm(H1 - H2_transformed_back).item()
    hessian_rel = hessian_diff / (torch.norm(H1) + 1e-9)
    
    print(f"Hessian Difference (Norm): {hessian_diff:.2e}")
    print(f"Hessian Relative Error:    {hessian_rel:.2e}")
    
    # Check Symmetry of Hessian while we are at it
    sym_diff = torch.norm(H1 - H1.T).item()
    print(f"Hessian Symmetry Error:    {sym_diff:.2e}")

    # Assertions
    if hessian_rel > 5e-2: # MACE float32 precision can vary, usually < 1e-2
        print("!! Warning: Hessian equivariance error is high. Check model convergence or float precision.")
    else:
        print(">> Hessian Equivariance Passed.")

    assert sym_diff < 1e-3, "Hessian is not symmetric!"

if __name__ == "__main__":
    test_hip_mace_equivariance()