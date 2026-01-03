import os

from mace.calculators import mace_mp
from ase import build

import numpy as np
import torch
import torch.nn.functional
from ase import build
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R
import warnings

from mace import data, modules, tools
from mace.tools import torch_geometric

# Suppress TorchScript type annotation warnings
warnings.filterwarnings("ignore", 
                       message=".*TorchScript type system doesn't support instance-level annotations.*",
                       category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


def test_mace_mp_calc():
    print("\nTesting MACE MP calculator")
    # Load the MACE model
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)
    print()

    atoms = build.molecule('H2O')
    atoms.calc = calc
    print("potential energy", atoms.get_potential_energy())

    mace_mp_model = calc.models[0]
    
    atomic_energies = mace_mp_model.atomic_energies_fn.atomic_energies.cpu().numpy()
    atomic_numbers = mace_mp_model.atomic_numbers.tolist()
    
    print("\nModel:")
    # LinearReadoutBlock
    # NonLinearReadoutBlock
    # RealAgnosticResidualInteractionBlock
    # RealAgnosticResidualInteractionBlock
    # EquivariantProductBasisBlock
    # EquivariantProductBasisBlock
    # output1 = mace_mp_model(batch.to_dict(), training=True)

    for ro in mace_mp_model.readouts:
        print(ro.__class__.__name__)
    for interaction in mace_mp_model.interactions:
        print(interaction.__class__.__name__)
    for product in mace_mp_model.products:
        print(product.__class__.__name__)
    return

def test_hip_mace():
    print("\nTesting HIP MACE")
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        properties={
            "forces": np.array( # Nx3
                [
                    [0.0, -1.3, 0.0],
                    [1.0, 0.2, 0.0],
                    [0.0, 1.1, 0.3],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "energy": -1.5,
            "charges": np.array([-2.0, 1.0, 1.0, 1.0]), # N
            "dipole": np.array([-1.5, 1.5, 2.0]), # 3
            "polarizability": np.array( # 3x3
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "charges": 1.0,
            "dipole": 1.0,
            "polarizability": 1.0,
        },
    )

    # atomic_energies is an array of reference energies for each atomic species in the model,
    # typically the energy of an isolated atom for each atomic number. These are used to allow
    # the model to output formation or cohesive energies by subtracting these atomic energies
    # from the sum of total predicted energies in the system (or vice versa, depending on convention).
    # They are also required to set the correct baseline energy for each atomic type.
    atomic_numbers = torch.arange(start=1, end=config.atomic_numbers.max() + 1).tolist()
    atomic_energies = torch.zeros(len(atomic_numbers)).tolist()
    table = tools.AtomicNumberTable(atomic_numbers)

    # table = tools.AtomicNumberTable([1, 8])
    # atomic_numbers = table.zs
    # atomic_energies = np.array([1.0, 3.0], dtype=float)
    # Create MACE model
    r_max = 6.0
    model_config = dict(
        r_max=r_max,
        num_bessel=7,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=5,
        num_elements=len(atomic_numbers),
        # 32 * (2l+1), l=1 -> 32 * 3 = 96
        # 32 + 96 = 128
        hidden_irreps=o3.Irreps("32x0e + 32x1o + 32x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=atomic_numbers,
        correlation=3,
        radial_type="bessel",
        # Added for HIP Hessian prediction
        hip=True,
        hessian_feature_dim=16,
        hessian_use_last_layer_only=True,
        hessian_r_max=16.0,
        hessian_edge_lmax=3, # 2 or 3
    )
    model = modules.MACE(**model_config)
    model.to(device=device)
    # model_compiled = jit.compile(model)

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=r_max)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch.to(device=device)
    output1 = model(
        batch.to_dict(), 
        training=False,
        # compute_hessian=True, # autograd
        predict_hessian=True, # HIP
        # return_l_features=True, 
    )
    for k in ["energy", "forces", "hessian"]:
        print(k, output1[k].shape)

    # for k, v in batch.to_dict().items():
    #     print("DEBUG", k, v.shape)



if __name__ == "__main__":
    test_hip_mace()