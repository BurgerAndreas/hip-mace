import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric


torch.set_default_dtype(torch.float32)

ATOMIC_NUMBERS = [1, 8]
POSITIONS = np.array(
    [
        [0.0, -2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
SPECIES = np.array([8, 1, 1])


@pytest.fixture(scope="module", name="hip_model")
def hip_model_fixture():
    torch.manual_seed(1)
    model = modules.MACE(
        r_max=6.0,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(ATOMIC_NUMBERS),
        hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(ATOMIC_NUMBERS)),
        avg_num_neighbors=4,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=2,
        radial_type="bessel",
        hip=True,
        hessian_feature_dim=4,
        hessian_use_last_layer_only=True,
        hessian_r_max=16.0,
        hessian_edge_lmax=2,
    )
    model.eval()
    return model


def _predict_hessian(model, positions):
    config = data.Configuration(
        atomic_numbers=SPECIES,
        positions=positions,
        properties={
            "energy": 0.0,
            "forces": np.zeros_like(positions),
        },
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=tools.AtomicNumberTable(ATOMIC_NUMBERS),
        cutoff=model.r_max.item(),
    )
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                [atomic_data],
                batch_size=1,
                shuffle=False,
            )
        )
    )
    output = model(
        batch.to_dict(),
        training=False,
        compute_force=False,
        predict_hessian=True,
    )
    num_atoms = len(positions)
    return output["hessian"].reshape(num_atoms * 3, num_atoms * 3).detach()


def _block_diagonal_rotation(rotation, num_atoms):
    eye = torch.eye(num_atoms, dtype=rotation.dtype, device=rotation.device)
    return torch.einsum("ij,ab->iajb", eye, rotation).reshape(
        num_atoms * 3, num_atoms * 3
    )


def test_hip_hessian_is_symmetric(hip_model):
    hessian = _predict_hessian(hip_model, POSITIONS)

    assert torch.allclose(hessian, hessian.T, atol=1e-6, rtol=1e-6)


def test_hip_hessian_row_sums_are_zero(hip_model):
    hessian = _predict_hessian(hip_model, POSITIONS)

    assert torch.allclose(
        hessian.sum(dim=1),
        torch.zeros(hessian.shape[0], dtype=hessian.dtype),
        atol=1e-6,
        rtol=0.0,
    )


def test_hip_hessian_is_translation_invariant(hip_model):
    hessian = _predict_hessian(hip_model, POSITIONS)
    translated_hessian = _predict_hessian(
        hip_model,
        POSITIONS + np.array([2.5, -1.25, 0.75]),
    )

    assert torch.allclose(hessian, translated_hessian, atol=1e-6, rtol=1e-6)


def test_hip_hessian_is_rotation_equivariant(hip_model):
    hessian = _predict_hessian(hip_model, POSITIONS)
    angle = 0.37
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=hessian.dtype,
    )

    rotated_positions = POSITIONS @ rotation.numpy().T
    rotated_hessian = _predict_hessian(hip_model, rotated_positions)
    big_rotation = _block_diagonal_rotation(rotation, len(POSITIONS))
    expected_rotated_hessian = big_rotation @ hessian @ big_rotation.T

    assert torch.allclose(
        rotated_hessian,
        expected_rotated_hessian,
        atol=1e-5,
        rtol=1e-5,
    )


def test_hip_hessian_has_near_zero_acoustic_modes(hip_model):
    hessian = _predict_hessian(hip_model, POSITIONS)
    translations = torch.eye(3, dtype=hessian.dtype).repeat(len(POSITIONS), 1)
    eigenvalues = torch.linalg.eigvalsh((hessian + hessian.T) / 2)

    assert torch.allclose(
        hessian @ translations,
        torch.zeros_like(translations),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.all(eigenvalues.abs().sort().values[:3] < 1e-5)
