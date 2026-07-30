#!/usr/bin/env python3
"""Test num_interactions_hessian for the pair-v2 HIP head."""

import torch
from e3nn import o3

from mace.modules import ScaleShiftMACE, interaction_classes


def test_num_interactions_hessian():
    config = {
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 3,
        "interaction_cls": interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
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

    model_no_hip = ScaleShiftMACE(**config, hip=False, num_interactions_hessian=0)
    assert not hasattr(model_no_hip, "hessian_head_v2")

    model_hip_0 = ScaleShiftMACE(**config, hip=True, num_interactions_hessian=0)
    assert model_hip_0.num_interactions_hessian == 0
    assert len(model_hip_0.hessian_head_v2.refine_interactions) == 0

    model_hip_2 = ScaleShiftMACE(**config, hip=True, num_interactions_hessian=2)
    assert model_hip_2.num_interactions_hessian == 2
    assert len(model_hip_2.hessian_head_v2.refine_interactions) == 2
    assert len(model_hip_2.hessian_head_v2.refine_products) == 2

    num_atoms = 5
    data = {
        "positions": torch.randn(num_atoms, 3, requires_grad=True),
        "node_attrs": torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2, 0, 1]), num_classes=3
        ).float(),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        "shifts": torch.zeros(4, 3),
        "unit_shifts": torch.zeros(4, 3),
        "cell": torch.eye(3).unsqueeze(0).expand(2, 3, 3),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "ptr": torch.tensor([0, 3, 5], dtype=torch.long),
    }

    output = model_hip_2(data, predict_hessian=False)
    assert output["energy"].shape[0] == 2
    assert output["hessian"] is None

    output = model_hip_2(data, predict_hessian=True)
    assert output["hessian"].shape[0] == num_atoms * 3
    assert output["hessian"].shape[1] == 3


if __name__ == "__main__":
    test_num_interactions_hessian()
    print("All tests passed!")
