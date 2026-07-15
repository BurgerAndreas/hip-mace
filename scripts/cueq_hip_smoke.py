#!/usr/bin/env python3
"""Correctness and training-speed smoke test for CUEQ HIP heads."""

import argparse
import os
import statistics
import time
import warnings

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.modules.wrapper_ops import CuEquivarianceConfig


ATOMIC_NUMBERS = [1, 8]
POSITIONS = np.array(
    [[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
)
SPECIES = np.array([8, 1, 1])


def create_batch(device, positions=POSITIONS, batch_size=1):
    config = data.Configuration(
        atomic_numbers=SPECIES,
        positions=positions,
        properties={"energy": 0.0, "forces": np.zeros_like(positions)},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=tools.AtomicNumberTable(ATOMIC_NUMBERS),
        cutoff=6.0,
    )
    return next(
        iter(
            torch_geometric.dataloader.DataLoader(
                [atomic_data] * batch_size, batch_size=batch_size
            )
        )
    ).to(device).to_dict()


def create_model(head, device, channels=8, use_cueq=False):
    cueq_config = None
    if use_cueq:
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout="ir_mul",
            group="O3_e3nn",
            optimize_all=True,
            conv_fusion=True,
        )
    return modules.ScaleShiftMACE(
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
        hidden_irreps=o3.Irreps(
            f"{channels}x0e + {channels}x1e + {channels}x1o + {channels}x2e"
        ),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(ATOMIC_NUMBERS)),
        avg_num_neighbors=4,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=3,
        use_reduced_cg=False,
        radial_type="bessel",
        hip=True,
        hessian_feature_dim=channels,
        hessian_head_type=head,
        hessian_r_max=16.0,
        hessian_fully_connected=True,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        cueq_config=cueq_config,
    ).to(device)


def loss_from_output(output):
    return (
        output["energy"].square().mean()
        + output["forces"].square().mean()
        + output["hessian"].square().mean()
    )


def take_step(model, optimizer, batch):
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch, training=True, compute_force=True, predict_hessian=True
    )
    loss = loss_from_output(output)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return output, loss, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["pair_v2", "eqv2_v1", "message_v1"],
        choices=["pair_v2", "eqv2_v1", "message_v1"],
    )
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(device)}")

    for head in args.heads:
        torch.manual_seed(1)
        eager_model = create_model(head, device, channels=args.channels).train()
        torch.manual_seed(1)
        cueq_model = create_model(
            head, device, channels=args.channels, use_cueq=True
        ).train()
        eager_batch = create_batch(device, batch_size=args.batch_size)
        cueq_batch = create_batch(device, batch_size=args.batch_size)

        if args.batch_size == 1:
            angle = 0.37
            rotation = torch.tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.get_default_dtype(),
            )
            rotated_batch = create_batch(
                device, POSITIONS @ rotation.cpu().numpy().T
            )
            cueq_model.eval()
            with torch.no_grad():
                hessian = cueq_model(
                    cueq_batch,
                    training=False,
                    compute_force=False,
                    predict_hessian=True,
                )["hessian"].reshape(9, 9)
                rotated_hessian = cueq_model(
                    rotated_batch,
                    training=False,
                    compute_force=False,
                    predict_hessian=True,
                )["hessian"].reshape(9, 9)
            big_rotation = torch.kron(torch.eye(3, device=device), rotation)
            torch.testing.assert_close(
                rotated_hessian,
                big_rotation @ hessian @ big_rotation.T,
                atol=2e-4,
                rtol=2e-4,
            )
            cueq_model.train()

        eager_optimizer = torch.optim.AdamW(eager_model.parameters(), lr=1e-3)
        cueq_optimizer = torch.optim.AdamW(cueq_model.parameters(), lr=1e-3)
        take_step(eager_model, eager_optimizer, eager_batch)
        take_step(cueq_model, cueq_optimizer, cueq_batch)
        eager_times = [
            take_step(eager_model, eager_optimizer, eager_batch)[2]
            for _ in range(args.benchmark_steps)
        ]
        cueq_times = [
            take_step(cueq_model, cueq_optimizer, cueq_batch)[2]
            for _ in range(args.benchmark_steps)
        ]
        eager_median = statistics.median(eager_times)
        cueq_median = statistics.median(cueq_times)
        print(
            f"{head}: eager={eager_median * 1000:.2f}ms "
            f"cueq={cueq_median * 1000:.2f}ms "
            f"speedup={eager_median / cueq_median:.2f}x"
        )


if __name__ == "__main__":
    main()
