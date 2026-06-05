import tempfile

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim

from mace.tools import (
    AtomicNumberTable,
    CheckpointHandler,
    CheckpointState,
    atomic_numbers_to_indices,
)
from mace.tools.train import MACELoss


def test_atomic_number_table():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_indices(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]["lr"], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]["lr"], initial_lr)


class ZeroLoss(nn.Module):
    def forward(self, pred, ref):
        return torch.zeros((), dtype=ref.hessian.dtype, device=ref.hessian.device)


class HessianBatch:
    num_graphs = 2
    energy = None
    forces = None
    stress = None
    virials = None
    dipole = None
    polarizability = None

    def __init__(self, hessian):
        self.hessian = hessian
        self.ptr = torch.tensor([0, 2, 3])


def test_mace_loss_reports_hessian_block_maes():
    graph_0 = torch.zeros(6, 6)
    graph_0[:3, :3] = 1.0
    graph_0[3:, 3:] = 3.0
    graph_0[:3, 3:] = 5.0
    graph_0[3:, :3] = 7.0
    graph_1 = torch.full((3, 3), 9.0)
    hessian_delta = torch.cat([graph_0.reshape(-1), graph_1.reshape(-1)])

    metric = MACELoss(loss_fn=ZeroLoss())
    metric.update(
        HessianBatch(hessian=hessian_delta),
        {"hessian": torch.zeros_like(hessian_delta)},
    )
    _, aux = metric.compute()

    assert np.isclose(aux["mae_h"], 5.0)
    assert np.isclose(aux["mae_h_diag"], 117.0 / 27.0)
    assert np.isclose(aux["mae_h_off_diag"], 6.0)
