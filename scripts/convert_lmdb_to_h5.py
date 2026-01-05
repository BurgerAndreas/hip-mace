"""Convert a torch_geometric LMDB (or directory of .lmdb) to MACE HDF5 shards.

Writes sharded HDF5 files under <h5_prefix>/train (train_0.h5 ...). Extracts
isolated-atom energies (E0s) and writes a statistics.json (atomic_energies,
avg_num_neighbors not computed here, mean/std not computed here).

Stores all torch_geometric.Data properties under the MACE `properties` group
by canonical names (energy->'energy', forces->'forces', stress->'stress',
dipole->'dipole', charges->'charges', polarizability->'polarizability',
and keeps 'hessian' if present).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import traceback
from tqdm import tqdm

import h5py
import numpy as np
import torch

from mace.data.utils import Configuration, save_configurations_as_HDF5
from mace.tools.default_keys import DefaultKeys
from mace.modules import compute_statistics
from mace.tools import torch_geometric
from mace.data.hdf5_dataset import HDF5Dataset
from mace.tools.utils import AtomicNumberTable

from mace.data.horm_lmdb import GLOBAL_ATOM_NUMBERS, HormLmdbDataset

def one_hot_to_atomic_numbers(one_hot: torch.Tensor) -> np.ndarray:
    if hasattr(one_hot, "argmax"):
        inds = one_hot.long().argmax(dim=1)
        if GLOBAL_ATOM_NUMBERS is None:
            raise RuntimeError("GLOBAL_ATOM_NUMBERS not available; provide data.z")
        zs = GLOBAL_ATOM_NUMBERS.to(inds.device)[inds].cpu().numpy().astype(int)
        return zs
    raise ValueError("one_hot must be a tensor with atom-type one-hot rows")


def data_to_configuration(data, head_name: str = "Default") -> Optional[Configuration]:
    """Convert a torch_geometric Data-like object to a MACE Configuration.

    Always returns a Configuration (do not extract isolated-atom energies).
    """
    # atomic numbers
    if hasattr(data, "z"):
        atomic_numbers = np.array(data.z.cpu().numpy(), dtype=int)
    elif hasattr(data, "one_hot"):
        atomic_numbers = one_hot_to_atomic_numbers(data.one_hot)
    else:
        raise RuntimeError("No atomic number info found on data (z or one_hot required)")

    natoms = int(len(atomic_numbers))

    # positions
    positions = data.pos.cpu().numpy()

    # properties mapping: use MACE property keys (energy, forces, stress, dipole, charges, polarizability)
    props = {}
    prop_weights = {}

    # helpers
    def get_attr(name):
        return getattr(data, name) if hasattr(data, name) else None

    energy = get_attr("energy")
    if energy is not None:
        props["energy"] = float(energy)
        prop_weights["energy"] = 1.0
    else:
        prop_weights["energy"] = 0.0

    forces = get_attr("forces")
    if forces is not None:
        props["forces"] = forces.cpu().numpy()
        prop_weights["forces"] = 1.0
    else:
        prop_weights["forces"] = 0.0

    stress = get_attr("stress")
    if stress is not None:
        props["stress"] = np.array(stress)
        prop_weights["stress"] = 1.0
    else:
        prop_weights["stress"] = 0.0

    dipole = get_attr("dipole")
    if dipole is not None:
        props["dipole"] = np.array(dipole)
        prop_weights["dipole"] = 1.0
    else:
        prop_weights["dipole"] = 0.0

    charges = get_attr("charges")
    if charges is not None:
        props["charges"] = np.array(charges)
        prop_weights["charges"] = 1.0
    else:
        prop_weights["charges"] = 0.0

    polarizability = get_attr("polarizability")
    if polarizability is not None:
        props["polarizability"] = np.array(polarizability)
        prop_weights["polarizability"] = 1.0
    else:
        prop_weights["polarizability"] = 0.0

    # include hessian if present (training may use it later)
    hessian = get_attr("hessian")
    if hessian is not None:
        # numpy dump of hessian (flattened or array)
        try:
            props["hessian"] = np.array(hessian)
        except Exception:
            props["hessian"] = hessian

    # cell and pbc
    cell = None
    pbc = None
    if hasattr(data, "cell"):
        try:
            cell = np.array(data.cell)
        except Exception:
            cell = None
    if hasattr(data, "pbc"):
        try:
            pbc = tuple(data.pbc)
        except Exception:
            pbc = None

    config_type = getattr(data, "config_type", "Default")
    weight = float(getattr(data, "weight", 1.0))

    # Build Configuration (do not special-case isolated atoms)
    config = Configuration(
        atomic_numbers=np.array(atomic_numbers),
        positions=np.array(positions) if positions is not None else None,
        properties=props,
        property_weights=prop_weights,
        cell=cell,
        pbc=pbc,
        weight=weight,
        config_type=config_type,
        head=head_name,
    )
    return config


def run_conversion(
    input_path: str,
    h5_prefix: str,
    r_max = 5.0
):
    if not os.path.exists(input_path):
        # look in default cache location
        dataset_dir = os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
        )
        if os.path.exists(os.path.join(dataset_dir, input_path)):
            input_path = os.path.join(dataset_dir, input_path)
    
    # Prepare output path: if h5_prefix ends with .h5 use it, else create directory
    if h5_prefix is None:
        out_dir = os.path.dirname(input_path)
        base = os.path.basename(os.path.normpath(input_path))
        base = base.replace(".lmdb", "")
        out_path = os.path.join(out_dir, f"{base}.h5")
    elif h5_prefix.endswith('.h5'):
        # is a filename
        out_dir = os.path.dirname(h5_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        # is a directory
        os.makedirs(h5_prefix, exist_ok=True)
        base = os.path.basename(os.path.normpath(input_path)).strip(".lmdb")
        out_path = os.path.join(h5_prefix, f"{base}.h5")
    
    ds = HormLmdbDataset(input_path)
    indices = list(range(len(ds)))

    configs: List[Configuration] = []

    for idx in tqdm(indices):
        data = ds[idx]
        config = data_to_configuration(data)
        configs.append(config)

    print(f"Writing {len(configs)} configurations to {out_path}")
    with h5py.File(out_path, "w") as f:
        save_configurations_as_HDF5(list(configs), 0, f)

    print(f"Wrote {len(configs)} configurations")
    

    # Compute dataset statistics (average neighbors, mean per-atom energies, RMS of forces)
    # Create a DataLoader over the newly written HDF5 file to compute graph-based stats
    # Construct z_table from GLOBAL_ATOM_NUMBERS and set a default cutoff r_max
    print("\nComputing dataset statistics")
    z_list = GLOBAL_ATOM_NUMBERS.tolist()
    z_table = AtomicNumberTable(z_list)
    h5_dataset = HDF5Dataset(out_path, r_max=r_max, z_table=z_table)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=h5_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    # If isolated atomic energies are not known, fall back to zeros for each global atom
    atomic_energies = np.zeros(len(GLOBAL_ATOM_NUMBERS), dtype=float)

    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)

    stats = {
        "atomic_energies": atomic_energies.tolist(),
        "avg_num_neighbors": float(avg_num_neighbors),
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
    }

    stats_path = str(Path(out_path).with_suffix(".json"))
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    print(f"Wrote statistics to {stats_path}")

    # Print sizes of original LMDB (file or directory) and the new HDF5 file
    def get_size_bytes(p: str) -> int:
        if os.path.isfile(p):
            try:
                return os.path.getsize(p)
            except Exception:
                return 0
        if os.path.isdir(p):
            total = 0
            for root, _, files in os.walk(p):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except Exception:
                        continue
            return total
        return 0

    def human_readable(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        x = float(n)
        for u in units:
            if x < 1024.0:
                return f"{x:.1f}{u}"
            x /= 1024.0
        return f"{x:.1f}PB"

    orig_size = get_size_bytes(input_path)
    new_size = get_size_bytes(out_path)
    print(f"Original LMDB size: {orig_size} bytes ({human_readable(orig_size)})")
    print(f"New HDF5 size: {new_size} bytes ({human_readable(new_size)})")


def main():
    parser = argparse.ArgumentParser()
    # dataset_files = [
    #     "ts1x-val.lmdb",
    #     "ts1x_hess_train_big.lmdb",
    #     "RGD1.lmdb",
    # ]
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--h5_prefix", default=None, help="Output HDF5 prefix (directory or .h5 file)")
    args = parser.parse_args()
    run_conversion(args.in_file, args.h5_prefix)


if __name__ == "__main__":
    main()
