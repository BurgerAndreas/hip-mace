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
from datetime import datetime
from typing import List, Optional
import shutil
from tqdm import tqdm

import h5py
import numpy as np
import torch

from mace.data.utils import Configuration, save_configurations_as_HDF5
from mace.tools.default_keys import DefaultKeys
from mace.modules import compute_statistics
from mace.tools import torch_geometric
from mace.data.hdf5_dataset import HDF5Dataset, dataset_from_sharded_hdf5
from mace.tools.utils import AtomicNumberTable

from mace.data.horm_lmdb import GLOBAL_ATOM_NUMBERS, HormLmdbDataset


def tprint(*args, **kwargs):
    """Print with timestamp prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)


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


def write_shard(configs: List[Configuration], shard_path: str) -> None:
    """Write a list of configurations to a single HDF5 shard file."""
    with h5py.File(shard_path, "w") as f:
        save_configurations_as_HDF5(configs, 0, f)


def resolve_input_path(input_path: str) -> str:
    """Resolve input_path, checking kagglehub cache if not found directly."""
    if os.path.exists(input_path):
        return input_path
    base_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/"
    )
    versions_dir = os.path.join(base_dir, "versions")
    if os.path.isdir(versions_dir):
        version_nums = sorted(
            (int(v) for v in os.listdir(versions_dir) if v.isdigit()),
            reverse=True,
        )
        for v in version_nums:
            candidate = os.path.join(versions_dir, str(v), input_path)
            if os.path.exists(candidate):
                return candidate
    return input_path


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


def run_conversion(
    input_path: str,
    h5_prefix: str,
    r_max: float,
    shard_size: int = 10000,
    skip_statistics: bool = False,
):
    input_path = resolve_input_path(input_path)

    # Output directory for shards
    if h5_prefix is not None:
        out_dir = h5_prefix
    else:
        parent = os.path.dirname(input_path)
        base = os.path.basename(os.path.normpath(input_path)).replace(".lmdb", "")
        out_dir = os.path.join(parent, base)

    # Clean up existing output directory
    if os.path.isdir(out_dir):
        tprint(f"Removing existing output directory {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ds = HormLmdbDataset(input_path)
    num_configs = len(ds)
    num_shards = (num_configs + shard_size - 1) // shard_size
    tprint(f"Converting {num_configs} configs -> {num_shards} shards of {shard_size} in {out_dir}")

    # Stream: read a shard's worth of configs, write to H5, free memory
    total_written = 0
    pbar = tqdm(total=num_configs, desc="Converting")
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, num_configs)
        configs = []
        for idx in range(start, end):
            data = ds[idx]
            configs.append(data_to_configuration(data))
            pbar.update(1)

        shard_path = os.path.join(out_dir, f"shard_{shard_idx}.h5")
        write_shard(configs, shard_path)
        total_written += len(configs)
        del configs  # free memory

    pbar.close()
    tprint(f"Wrote {total_written} configs across {num_shards} shards to {out_dir}")

    if skip_statistics:
        tprint("Skipping dataset statistics computation")
        orig_size = get_size_bytes(input_path)
        new_size = get_size_bytes(out_dir)
        tprint(f"Original LMDB size: {human_readable(orig_size)}")
        tprint(f"Sharded HDF5 size: {human_readable(new_size)} ({num_shards} files)")
        tprint(f"Output directory: {os.path.abspath(out_dir)}")
        return

    # Compute dataset statistics
    tprint(f"Computing dataset statistics with r_max={r_max}")
    z_list = GLOBAL_ATOM_NUMBERS.tolist()
    z_table = AtomicNumberTable(z_list)
    h5_dataset = dataset_from_sharded_hdf5(out_dir, z_table=z_table, r_max=r_max)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=h5_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    atomic_energies = np.zeros(len(GLOBAL_ATOM_NUMBERS), dtype=float)
    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)

    stats = {
        "atomic_energies": atomic_energies.tolist(),
        "avg_num_neighbors": float(avg_num_neighbors),
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
    }

    stats_path = os.path.join(out_dir, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    tprint(f"Wrote statistics to {stats_path}")

    orig_size = get_size_bytes(input_path)
    new_size = get_size_bytes(out_dir)
    tprint(f"Original LMDB size: {human_readable(orig_size)}")
    tprint(f"Sharded HDF5 size: {human_readable(new_size)} ({num_shards} files)")
    tprint(f"Output directory: {os.path.abspath(out_dir)}")


def main():
    parser = argparse.ArgumentParser()
    # dataset_files = [
    #     "ts1x-val.lmdb",
    #     "ts1x_hess_train.lmdb",
    #     "RGD1.lmdb",
    # ]
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--h5_prefix", default=None, help="Output directory for HDF5 shards")
    parser.add_argument("--r_max", default=5.0, type=float, help="Cutoff radius for computing statistics.")
    parser.add_argument("--shard_size", default=10000, type=int, help="Configs per HDF5 shard file")
    parser.add_argument(
        "--skip_statistics",
        action="store_true",
        help="Only write HDF5 shards; skip the post-conversion statistics pass.",
    )
    args = parser.parse_args()
    run_conversion(
        args.in_file,
        args.h5_prefix,
        args.r_max,
        args.shard_size,
        args.skip_statistics,
    )


if __name__ == "__main__":
    main()
