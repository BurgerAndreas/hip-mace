#!/usr/bin/env python3
"""Least-squares fit elemental E0s on a HORM LMDB train subset and compare to Transition1x refs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# Transition1x REFERENCE_ENERGIES (eV), copied into horm_* configs
T1X_E0S = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
}
ATOMIC_NUMBERS = list(T1X_E0S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train_path",
        default="./data/ts1x/ts1x_hess_train",
        help="HORM LMDB file or directory of shards",
    )
    p.add_argument("--n_samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default="results/horm_e0_fit/e0_fit_comparison.json",
        help="JSON output path",
    )
    return p.parse_args()


def composition_counts(z: np.ndarray, zs: list[int]) -> np.ndarray:
    return np.array([(z == atomic_number).sum() for atomic_number in zs], dtype=np.float64)


def residual_stats(A: np.ndarray, B: np.ndarray, e0: np.ndarray) -> dict[str, float]:
    pred = A @ e0
    resid = B - pred
    n_atoms = A.sum(axis=1)
    per_atom = resid / np.maximum(n_atoms, 1.0)
    return {
        "mae_eV": float(np.mean(np.abs(resid))),
        "rmse_eV": float(np.sqrt(np.mean(resid**2))),
        "mae_per_atom_eV": float(np.mean(np.abs(per_atom))),
        "rmse_per_atom_eV": float(np.sqrt(np.mean(per_atom**2))),
        "mean_resid_eV": float(np.mean(resid)),
        "mean_per_atom_eV": float(np.mean(per_atom)),
        "std_per_atom_eV": float(np.std(per_atom, ddof=1)),
        "mace_mean": float(np.mean(per_atom)),
        "mace_std": float(np.std(per_atom, ddof=1)),
    }


def read_hdf5_subset(path: Path, indices: np.ndarray, zs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Read selected configurations from the sharded HORM HDF5 conversion."""
    files = sorted(path.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 shards found in {path}")

    lengths = []
    for file_path in files:
        with h5py.File(file_path, "r") as h5_file:
            lengths.append(len(h5_file["config_batch_0"]))
    cumulative = np.cumsum([0, *lengths])

    A = np.zeros((len(indices), len(zs)), dtype=np.float64)
    B = np.zeros(len(indices), dtype=np.float64)
    for row, index in enumerate(indices):
        file_index = int(np.searchsorted(cumulative[1:], index, side="right"))
        config_index = int(index - cumulative[file_index])
        with h5py.File(files[file_index], "r") as h5_file:
            config = h5_file["config_batch_0"][f"config_{config_index}"]
            A[row] = composition_counts(config["atomic_numbers"][()], zs)
            B[row] = float(config["properties"]["energy"][()])
    return A, B


def read_hdf5_all(path: Path, zs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Read every configuration once, keeping each HDF5 shard open per pass."""
    rows = []
    energies = []
    for file_path in sorted(path.glob("*.h5")):
        with h5py.File(file_path, "r") as h5_file:
            for config in h5_file["config_batch_0"].values():
                rows.append(composition_counts(config["atomic_numbers"][()], zs))
                energies.append(float(config["properties"]["energy"][()]))
    return np.asarray(rows, dtype=np.float64), np.asarray(energies, dtype=np.float64)


def main() -> None:
    args = parse_args()
    zs = ATOMIC_NUMBERS
    train_path = Path(args.train_path)
    if train_path.is_dir() and list(train_path.glob("*.h5")):
        files = sorted(train_path.glob("*.h5"))
        n_total = 0
        for file_path in files:
            with h5py.File(file_path, "r") as h5_file:
                n_total += len(h5_file["config_batch_0"])
    else:
        raise ValueError(
            "This fitting utility expects the converted HORM HDF5 directory "
            "(for example data/ts1x/ts1x_hess_train)."
        )
    n = min(args.n_samples, n_total) if args.n_samples > 0 else n_total
    if n == n_total:
        A, B = read_hdf5_all(train_path, zs)
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(n_total, size=n, replace=False)
        indices.sort()
        A, B = read_hdf5_subset(train_path, indices, zs)

    e0_fit, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)
    fitted = {z: float(e0_fit[i]) for i, z in enumerate(zs)}
    ref = np.array([T1X_E0S[z] for z in zs], dtype=np.float64)

    out = {
        "train_path": str(Path(args.train_path).resolve()),
        "n_total": n_total,
        "n_samples": n,
        "seed": args.seed,
        "atomic_numbers": zs,
        "lstsq_rank": int(rank),
        "lstsq_singular_values": [float(x) for x in singular],
        "t1x_reference_E0s": {str(z): T1X_E0S[z] for z in zs},
        "fitted_E0s": {str(z): fitted[z] for z in zs},
        "delta_fitted_minus_t1x": {str(z): fitted[z] - T1X_E0S[z] for z in zs},
        "residuals_with_t1x_E0s": residual_stats(A, B, ref),
        "residuals_with_fitted_E0s": residual_stats(A, B, e0_fit),
        "note": (
            "Fitted E0s are least-squares average atomic energies on molecular "
            "compositions (MACE E0s='average'), not isolated-atom DFT energies. "
            "HORM energies were recomputed with GPU4PySCF; T1x refs come from "
            "Transition1x REFERENCE_ENERGIES at ωB97x/6-31G(d)."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote {out_path}")
    print(f"n_samples={n}/{n_total} rank={rank}")
    print("Z  symbol   T1x_E0           fitted           delta")
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}
    for z in zs:
        print(
            f"{z:2d}  {symbols[z]:1s}     "
            f"{T1X_E0S[z]:16.8f}  {fitted[z]:16.8f}  {fitted[z] - T1X_E0S[z]:+12.6f}"
        )
    print("\nResidual energy after subtracting sum(n_z * E0_z):")
    for label, stats in (
        ("T1x refs", out["residuals_with_t1x_E0s"]),
        ("fitted", out["residuals_with_fitted_E0s"]),
    ):
        print(
            f"  {label:8s}  MAE/atom={stats['mae_per_atom_eV']:.6f} eV  "
            f"RMSE/atom={stats['rmse_per_atom_eV']:.6f} eV  "
            f"mean/atom={stats['mean_per_atom_eV']:.6f} eV"
        )


if __name__ == "__main__":
    main()
