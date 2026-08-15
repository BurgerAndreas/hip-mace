#!/usr/bin/env python3
"""Compute a global RMS scale for HORM training Hessian targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sum_squared = 0.0
    count = 0
    n_configs = 0
    for shard in sorted(args.train_path.glob("*.h5")):
        with h5py.File(shard, "r") as h5_file:
            for config in h5_file["config_batch_0"].values():
                hessian = np.asarray(config["properties"]["hessian"][()], dtype=np.float64)
                sum_squared += float(np.square(hessian).sum())
                count += hessian.size
                n_configs += 1

    if count == 0:
        raise RuntimeError("No Hessian components were found")
    result = {
        "train_path": str(args.train_path.resolve()),
        "n_configs": n_configs,
        "n_hessian_components": count,
        "global_hessian_rms_eV_per_A2": float(np.sqrt(sum_squared / count)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
