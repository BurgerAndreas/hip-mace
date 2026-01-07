import torch
import argparse
import numpy as np
import wandb
import pandas as pd
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader as TGDataLoader

from mace.data import AtomicData, AtomicDataset
from mace.tools import to_numpy
from mace.modules.frequency_analysis import analyze_frequencies_np
from mace.tools import Z_TO_ATOM_SYMBOL
from mace.tools.checkpoint import load_checkpoint
from mace.calculators.mace import MACECalculator

def evaluate(
    lmdb_path,
    checkpoint_path,
    config_path,  # not used
    hessian_method,
    max_samples=None,
    wandb_run_id=None,
    wandb_kwargs={},
    redo=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up save location
    dataset_name = lmdb_path.split("/")[-1].split(".")[0]
    results_dir = "results_evalhorm"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_name = checkpoint_path.split("/")[-1].split(".")[0]
    results_file = (
        f"{results_dir}/{ckpt_name}_{dataset_name}_{hessian_method}_metrics.csv"
    )

    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)
        aggregated_results = None # only calculate below if re-evaluated
        return df_results, aggregated_results

    # Load model using correct MACE checkpoint API
    print(f"Loading model from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = checkpoint["model"]
    model = model.to(device)
    model.eval()

    # Load dataset using MACE AtomicDataset
    # Assumes dataset is a LMDB, see configs/horm_100.yaml
    print(f"Loading data from {lmdb_path}")
    dataset = AtomicDataset(lmdb_path)
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=True)

    sample_metrics = []
    n_samples = 0
    n_total_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)

    # Setup wandb if needed
    if wandb_run_id is None:
        wandb.init(
            project="hip-mace",
            name=f"{ckpt_name}_{hessian_method}_{max_samples if max_samples else 'all'}",
            config={
                "checkpoint": checkpoint_path,
                "dataset": lmdb_path,
                "max_samples": max_samples,
                "model_name": "MACE",
                "config_path": config_path,
                "hessian_method": hessian_method,
            },
            tags=["hormmetrics"],
            **wandb_kwargs,
        )

    # Main evaluation loop
    for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
        batch = batch.to(device)
        n_atoms = batch.pos.shape[0]

        # TIMING for hessian only (not I/O)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Forward pass: select autograd (default) or MACE HIP (predict) Hessian
        # Reference: test_hessian.py and test_hip.py
        batch.requires_grad = True
        with torch.no_grad() if hessian_method == "predict" else torch.enable_grad():
            if hessian_method == "autograd":
                # Use automatic differentiation (see tests/test_hessian.py)
                mace_out = model(batch, compute_hessian=True)
                hessian_model = mace_out["hessian"] # shape (n_atoms*3, n_atoms*3)
                energy_model = mace_out["energy"]
                force_model = mace_out["forces"]
            elif hessian_method == "predict":
                # Use HIP predicted hessian (see tests/test_hip.py)
                mace_out = model(batch, predict_hessian=True)
                hessian_model = mace_out["hessian"]
                energy_model = mace_out["energy"]
                force_model = mace_out["forces"]
            else:
                raise ValueError("Choose hessian_method from ['autograd', 'predict'].")

        end_event.record()
        torch.cuda.synchronize()
        time_taken = start_event.elapsed_time(end_event) 
        memory_usage = torch.cuda.max_memory_allocated() / 1e6 # MB

        hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)

        # True data
        ae_true = batch.y
        force_true = batch.forces
        hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)

        # Eigendecomposition
        eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)
        eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

        # Compute errors
        e_error = torch.mean(torch.abs(energy_model.squeeze() - ae_true))
        f_error = torch.mean(torch.abs(force_model - force_true))
        h_error = torch.mean(torch.abs(hessian_model - hessian_true))
        asymmetry_error = torch.mean(torch.abs(hessian_model - hessian_model.T))
        true_asymmetry_error = torch.mean(torch.abs(hessian_true - hessian_true.T))

        # Additional (as in original)
        eigval_mae = torch.mean(torch.abs(eigvals_model - eigvals_true))
        eigval1_mae = torch.abs(eigvals_model[0] - eigvals_true[0])
        eigval2_mae = torch.abs(eigvals_model[1] - eigvals_true[1])
        eigvec1_cos = torch.abs(torch.dot(eigvecs_model[:, 0], eigvecs_true[:, 0]))
        eigvec2_cos = torch.abs(torch.dot(eigvecs_model[:, 1], eigvecs_true[:, 1]))

        # Analyze frequency & Eckart (mass weighting)
        symbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z]
        true_freqs = analyze_frequencies_np(
            hessian=hessian_true.detach().cpu().numpy(),
            cart_coords=batch.pos.detach().cpu().numpy(),
            atomsymbols=symbols,
        )
        true_neg_num = true_freqs["neg_num"]
        true_eigvecs_eckart = torch.tensor(true_freqs["eigvecs"])
        true_eigvals_eckart = torch.tensor(true_freqs["eigvals"])

        freqs_model = analyze_frequencies_np(
            hessian=hessian_model.detach().cpu().numpy(),
            cart_coords=batch.pos.detach().cpu().numpy(),
            atomsymbols=symbols,
        )
        freqs_model_neg_num = freqs_model["neg_num"]
        eigvecs_model_eckart = torch.tensor(freqs_model["eigvecs"])
        eigvals_model_eckart = torch.tensor(freqs_model["eigvals"])

        # Collect metrics
        sample_data = {
            "sample_idx": n_samples,
            "natoms": n_atoms,
            "energy_error": e_error.item(),
            "forces_error": f_error.item(),
            "hessian_error": h_error.item(),
            "asymmetry_error": asymmetry_error.item(),
            "true_asymmetry_error": true_asymmetry_error.item(),
            "eigval_mae": eigval_mae.item(),
            "eigval1_mae": eigval1_mae.item(),
            "eigval2_mae": eigval2_mae.item(),
            "eigvec1_cos": eigvec1_cos.item(),
            "eigvec2_cos": eigvec2_cos.item(),
            "time": float(time_taken),
            "memory": float(memory_usage),
            "true_neg_num": true_neg_num,
            "true_is_minima": int(true_neg_num == 0),
            "true_is_ts": int(true_neg_num == 1),
            "true_is_ts_order2": int(true_neg_num == 2),
            "true_is_higher_order": int(true_neg_num > 2),
            "model_neg_num": freqs_model_neg_num,
            "model_is_ts": int(freqs_model_neg_num == 1),
            "model_is_minima": int(freqs_model_neg_num == 0),
            "model_is_ts_order2": int(freqs_model_neg_num == 2),
            "model_is_higher_order": int(freqs_model_neg_num > 2),
            "neg_num_agree": int(true_neg_num == freqs_model_neg_num),
            "eigval_mae_eckart": torch.mean(
                torch.abs(eigvals_model_eckart - true_eigvals_eckart)
            ).item(),
            "eigval1_mae_eckart": torch.abs(
                eigvals_model_eckart[0] - true_eigvals_eckart[0]
            ).item(),
            "eigval2_mae_eckart": torch.abs(
                eigvals_model_eckart[1] - true_eigvals_eckart[1]
            ).item(),
            "eigvec1_cos_eckart": torch.abs(
                torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
            ).item(),
            "eigvec2_cos_eckart": torch.abs(
                torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
            ).item(),
        }

        sample_metrics.append(sample_data)
        n_samples += 1
        torch.cuda.empty_cache()

        if max_samples is not None and n_samples >= max_samples:
            break

    # Save per-sample results
    df_results = pd.DataFrame(sample_metrics)
    df_results.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    # Aggregate
    aggregated_results = {
        "energy_mae": df_results["energy_error"].mean(),
        "forces_mae": df_results["forces_error"].mean(),
        "hessian_mae": df_results["hessian_error"].mean(),
        "asymmetry_mae": df_results["asymmetry_error"].mean(),
        "true_asymmetry_mae": df_results["true_asymmetry_error"].mean(),
        "eigval_mae": df_results["eigval_mae"].mean(),
        "eigval1_mae": df_results["eigval1_mae"].mean(),
        "eigval2_mae": df_results["eigval2_mae"].mean(),
        "eigvec1_cos": df_results["eigvec1_cos"].mean(),
        "eigvec2_cos": df_results["eigvec2_cos"].mean(),
        # Eckart projection
        "eigval_mae_eckart": df_results["eigval_mae_eckart"].mean(),
        "eigval1_mae_eckart": df_results["eigval1_mae_eckart"].mean(),
        "eigval2_mae_eckart": df_results["eigval2_mae_eckart"].mean(),
        "eigvec1_cos_eckart": df_results["eigvec1_cos_eckart"].mean(),
        "eigvec2_cos_eckart": df_results["eigvec2_cos_eckart"].mean(),
        # Frequencies
        "neg_num_agree": df_results["neg_num_agree"].mean(),
        "true_neg_num": df_results["true_neg_num"].mean(),
        "model_neg_num": df_results["model_neg_num"].mean(),
        "true_is_ts": df_results["true_is_ts"].mean(),
        "true_is_minima": df_results["true_is_minima"].mean(),
        "true_is_ts_order2": df_results["true_is_ts_order2"].mean(),
        "true_is_higher_order": df_results["true_is_higher_order"].mean(),
        "model_is_ts": df_results["model_is_ts"].mean(),
        "model_is_minima": df_results["model_is_minima"].mean(),
        "model_is_ts_order2": df_results["model_is_ts_order2"].mean(),
        "model_is_higher_order": df_results["model_is_higher_order"].mean(),
        "is_ts_agree": (df_results["model_is_ts"] == df_results["true_is_ts"]).mean(),
        # Speed
        "time": df_results["time"].mean(),  # ms
        "memory": df_results["memory"].mean(),
    }

    wandb.log(aggregated_results)
    if wandb_run_id is None:
        wandb.finish()

    return df_results, aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MACE model on dataset (autograd vs HIP hessian)")
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        required=True,
        help="Path to checkpoint file (MACE .pt)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file. (not used)",
    )
    parser.add_argument(
        "--hessian_method",
        "-hm",
        choices=["autograd", "predict"],
        type=str,
        required=True,
        help="Hessian computation method: autograd (autodiff), predict (HIP/predicted)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset file name or path (e.g. RGD1_100.lmdb, see configs/horm_100.yaml)",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all samples)",
    )
    parser.add_argument(
        "--redo",
        "-r",
        type=bool,
        default=False,
        help="Run eval from scratch even if results already exist",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    checkpoint_path = args.ckpt_path
    lmdb_path = args.dataset
    max_samples = args.max_samples
    config_path = args.config_path
    hessian_method = args.hessian_method
    redo = args.redo

    df_results, aggregated_results = evaluate(
        lmdb_path=lmdb_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        hessian_method=hessian_method,
        max_samples=max_samples,
        redo=redo,
    )

