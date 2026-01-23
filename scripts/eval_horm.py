import torch
import argparse
import numpy as np
import wandb
import pandas as pd
import os
import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader as TGDataLoader

from mace import data, modules, tools

from mace.modules.frequency_analysis import analyze_frequencies_np, Z_TO_ATOM_SYMBOL
from mace.tools.torch_tools import to_numpy
from mace.tools.checkpoint import CheckpointIO, CheckpointBuilder
from mace.calculators.mace import MACECalculator
from mace.tools.run_train_utils import (
    combine_datasets,
    load_dataset_for_path,
    normalize_file_paths,
)
from mace.tools.multihead_tools import (
    HeadConfig,
    apply_pseudolabels_to_pt_head_configs,
    assemble_replay_data,
    dict_head_to_dataclass,
    prepare_default_head,
    prepare_pt_head,
)
from mace.data import KeySpecification, update_keyspec_from_kwargs

import argparse
from typing import Dict

import ase.data
import ase.io
import numpy as np
import torch
from e3nn import o3
import glob

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils


def load_config_from_wandb_run(wandb_run: str, wandb_project: str = None, wandb_entity: str = None) -> dict:
    """
    Load config from a wandb run.

    Args:
        wandb_run: Either a full run path "entity/project/run_id" or just a run_id
        wandb_project: Project name (required if wandb_run is just a run_id)
        wandb_entity: Entity name (optional, uses default if not provided)

    Returns:
        dict: The run config including checkpoints_dir
    """
    api = wandb.Api()

    # Parse the wandb_run argument
    parts = wandb_run.split("/")
    if len(parts) == 3:
        # Full path: entity/project/run_id
        run_path = wandb_run
    elif len(parts) == 2:
        # project/run_id format
        run_path = wandb_run
    elif len(parts) == 1:
        # Just run_id, need project (and optionally entity)
        if wandb_project is None:
            raise ValueError(
                "wandb_project is required when wandb_run is just a run_id. "
                "Either provide --wandb_project or use full path: entity/project/run_id"
            )
        if wandb_entity:
            run_path = f"{wandb_entity}/{wandb_project}/{wandb_run}"
        else:
            run_path = f"{wandb_project}/{wandb_run}"
    else:
        raise ValueError(
            f"Invalid wandb_run format: {wandb_run}. "
            "Use either 'run_id', 'project/run_id', or 'entity/project/run_id'"
        )

    print(f"Loading config from wandb run: {run_path}")
    run = api.run(run_path)

    # Get config from the run
    config = dict(run.config)

    # Also try to get full params from summary if available
    if "params" in run.summary:
        try:
            full_params = json.loads(run.summary["params"])
            # Merge full_params into config (full_params takes precedence for any overlapping keys)
            for key, value in full_params.items():
                if key not in config:
                    config[key] = value
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"Loaded config with checkpoints_dir: {config.get('checkpoints_dir', 'NOT FOUND')}")
    return config

def evaluate_hessian_on_horm_dataset(
    args: argparse.Namespace,
    max_samples=None,
    redo=False,
):
    print("\n\n--- Evaluating Hessian on HORM dataset ---")
    hessian_method = "predict" if args.predict_hessian else "autograd"
    print("Hessian method: ", hessian_method)
    
    # checkpoint usually looks like this:
    # checkpoints/20260105_112332/horm100_run-42/
    # horm100_run-42_epoch-99.pt  horm100_run-42.model
    # .model file ontains only the trained model weights and architecture, without optimizer state
    model_files = glob.glob(str(args.checkpoints_dir + "/*.model"))
    if len(model_files) == 0:
        raise FileNotFoundError(f"No .model file found in {args.checkpoints_dir}: {os.listdir(args.checkpoints_dir)}")
    if len(model_files) > 1:
        raise ValueError(f"Multiple .model files found in {args.checkpoints_dir}: {model_files}")
    checkpoint_path = model_files[0]
    
    device = torch_tools.init_device(args.device)

    # Check bfloat16 support if requested
    if args.default_dtype == "bfloat16":
        if not torch_tools.check_bfloat16_support(device):
            if device.type == "cuda":
                capability = torch.cuda.get_device_capability(device)
                print(
                    f"WARNING: GPU compute capability {capability[0]}.{capability[1]} does not fully support bfloat16. "
                    f"Evaluation may produce incorrect results. Consider using float32 instead."
                )
            else:
                print(f"WARNING: Device {device} may not fully support bfloat16.")
        else:
            print(f"Device {device} supports bfloat16 evaluation")

    torch_tools.set_default_dtype(args.default_dtype)

    # Load model
    model = torch.load(f=checkpoint_path, map_location=args.device)
    if args.enable_cueq:
        print("Converting models to CuEq for acceleration")
        model = run_e3nn_to_cueq(model, device=device)
    # shouldn't be necessary but seems to help with CUDA problems
    model = model.to(
        args.device
    )  

    for param in model.parameters():
        param.requires_grad = False

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))
    dataset = load_dataset_for_path(
        file_path=args.valid_file,
        r_max=args.r_max,
        z_table=z_table,
        head_config=HeadConfig("default", args.key_specification),
        heads=["Default"],
    )
    dataloader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    
    # Set up save location
    dataset_name = args.valid_file.split("/")[-1].split(".")[0]
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


    sample_metrics = []
    n_samples = 0
    n_total_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)

    # Setup wandb if needed
    wandb.init(
        project=args.wandb_project,
        name=f"{ckpt_name}_{hessian_method}_{max_samples if max_samples else 'all'}",
        config={
            "checkpoint_path": checkpoint_path,
            "max_samples": max_samples,
            "hessian_method": hessian_method,
        },
        tags=["hormmetrics"],
        id=args.wandb_run_id,
        #  If a run with the specified id exists, it will resume,
        # otherwise, a new run will be created.
        resume="allow", 
    )

    # Main evaluation loop
    for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
        batch = batch.to(device)
        n_atoms = batch["positions"].shape[0]
        if n_samples == 0:
            print(batch.keys)

        # TIMING for hessian only (not I/O)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Forward pass: select autograd (default) or MACE HIP (predict) Hessian
        # Reference: test_hessian.py and test_hip.py
        batch["positions"].requires_grad = True # for copmuting the forces
        # with torch.no_grad() if hessian_method == "predict" else torch.enable_grad():
        if hessian_method == "autograd":
            # Use automatic differentiation (see tests/test_hessian.py)
            mace_out = model(batch, compute_hessian=True, predict_hessian=False)
            hessian_model = mace_out["hessian"] # shape (n_atoms*3, n_atoms*3)
            energy_model = mace_out["energy"]
            force_model = mace_out["forces"]
        elif hessian_method == "predict":
            # Use HIP predicted hessian (see tests/test_hip.py)
            mace_out = model(batch, compute_hessian=False, predict_hessian=True)
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

        # True data - using correct batch keys
        ae_true = batch["energy"]
        force_true = batch["forces"]
        hessian_true = batch["hessian"].reshape(n_atoms * 3, n_atoms * 3)

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

        # node_attrs is one-hot encoded, so we need to convert it to atomic numbers
        batch["z"] = torch.tensor([z_table.index_to_z(z) for z in batch["node_attrs"].argmax(dim=-1)])
        symbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in batch["z"]]
        
        # Analyze frequency & Eckart (mass weighting)
        true_freqs = analyze_frequencies_np(
            hessian=to_numpy(hessian_true),
            cart_coords=to_numpy(batch["positions"]),
            atomsymbols=symbols,
        )
        true_neg_num = true_freqs["neg_num"]
        true_eigvecs_eckart = torch.tensor(true_freqs["eigvecs"])
        true_eigvals_eckart = torch.tensor(true_freqs["eigvals"])

        freqs_model = analyze_frequencies_np(
            hessian=to_numpy(hessian_model),
            cart_coords=to_numpy(batch["positions"]),
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
    wandb.finish()

    return df_results, aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MACE model on dataset (autograd vs HIP hessian)"
    )

    # Main argument: checkpoint dir OR wandb run ID (auto-detected)
    parser.add_argument(
        "run",
        type=str,
        help="Checkpoint directory path OR wandb run ID. Auto-detected: if path exists as directory, treated as checkpoint dir; otherwise as wandb run ID.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="hip-mace",
        help="Wandb project name (default: hip-mace)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity name (optional)",
    )

    # Evaluation options
    parser.add_argument(
        "--predict_hessian",
        action="store_true",
        default=False,
        help="Use HIP predicted hessian instead of autograd",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="Validation dataset file path (overrides config)",
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
        action="store_true",
        default=False,
        help="Run eval from scratch even if results already exist",
    )

    # Device and precision
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--default_dtype",
        type=str,
        default="float64",
        help="Default dtype (default: float64)",
    )
    parser.add_argument(
        "--enable_cueq",
        action="store_true",
        default=False,
        help="Enable CuEq acceleration",
    )

    # Data keys (can override from config)
    parser.add_argument("--energy_key", type=str, default="energy")
    parser.add_argument("--forces_key", type=str, default="forces")
    parser.add_argument("--hessian_key", type=str, default="hessian")

    args = parser.parse_args()

    # Auto-detect if run is a checkpoint dir or wandb run ID
    if os.path.isdir(args.run):
        # It's a checkpoint directory
        print(f"Detected checkpoint directory: {args.run}")
        args.checkpoints_dir = args.run
        args.wandb_run_id = None
    else:
        # Treat as wandb run ID
        print(f"Detected wandb run ID: {args.run}")
        config = load_config_from_wandb_run(
            wandb_run=args.run,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )

        # Set checkpoints_dir from wandb config
        if "checkpoints_dir" not in config:
            raise ValueError(f"checkpoints_dir not found in wandb run config")
        args.checkpoints_dir = config["checkpoints_dir"]

        # Set other args from config if not explicitly provided
        if args.valid_file is None and "valid_file" in config:
            args.valid_file = config["valid_file"]
        if "r_max" in config:
            args.r_max = config["r_max"]
        if "default_dtype" in config and args.default_dtype == "float64":
            # Only override if user didn't explicitly set it
            args.default_dtype = config["default_dtype"]

        # Use wandb run id for logging
        parts = args.run.split("/")
        args.wandb_run_id = parts[-1]  # Last part is always the run_id

    # Validate that we have required args
    if args.valid_file is None:
        parser.error("--valid_file is required (either directly or from wandb config)")

    # Set r_max if not already set (will be loaded from model)
    if not hasattr(args, "r_max"):
        args.r_max = 5.0  # Default, model will override if needed

    torch.manual_seed(42)

    df_results, aggregated_results = evaluate_hessian_on_horm_dataset(
        args=args,
        max_samples=args.max_samples,
        redo=args.redo,
    )

    if aggregated_results is not None:
        print("\n--- Aggregated Results ---")
        for key, value in aggregated_results.items():
            print(f"{key}: {value:.6f}")
