# HIP-MACE

An implementation of MACE as Hessian Interatomic Potential (HIP).
HIPs are MLIPs that predict the Hessian in addition to the force and energy.
The Hessian is the second derivative of the energy with respect to the input atom coordinates.

HIP was introduced in our paper https://arxiv.org/abs/2509.21624

The original implementation of HIP for EquiformerV2 is here: https://github.com/BurgerAndreas/hip

## HORM-T1x results (corrected E0s and normalization)

As expected, autograd Hessians are much worse than HIP predicted ones, and ~30x slower per sample on average on these small molecules.

| Method | E (meV) ↓ | F (meV·Å⁻¹) ↓ | H meV·Å⁻² ↓ | eigval MAE ↓ | 1st eigvec cos mean ↑ | 1st eigvec cos median ↑ | time (ms) ↓ |
|:-------|----:|----:|----:|-------------:|----------------------:|------------------------:|------------:|
| HIP predict | 26.2 | 32.7 | **65.7** | **0.0364** | **0.915** | **0.989** | **84.5** |
| Energy autograd ∂²E/∂r² | 26.2 | 32.7 | 308.5 | 0.7010 | 0.962 | 0.997 | 2737 |

Setup: hidden_irreps and hessian head dim 512, lr=0.08, loss weights e1 / f10 / h25, batch 64, 100 warmup + 1000 epochs, only_cueq=true.
Means over 10,000 `ts1x-val` samples from the corrected-E0 checkpoint. Eigvalue and eigvectir metrics use Eckart (mass-weighted) projection.
Time is measured as the model forward (E/F + Hessian) using CUDA-events, in ms/sample.

### HORM-T1x E0 reference and normalization

Past checkpoints were incorrectly trained with Transition1x isolated-atom reference E0s and normalization statistics computed with zero E0s. The checkpoint and evaluation outputs referenced above have been replaced with the corrected run.
Fitting all 1,725,362 training geometries instead gives a residual energy MAE of 0.110 eV/atom, compared with 4.259 eV/atom using the Transition1x references. 
Current HORM-T1x runs instead use:

```yaml
atomic_numbers: "[1, 6, 7, 8]"
E0s: "{1: -16.71286055092946, 6: -1035.5222423901348, 7: -1489.1419582366752, 8: -2046.2408339051071}"
mean: 0.006128199456958379
std: 0.13583519684389536
```

### Download the pretrained checkpoint

Weights: [huggingface.co/andreasburger/hip](https://huggingface.co/andreasburger/hip) (`ckpt/hip_mace.model` + `ckpt/hip_mace.yaml`).

```bash
uv sync
mkdir -p checkpoints/hip_mace
uv run hf download andreasburger/hip ckpt/hip_mace.model ckpt/hip_mace.yaml --local-dir /tmp/hip_mace_dl
cp /tmp/hip_mace_dl/ckpt/hip_mace.model /tmp/hip_mace_dl/ckpt/hip_mace.yaml checkpoints/hip_mace/
ln -sfn hip_mace.yaml checkpoints/hip_mace/config.yaml
```

Eval on HORM val (HIP predicted Hessians; use a GPU node / Slurm for anything large):

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
uv run scripts/eval_horm.py checkpoints/hip_mace \
  --max_samples 1000 --valid_file ./data/ts1x/ts1x-val --hessian_method=predict
```

ASE calculator (energy / forces):

```python
from ase import Atoms
from mace.calculators import MACECalculator

atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
atoms.calc = MACECalculator(model_paths="checkpoints/hip_mace/hip_mace.model", device="cuda")
print(atoms.get_potential_energy(), atoms.get_forces())
```

### Train on Slurm

Submit from the repository root. Pass site-specific flags (`-A`, `-p`, …) yourself:

```bash
sbatch -J horm-pairv2-512 scripts/run_with_requeue.sbatch \
  mace/cli/run_train.py --config=configs/horm_t1x_hip_pairv2_512_mae.yaml --no_restart_latest
```

### Evaluate on Slurm

```bash
sbatch -J eval-pairv2 scripts/run_with_requeue.sbatch \
  scripts/eval_horm.py checkpoints/horm_t1x_hip_pairv2_512_lr0p08_mae_e1_f10_h25_1000_wu100_cueq_bs64 \
  --max_samples 10000 --valid_file ./data/ts1x/ts1x-val --redo
```

### Training HIP-MACE

Setup the environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Test HIP-MACE
```bash
uv run tests/test_hip.py
uv run tests/test_equivariance.py
```

Train HIP-MACE to overfit a small subset of 100 datapoints
```bash
# with HIP
uv run scripts/run_train.py --config=configs/horm_100.yaml --hip=true 
# uv run scripts/run_train.py --config=configs/horm_100.yaml --hip=true --batch_size=4 --valid_batch_size=4 --eval_interval=1 --wandb=False --save_checkpoints=False

# without HIP (energy and forces)
uv run scripts/run_train.py --config=configs/horm_100.yaml
```

Get and processed the data
```bash
# download the HORM dataset
export KAGGLEHUB_CACHE=/path/to/your/preferred/directory
uv run scripts/download_horm_data_kaggle.py
# process the data
uv run scripts/convert_lmdb_to_h5.py --in_file "data/sample_100.lmdb" 
uv run scripts/convert_lmdb_to_h5.py --in_file "ts1x-val.lmdb" 
uv run scripts/convert_lmdb_to_h5.py --in_file "ts1x_hess_train.lmdb" 
# uv run scripts/convert_lmdb_to_h5.py --in_file "RGD1.lmdb" 
```

Train HIP-MACE on the full dataset
```bash
uv run scripts/run_train.py --config=configs/horm_t1x.yaml --hip=true 
```

Train regular MACE model on just energy and forces
```bash
uv run scripts/run_train.py --config=configs/horm_t1x.yaml
```

Compare HIP-MACE to autograd Hessians from regular MACE (will run automatically at the end of training)
```bash
uv run scripts/eval_horm.py
```

## License

The MACE code is published and distributed under the [MIT License](MIT.md). (Note that some of the models linked above come with different licenses).
