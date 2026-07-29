# HORM t1x HIP cueq eval (512 / bs64 / 1000ep), 10k val samples

HIP predicted Hessians on `./data/ts1x/ts1x-val` with `--max_samples 10000 --redo`.
Jobs: **30748307** (eqv2), **30748308** (message), **30748309** (pair). All COMPLETED ~17 min.

Metrics are means over 10k samples. Errors in eV / eV·Å⁻¹ / eV·Å⁻² (×1000 → meV).
Eigval / eigvec metrics use Eckart (mass-weighted) projection.
`time` is CUDA-event forward time (E/F + HIP Hessian), ms/sample.

## Results (cueq, 1000 epochs)

| Head | E MAE | F MAE | H MAE | eigval MAE | 1st eigval MAE | 1st eigvec cos mean | 1st eigvec cos median | TS agree | fwd ms |
|------|------:|------:|------:|-----------:|---------------:|-------------------:|---------------------:|---------:|-------:|
| pair_v2 | 0.0292 | 0.0340 | 0.0704 | 0.0387 | 0.1411 | 0.9060 | 0.9884 | 0.810 | 77.6 |
| message_v1 | 0.0306 | 0.0350 | 0.0715 | 0.0396 | 0.1411 | 0.9033 | 0.9881 | 0.805 | 81.0 |
| eqv2_v1 | 0.0350 | 0.0376 | 0.0745 | 0.0414 | 0.1526 | 0.9008 | 0.9864 | 0.796 | 82.5 |

Same table in meV:

| Head | E (meV) | F (meV/Å) | H (meV/Å²) |
|------|--------:|----------:|-----------:|
| pair_v2 | 29.2 | 34.0 | 70.4 |
| message_v1 | 30.6 | 35.0 | 71.5 |
| eqv2_v1 | 35.0 | 37.6 | 74.5 |

## vs prior non-cueq 500ep (where available)

| Head | ΔE | ΔF | ΔH | Δ eigval MAE | Δ 1st eigvec cos | Δ TS agree |
|------|---:|---:|---:|-------------:|-----------------:|-----------:|
| message_v1 | −0.0039 | −0.0013 | −0.0025 | −0.0026 | +0.0032 | +0.014 |
| eqv2_v1 | +0.0020 | +0.0015 | +0.0028 | +0.0004 | −0.0003 | −0.008 |

## CSVs (local, gitignored under `results_evalhorm/`)

- `results_evalhorm/horm_t1x_hip_pairv2_512_lr0p08_mae_e1_f10_h25_1000_wu100_cueq_bs64_run-42_ts1x-val_predict_metrics.csv`
- `results_evalhorm/horm_t1x_hip_message_v1_512_lr0p08_mae_e1_f10_h25_1000_wu100_cueq_bs64_run-42_ts1x-val_predict_metrics.csv`
- `results_evalhorm/horm_t1x_hip_eqv2_v1_512_lr0p08_mae_e1_f10_h25_1000_wu100_cueq_bs64_run-42_ts1x-val_predict_metrics.csv`
