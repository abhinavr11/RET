# Koopman models on RET macrostates

Two-stage setup: the RET encoder stays frozen and produces z_t. On top of it we learn a lifting
g = φ(z), a linear operator K with g_{t+1} ≈ K g_t, and a readout z = C g. Then we measure how close
this linear model gets to the nonlinear predictor T, at one step and at k steps.

```
koopman/                     library
  config.py                  experiment config (YAML + key=value overrides)
  data.py                    exported z trajectories, window sampling inside documents
  forecasters.py             common predict(z0, ks) interface + mean/persistence/DMD/EDMD/MLP/RET-T/direct/Ulam
  models.py                  deep Koopman: lifting ψ, operator (dense | blockdiag | lstsq), decoder
  losses.py                  multi-step z loss, lifted linearity, spectral penalty, VICReg on ψ
  baselines.py               closed-form fits (DMD/EDMD/Ulam) and MLP training loops
  train_koopman.py           deep Koopman training loop
  metrics.py                 pooled R² (same definition as the closure probes), skill, cluster accuracy
  analysis.py, plots.py      spectrum, slow modes vs cluster groups, linearity residual, figures
scripts/koopman/
  export_z.py                Phase 1   z trajectories from a RET checkpoint
  fit_baselines.py           Phase 2
  train.py, sweep.sh         Phase 3
  eval_horizon.py            Phase 4
  koopman.sbatch             Slurm template
configs/koopman/{gptoss20b,pythia160m}.yaml
tests/test_koopman.py
```

No new dependencies: everything runs in the existing RET environment. The scripts add the repo
root to `sys.path`, so run them from the repo root. Quick check (CPU, about 20 s):

```bash
pytest tests/test_koopman.py -q
```

## 1. Export z (Phase 1)

The script loads the student encoder from the checkpoint and encodes whole documents, never
token by token. Only z (128 floats per token) is written, so the output is small.

**GPT-OSS-20B / NuminaMath, if the layer-11 hidden-state cache exists** (no LLM is loaded):

```bash
python scripts/koopman/export_z.py \
  --checkpoint outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt \
  --out-dir outputs/koopman/GptOss20B_numina_l11/z \
  --max-docs train=16000 val=500 test=1000
```

Prompt lengths are read automatically from `data/numina_math_64k_samples/GptOss20B/<split>` (the
`prompt_length` column), after a check that the cached tokens match. If that check fails, you get
a warning and the prompt lengths are 0. Prompt lengths only matter for `data.exclude_prompt=true`.

**Without the hidden-state cache** (skips the ~400 GB cache; GPT-OSS runs up to layer 11 and only z
is written). This needs the Arrow data at `data/numina_math_64k_samples/GptOss20B/{train,val,test}`:

```bash
python scripts/koopman/export_z.py \
  --checkpoint outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt \
  --data-config configs/GptOss20B/cache/numina_math_l11.yaml \
  --out-dir outputs/koopman/GptOss20B_numina_l11/z \
  --max-docs train=16000 val=500 test=1000 --batch-size 4
```

This path runs GPT-OSS forward over every token, so try `--max-docs train=2000` first to time it.

**Pythia-160M / Pile chunks** uses the checkpoint's own chunked config. It streams the Pile from
the Hugging Face Hub, so it needs network access on the node:

```bash
python scripts/koopman/export_z.py \
  --checkpoint outputs/Pythia160M/pile/ret_autonomous/checkpoint_final.pt \
  --out-dir outputs/koopman/Pythia160M_pile_l6/z \
  --max-docs train=20000 val=500 test=1000
```

For Pythia on NuminaMath instead, add `--data-config configs/Pythia160M/baselines/h_numina_math.yaml`.

**Check after export:** `<z_dir>/export_meta.json` lists the documents and tokens per split. Each
`<split>/meta.json` has `mean_length`, plus `mean_prompt_length`, which should be > 0 for NuminaMath.

## 2. Baselines (Phase 2)

```bash
python scripts/koopman/fit_baselines.py configs/koopman/gptoss20b.yaml
python scripts/koopman/fit_baselines.py configs/koopman/gptoss20b.yaml --only dmd,edmd   # subset
```

The baselines are mean, persistence, DMD, EDMD (RFF-512, RFF-2048, RBF-256 at k-means centres),
an Ulam chain on clusters, RET's own T rolled out, closure-probe MLPs (hidden 64–512; trained
one-step `H1` and through a 32-step rollout `H32`), and the direct k-step MLPs.

Each goes to `<experiment_dir>/baselines/<name>/`. Existing baselines are skipped; `--force` refits them.

## 3. Deep Koopman (Phase 3)

```bash
python scripts/koopman/train.py configs/koopman/gptoss20b.yaml                       # default: N=512, dense, linear readout
python scripts/koopman/train.py configs/koopman/gptoss20b.yaml model.lift_dim=1024 train.seed=1
python scripts/koopman/train.py configs/koopman/gptoss20b.yaml model.operator=lstsq
python scripts/koopman/train.py configs/koopman/gptoss20b.yaml model.operator=blockdiag \
    model.include_state=false model.decoder=linear_learned
python scripts/koopman/train.py configs/koopman/gptoss20b.yaml model.decoder=mlp      # nonlinear-readout ablation
```

The full grid runs sequentially. Finished runs are skipped, so you can resubmit after a timeout:

```bash
bash scripts/koopman/sweep.sh configs/koopman/gptoss20b.yaml          # N {129,256,512,1024} x H {1,8,32} x {linear_fixed, mlp} x seeds {0,1,2}
SEEDS=0 NS="256 512" HS=32 bash scripts/koopman/sweep.sh configs/koopman/gptoss20b.yaml
OPERATORS="lstsq blockdiag" SEEDS=0 bash scripts/koopman/sweep.sh configs/koopman/gptoss20b.yaml
```

Each run writes `<experiment_dir>/runs/<name>/`, containing:
- `model.pt` (the best validation checkpoint)
- `train_log.csv` (training curves)
- `val_eval.json` (validation R²(k), spectral radius)
- `spectrum.npz`

On a single GPU, expect minutes per run. Data is 128-dimensional, so the GPU is barely loaded.

## 4. Evaluation (Phase 4)

```bash
python scripts/koopman/eval_horizon.py configs/koopman/gptoss20b.yaml            # all baselines + runs, test split
python scripts/koopman/eval_horizon.py configs/koopman/gptoss20b.yaml eval.split=val --tag val
```

Results go to `<experiment_dir>/eval/<split>/`:

| file | content |
|---|---|
| `summary.md` | table: R²(1) on all pairs (the paper's closure number), R²(k) and skill(k) for k ∈ {1, 8, 32, 128} |
| `summary.csv`, `horizon_long.csv` | every metric at every k |
| `r2_vs_k.png`, `skill_vs_k.png` | horizon curves; the gap to `direct_mlp` is compounding error |
| `cluster_acc_vs_k.png`, `group_acc_vs_k.png` | behaviour-level accuracy (`group_*` only with `clusters.source=ret`) |
| `cos_vs_k.png` | cosine(pred, true); `norm_ratio` in the CSV shows drift off the LayerNorm shell |
| `eigs_<model>.png` | eigenvalues of K on the unit disc |
| `eta2_<model>.png` | how well each slow eigenfunction separates the cluster labels, next to the top PCs of z |
| `results.json` | all of the above, plus timescales, periods and the linearity residual per label |

How to read the numbers:
- **skill** = 1 − MSE / MSE_persistence. Persistence scores 0; anything below 0 is worse than copying z_t.
- **r2_onestep_allpairs** uses every (z_t, z_{t+1}) pair of the split, which is directly comparable
  to the closure R² in the paper. The horizon metrics use only starts where t+128 is still in the
  same document, so they are computed on a slightly different set of positions.

## Config notes

- **`clusters.source`**: `kmeans` (the default) fits K-means on standardized train z. After running
  `scripts/clustering/train_centers.py`, set `clusters.source: ret` and `clusters.ret_dir` to its
  output directory to use the paper's 64 clusters and 12 groups.
- **`data.exclude_prompt=true`**: only uses pairs that start inside the generation.
- **`loss.loss_space: raw`**: weights z dimensions by their raw variance, so the training loss
  matches the pooled R² that is reported.
- **`data.data_device: cuda`**: keeps the trajectories on the GPU. It is faster but uses about
  9 GB for 16k documents.

## Troubleshooting

- **Non-finite loss**: training stops and saves `nan_state.pt`. Try `train.lr=3e-4`, `loss.w_spec=10`,
  or `model.operator=blockdiag`, which is stable by construction.
- **Out of memory**:
  - In export, reduce `--batch-size`.
  - In training, reduce `train.batch_size` or `data.max_train_docs`, or set `data.in_memory=false`.
  - In evaluation, reduce `eval.batch_size` or `eval.max_starts`.
- **The RET T baseline is skipped**: set `ret_checkpoint:` in the config. By default it is read
  from `<z_dir>/export_meta.json`.

When something fails, send me the full command and the traceback (the end of the Slurm log).
