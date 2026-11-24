#!/bin/bash
# Slurm template for the Koopman pipeline.
#
# Before the first use: copy the #SBATCH account/partition lines and the environment setup
# (module load / conda activate) from the job script you already use for RET, and run
#   mkdir -p logs
#
#   sbatch scripts/koopman/koopman.sbatch export      # Phase 1 (GPU; loads GPT-OSS only in document mode)
#   sbatch scripts/koopman/koopman.sbatch baselines   # Phase 2
#   sbatch --array=0-2 scripts/koopman/koopman.sbatch sweep   # Phase 3, one seed per array task
#   sbatch scripts/koopman/koopman.sbatch eval        # Phase 4
#
# Variables (export them or pass with sbatch --export=ALL,CFG=...):
#   CFG          Koopman config                (default configs/koopman/gptoss20b.yaml)
#   CKPT         RET checkpoint for export     (default: the GPT-OSS L11 run)
#   ZDIR         export output dir             (default: data.z_dir of CFG)
#   MAXDOCS      export doc counts             (default "train=16000 val=500 test=1000")
#   EXPORT_ARGS  extra export_z.py arguments, e.g.
#                "--data-config configs/GptOss20B/cache/numina_math_l11.yaml"  (no hidden-state cache)
#   SWEEP_ENV    e.g. "NS='256 512' HS=32"     (grid overrides for sweep.sh)
#
#SBATCH --job-name=koopman
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# --- environment (copy from your RET job script) ---
# module load ...
# source activate ret
export PYTHONUNBUFFERED=1

STEP=${1:-}
CFG=${CFG:-configs/koopman/gptoss20b.yaml}
CKPT=${CKPT:-outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt}
ZDIR=${ZDIR:-$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['z_dir'])" "$CFG")}
MAXDOCS=${MAXDOCS:-"train=16000 val=500 test=1000"}
EXPORT_ARGS=${EXPORT_ARGS:-}

echo "[koopman] step=$STEP cfg=$CFG host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

case "$STEP" in
  export)
    # shellcheck disable=SC2086
    python scripts/koopman/export_z.py --checkpoint "$CKPT" --out-dir "$ZDIR" --max-docs $MAXDOCS $EXPORT_ARGS
    ;;
  baselines)
    python scripts/koopman/fit_baselines.py "$CFG"
    ;;
  train)
    python scripts/koopman/train.py "$CFG" "${@:2}"
    ;;
  sweep)
    SEED=${SLURM_ARRAY_TASK_ID:-0}
    eval "SEEDS=$SEED ${SWEEP_ENV:-} bash scripts/koopman/sweep.sh \"$CFG\" ${*:2}"
    ;;
  eval)
    python scripts/koopman/eval_horizon.py "$CFG" "${@:2}"
    ;;
  *)
    echo "usage: sbatch scripts/koopman/koopman.sbatch {export|baselines|train|sweep|eval} [key=value ...]"
    exit 1
    ;;
esac
echo "[koopman] done $(date)"
