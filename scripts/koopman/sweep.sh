#!/usr/bin/env bash
# Phase-3 sweep: lift dim N x training horizon H x decoder x operator x seed, run one after another.
# Runs that already have val_eval.json are skipped, so the script can be re-submitted after a timeout.
#
#   bash scripts/koopman/sweep.sh configs/koopman/gptoss20b.yaml
#   SEEDS=1 NS="256 512" bash scripts/koopman/sweep.sh configs/koopman/gptoss20b.yaml train.steps=20000
#
# Grid (override with environment variables):
#   NS         lift dims             default "129 256 512 1024"   (129 = z + const only: DMD trained multi-step)
#   HS         final train horizon   default "1 8 32"
#   DECODERS   linear_fixed | linear_learned | mlp   default "linear_fixed mlp"
#   OPERATORS  dense | lstsq | blockdiag             default "dense"
#   SEEDS                            default "0 1 2"
# blockdiag forces model.include_state=false and swaps linear_fixed for linear_learned.
# Extra arguments are passed to every train.py call as key=value overrides.
set -euo pipefail

CFG=${1:?usage: sweep.sh <config.yaml> [key=value ...]}
shift
NS=${NS:-"129 256 512 1024"}
HS=${HS:-"1 8 32"}
DECODERS=${DECODERS:-"linear_fixed mlp"}
OPERATORS=${OPERATORS:-"dense"}
SEEDS=${SEEDS:-"0 1 2"}
PY=${PYTHON:-python}

EXP=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment_dir'])" "$CFG")
for arg in "$@"; do
  case $arg in experiment_dir=*) EXP=${arg#experiment_dir=} ;; esac
done

schedule_for() {
  case $1 in
    1) echo "[[0,1]]" ;;
    8) echo "[[0,1],[3000,8]]" ;;
    *) echo "[[0,1],[3000,8],[10000,$1]]" ;;
  esac
}

for seed in $SEEDS; do
  for N in $NS; do
    for H in $HS; do
      for dec in $DECODERS; do
        for op in $OPERATORS; do
          extra=()
          d=$dec
          if [ "$op" = "blockdiag" ]; then
            extra+=("model.include_state=false")
            [ "$d" = "linear_fixed" ] && d=linear_learned
          fi
          name="N${N}_${op}_${d}_H${H}_s${seed}"
          if [ -f "$EXP/runs/$name/val_eval.json" ]; then
            echo "[sweep] skip $name (done)"
            continue
          fi
          echo "[sweep] $(date '+%F %T') start $name"
          $PY scripts/koopman/train.py "$CFG" "run_name=$name" "model.lift_dim=$N" "model.operator=$op" \
            "model.decoder=$d" "train.seed=$seed" "loss.horizon_schedule=$(schedule_for "$H")" \
            ${extra[@]+"${extra[@]}"} "$@"
        done
      done
    done
  done
done
echo "[sweep] done"
