#!/usr/bin/env bash
# Stage 3 of the three-stage protocol (PPT 261109_Real-Time_Roughness_Prediction_in_VR.pptx, slide 5;
# see claude/whc2027_apt_relabeling_strategy.md §12, which supersedes §7's "same recipe for every
# model" principle):
#   1. run/run_cv.sh lambda_sweep  - Ours only, light CV, picks --lambda_ratio
#   2. run/run_cv.sh tune_epochs   - Ours under RECIPE+LAMBDA, each baseline under its own paper-setting
#                                    recipe (--lambda_ratio 1, no RECIPE/--aug) - picks every model's own
#                                    best_epoch (see test/pick/pick_epochs.py)
#   3. (this script)           - every model trained ONCE on the full Tier B + Tier A' pool (98 textures,
#                                no CV / validation split - see test/holdout/holdout_eval.py), evaluated ONCE on
#                                the Tier A holdout (8 native APT textures, never used in training), then
#                                compared + significance-tested (test/holdout/holdout_compare.py)
#
# BASELINE RECIPE (2026-11-18, see claude/whc2027_apt_relabeling_strategy.md §12): the paper's framing is
# now "Ours = architecture + recipe" vs "each baseline's own original method" (no shared-recipe fairness
# claim). So below, RECIPE / --aug / the LAMBDA argument apply to the transformer (Ours) ONLY; the other
# 6 models train with cv_experiment.py's own defaults (loss mse, no lds/aug/cosine - already "paper
# setting") and --lambda_ratio 1 (uniform - Tier A' upweighting is Ours' own device). Epoch count still
# comes from EPOCHS_CSV per model (now selected under each model's own recipe in stage 2, not Ours').
#
# Usage:  bash run/run_holdout.sh [lambda_ratio]  (or: cd run && bash run_holdout.sh [lambda_ratio])
#   lambda_ratio (default 16, the analytically-derived point estimate for Ours - pass the value
#   lambda_sweep actually picked if it differs) is used for the transformer ONLY; every baseline always
#   runs with --lambda_ratio 1 regardless of this argument.
#
# Environment overrides
#   RECIPE        Ours' (transformer) training flags, without --aug (added automatically) and without
#                 --select (no validation split exists in this stage, so cv_experiment.py's early-stopping
#                 selection doesn't apply here - see holdout_eval.py)
#                 default: "--loss l1 --cosine --lr 3e-4" (Recipe B minus --select)
#   EPOCHS_CSV    per-model epoch counts from run/run_cv.sh's tune_epochs stage (default
#                 output/results_cv/epochs.csv, columns model,best_epoch,...). A model missing from this
#                 file (or the file itself missing) falls back to config.yaml's epochs (300) unchanged -
#                 i.e. tune_epochs hasn't been run (yet) for that model.
#   FEATURE_MODE, MODELS, DATA_ROOT, FEAT_CACHE, OUT, LOGS  - same meaning as run/run_cv.sh
set -Eeuo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, regardless of invocation cwd (this script lives in run/)
export PYTHONUNBUFFERED=1
LAMBDA=${1:-16}
DATA_ROOT=${DATA_ROOT:-data/trustworthy}
FEAT_CACHE=${FEAT_CACHE:-data/features_cv}
OUT=${OUT:-output/results_holdout}
LOGS=${LOGS:-output/logs/holdout}
FEATURE_MODE=${FEATURE_MODE:-original}
MODELS=${MODELS:-"lr svr ann cnn_1d_wassem cnn_1d_scirep cnn_1d_simple transformer"}
RECIPE=${RECIPE:-"--loss l1 --cosine --lr 3e-4"}
EPOCHS_CSV=${EPOCHS_CSV:-output/results_cv/epochs.csv}
mkdir -p "$LOGS" "$OUT"

epoch_flag() {   # $1 = model name; prints "--epochs N" if epochs.csv has one, else nothing
  [[ -f "$EPOCHS_CSV" ]] || return 0
  local e
  e=$(awk -F, -v m="$1" 'NR>1 && $1==m {print $2; found=1} END{if(!found) exit 1}' "$EPOCHS_CSV") || return 0
  [[ -n "$e" ]] && echo "--epochs $e"
}

N_MODELS=$(wc -w <<< "$MODELS")
i=0
for m in $MODELS; do
  i=$((i + 1))
  if [[ "$m" == "transformer" ]]; then
    MRECIPE="$RECIPE"; MAUG="--aug"; MLAMBDA="$LAMBDA"
  else
    MRECIPE=""; MAUG=""; MLAMBDA="1"            # baseline: own paper-setting recipe, uniform loss weight
  fi
  EP=$(epoch_flag "$m")
  [[ -z "$EP" ]] && echo "[warn] $m: no entry in $EPOCHS_CSV - falling back to config.yaml's epochs (run tune_epochs first)"
  echo ">>> [$i/$N_MODELS] python -m test.holdout.holdout_eval --model $m $MRECIPE $MAUG $EP --lambda_ratio $MLAMBDA"
  python -m test.holdout.holdout_eval --model "$m" --data_root "$DATA_ROOT" --feature_cache_root "$FEAT_CACHE" \
    --feature_mode "$FEATURE_MODE" --out "$OUT" --amp $MRECIPE $MAUG $EP --lambda_ratio "$MLAMBDA" \
    2>&1 | tee -a "$LOGS/holdout_${m}.log"
done

python -m test.holdout.holdout_compare "$OUT" | tee "$OUT/HOLDOUT_SUMMARY.md"
