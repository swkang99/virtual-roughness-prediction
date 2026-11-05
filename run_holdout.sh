#!/usr/bin/env bash
# Stage 3 of the three-stage protocol (PPT 261109_Real-Time_Roughness_Prediction_in_VR.pptx, slide 5;
# see claude/whc2027_apt_relabeling_strategy.md §7 "same recipe" principle):
#   1. run_cv.sh lambda_sweep  - Ours only, light CV, picks --lambda_ratio
#   2. run_cv.sh tune_epochs   - every model, light CV under the SAME recipe + chosen lambda, picks each
#                                model's own best_epoch (see test/pick_epochs.py)
#   3. (this script)           - every model trained ONCE on the full Tier B + Tier A' pool (98 textures,
#                                no CV / validation split - see test/holdout/holdout_eval.py), evaluated ONCE on
#                                the Tier A holdout (8 native APT textures, never used in training), then
#                                compared + significance-tested (test/holdout/holdout_compare.py)
#
# Why only epoch count differs per model, and not the rest of the recipe: loss/lr/cosine/lds/aug are
# recipe DESIGN choices, deliberately identical across models so a win can't be attributed to a baseline
# getting a worse recipe. Epoch count is different in kind - it's "when did this architecture converge",
# which is legitimately architecture-dependent - so the SAME selection procedure (stage 2's CV, criterion
# val_tex_mae) is applied to every model, and only its per-model OUTCOME (best_epoch) is allowed to vary.
# No other hyperparameter is tuned per model, to avoid reopening that fairness question and to avoid
# growing the number of CV runs (see run_cv.sh's TUNE_REPEATS note).
#
# Usage:  bash run_holdout.sh [lambda_ratio]
#   lambda_ratio defaults to 16 (the analytically-derived point estimate) - pass the value lambda_sweep
#   actually picked if it differs. Must match what tune_epochs was run with, so epochs.csv (below) lines
#   up with this lambda.
#
# Environment overrides
#   RECIPE        training flags, without --aug (added automatically for the image models transformer /
#                 cnn_1d_simple) and without --select (no validation split exists in this stage, so
#                 cv_experiment.py's early-stopping selection doesn't apply here - see holdout_eval.py)
#                 default: "--loss l1 --cosine --lr 3e-4" (Recipe B minus --select)
#   EPOCHS_CSV    per-model epoch counts from run_cv.sh's tune_epochs stage (default
#                 output/results_cv/epochs.csv, columns model,best_epoch,...). A model missing from this
#                 file (or the file itself missing) falls back to config.yaml's epochs (300) unchanged -
#                 i.e. tune_epochs hasn't been run (yet) for that model.
#   FEATURE_MODE, MODELS, DATA_ROOT, FEAT_CACHE, OUT, LOGS  - same meaning as run_cv.sh
set -Eeuo pipefail
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

for m in $MODELS; do
  AUG=""
  [[ "$m" == "transformer" || "$m" == "cnn_1d_simple" ]] && AUG="--aug"
  EP=$(epoch_flag "$m")
  [[ -z "$EP" ]] && echo "[warn] $m: no entry in $EPOCHS_CSV - falling back to config.yaml's epochs (run tune_epochs first)"
  echo ">>> python -m test.holdout.holdout_eval --model $m $RECIPE $AUG $EP --lambda_ratio $LAMBDA"
  python -m test.holdout.holdout_eval --model "$m" --data_root "$DATA_ROOT" --feature_cache_root "$FEAT_CACHE" \
    --feature_mode "$FEATURE_MODE" --out "$OUT" --amp $RECIPE $AUG $EP --lambda_ratio "$LAMBDA" \
    2>&1 | tee -a "$LOGS/holdout_${m}.log"
done

python -m test.holdout.holdout_compare "$OUT" | tee "$OUT/HOLDOUT_SUMMARY.md"
