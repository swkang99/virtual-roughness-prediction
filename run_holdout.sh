#!/usr/bin/env bash
# Stage 2 of the two-stage protocol (PPT 261109_Real-Time_Roughness_Prediction_in_VR.pptx, slide 5):
# train every model ONCE on the full Tier B + Tier A' pool (98 textures, no CV / validation split - see
# test/holdout_eval.py), evaluate ONCE on the Tier A holdout (8 native APT textures, never used in
# training), then build the comparison table + significance tests (test/holdout_compare.py).
#
# lambda is fixed BEFORE this script runs, by run_cv.sh's lambda_sweep stage (5-fold CV, Ours only).
# Every model here uses that same lambda ("동일 레시피" - strategy doc §7).
#
# Usage:  bash run_holdout.sh [lambda_ratio]
#   lambda_ratio defaults to 16 (the analytically-derived point estimate) - pass the value lambda_sweep
#   actually picked if it differs.
#
# Environment overrides
#   RECIPE        training flags, without --aug (added automatically for the image models transformer /
#                 cnn_1d_simple) and without --select (no validation split exists in this stage, so
#                 cv_experiment.py's early-stopping selection doesn't apply here - see holdout_eval.py)
#                 default: "--loss l1 --cosine --lr 3e-4" (Recipe B minus --select)
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
mkdir -p "$LOGS" "$OUT"

for m in $MODELS; do
  AUG=""
  [[ "$m" == "transformer" || "$m" == "cnn_1d_simple" ]] && AUG="--aug"
  echo ">>> python -m test.holdout_eval --model $m $RECIPE $AUG --lambda_ratio $LAMBDA"
  python -m test.holdout_eval --model "$m" --data_root "$DATA_ROOT" --feature_cache_root "$FEAT_CACHE" \
    --feature_mode "$FEATURE_MODE" --out "$OUT" --amp $RECIPE $AUG --lambda_ratio "$LAMBDA" \
    2>&1 | tee -a "$LOGS/holdout_${m}.log"
done

python -m test.holdout_compare "$OUT" | tee "$OUT/HOLDOUT_SUMMARY.md"
