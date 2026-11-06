#!/usr/bin/env bash
# Runs the full three-stage protocol back-to-back, unattended (PPT 261109_Real-Time_Roughness_
# Prediction_in_VR.pptx slide 5; see claude/whc2027_apt_relabeling_strategy.md §7):
#   1. run/run_cv.sh lambda_sweep  - Ours only, light CV, auto-picks --lambda_ratio by lowest mean tex_MAE
#                                    over {8,16,27} (see test/pick/pick_lambda.py) -> output/results_cv/lambda.txt
#   2. run/run_cv.sh tune_epochs   - every model, light CV under that SAME lambda, picks each model's own
#                                    best_epoch (see test/pick/pick_epochs.py) -> output/results_cv/epochs.csv
#   3. run/run_holdout.sh          - every model trained ONCE on the full Tier B + Tier A' pool, evaluated
#                                    ONCE on the Tier A holdout, compared + significance-tested
#
# Each stage is a plain call to the scripts you'd run by hand - this just chains them and carries the
# auto-picked lambda from stage 1 into stages 2 and 3 instead of a human reading LAMBDA_SELECTED.md and
# retyping it. Finished CV folds are skipped on restart (run_cv.sh's own behavior), so re-running this
# script after a crash does not redo completed work - except holdout runs, which are single-shot per
# model and will retrain a model if run_holdout.sh is re-entered for it (see that script).
#
# Usage (from repository root, or from anywhere - this script cd's to the repo root itself):
#   bash run/run_all.sh                                   # foreground - stops if your SSH session drops
#   nohup bash run/run_all.sh > output/logs/run_all.log 2>&1 &   # background - survives SSH disconnect
#   disown                                            # (optional) detach from this shell entirely
#   tail -f output/logs/run_all.log                   # watch progress later, from any session
# or run it inside `tmux`/`screen` and just leave the pane running - either approach works; nohup is
# simplest if you don't already use tmux/screen on this machine.
#
# Environment overrides: anything run/run_cv.sh / run/run_holdout.sh accept (RECIPE, MODELS, DATA_ROOT, OUT,
# FEAT_CACHE, LOGS, FEATURE_MODE, LAMBDA_GRID, TUNE_REPEATS, EPOCHS_CSV, ...) - set OUT before calling
# this script if you override run_cv.sh's output dir, since this script reads $OUT/lambda.txt and
# $OUT/epochs.csv to chain stages 1->2->3.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # run/ - where run_cv.sh / run_holdout.sh also live
cd "$SCRIPT_DIR/.."                                          # repo root, regardless of invocation cwd
OUT=${OUT:-output/results_cv}

echo "=== [1/3] $(date '+%F %T') run_cv.sh lambda_sweep ==="
bash "$SCRIPT_DIR/run_cv.sh" lambda_sweep

LAMBDA_FILE="$OUT/lambda.txt"
[[ -f "$LAMBDA_FILE" ]] || { echo "ERROR: $LAMBDA_FILE missing - lambda_sweep did not complete (see output above)"; exit 1; }
LAMBDA=$(cat "$LAMBDA_FILE")
echo "=== lambda_sweep picked LAMBDA=$LAMBDA (see $OUT/LAMBDA_SELECTED.md for the full comparison) ==="

echo "=== [2/3] $(date '+%F %T') run_cv.sh tune_epochs (LAMBDA=$LAMBDA) ==="
LAMBDA=$LAMBDA bash "$SCRIPT_DIR/run_cv.sh" tune_epochs

EPOCHS_FILE="$OUT/epochs.csv"
[[ -f "$EPOCHS_FILE" ]] || { echo "ERROR: $EPOCHS_FILE missing - tune_epochs did not complete (see output above)"; exit 1; }
# run_holdout.sh's own EPOCHS_CSV default ("output/results_cv/epochs.csv") only matches this script's
# $OUT by coincidence of both scripts' defaults - pass it explicitly so overriding $OUT above (per the
# env-override note up top) can't silently send stage 3 back to config.yaml's un-tuned epoch count.
echo "=== [3/3] $(date '+%F %T') run_holdout.sh $LAMBDA (EPOCHS_CSV=$EPOCHS_FILE) ==="
EPOCHS_CSV="$EPOCHS_FILE" bash "$SCRIPT_DIR/run_holdout.sh" "$LAMBDA"

echo "=== DONE $(date '+%F %T') - see output/results_holdout/HOLDOUT_SUMMARY.md ==="
