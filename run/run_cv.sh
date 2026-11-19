#!/usr/bin/env bash
# Texture-level 5-fold CV (x 10 repeats) for the proposed model and all baselines.
# Finished folds are skipped on restart, so every stage can be stopped and resumed.
# All results go to output/results_cv and logs to output/logs (output/ is git-ignored).
#
# Usage:  bash run/run_cv.sh <stage>  (or: cd run && bash run_cv.sh <stage> - both work, see the cd below)
#   compare    : all models (LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN, Simple 1D-CNN, Transformer),
#                paper setting, 5-fold CV x 10, then the comparison table
#   baseline   : transformer only, 5-fold CV x 10
#   ablation   : transformer under-fitting remedies (cumulative), 5-fold CV x 10
#                DONE (2026-10-24) - this is how Recipe B below was chosen; kept for reproducibility,
#                not meant to be rerun unless the recipe itself is being revisited.
#   all        : compare + ablation
#   mae_screen : transformer, MAE-oriented recipe (L1 + best val texture MAE + aug + cosine), 5-fold CV x 10
#                DONE (2026-10-24) - same status as 'ablation' above.
#   final      : transformer with the chosen RECIPE, 5-fold CV x 10, compared with the baselines
#   lambda_sweep : transformer ONLY, RECIPE, 5-fold CV x 2 repeats (not 10), for each --lambda_ratio in
#                LAMBDA_GRID (default "8 16 27" - see LAMBDA_GRID below); a light check of whether the
#                analytically-derived ratio (16.0) sits in a flat region or a clear local optimum exists.
#   tune_epochs : EVERY model (except LR/SVR, which have no epoch concept), RECIPE + the chosen LAMBDA,
#                5-fold CV x TUNE_REPEATS (default 2, NOT 10) - the one hyperparameter selected per model
#                (see run/run_holdout.sh's module-level note for why only this one, not the rest of RECIPE).
#                Writes output/results_cv/epochs.csv (model,best_epoch) via test/pick/pick_epochs.py, which
#                run/run_holdout.sh then reads automatically.
#                Once lambda (lambda_sweep) and per-model epoch (tune_epochs) are both picked, see
#                run/run_holdout.sh for the actual comparison: no CV, a single run per model on the full
#                Tier B+A' pool, evaluated once on the Tier A holdout.
#   fairness   : OPTIONAL supplement - same RECIPE for ANN / Hassan 1D-CNN (no aug: feature inputs)
#                and Simple 1D-CNN (with aug: same image inputs as the proposed model)
#   components : transformer component ablations with the final RECIPE, 5-fold CV x 10 each:
#                  input ablation (INPUT_VARIANTS, default "texture" = height/normal replaced by constant maps)
#                  LSFG ablation  (fusion gates frozen at equal weights 1/3)
#                then one table with the baselines, the full model and the ablations (same folds)
#
# Environment overrides
#   RECIPE        training flags of Ours' (transformer) final recipe (without --aug, which is added for
#                 image models). default: Recipe B "--loss l1 --select val_tex_mae --cosine --lr 3e-4"
#                 (selected after ablation + mae_screen, both DONE - see their notes above; not expected
#                 to change again)
#                 Recipe A (for reference): RECIPE="--select val_ccc --loss mse_ccc --lds --cosine --lr 3e-4" bash run/run_cv.sh final
#
# BASELINE RECIPE (2026-11-18, supersedes the §7 "same recipe for every model" decision - see
# claude/whc2027_apt_relabeling_strategy.md §12): $RECIPE / $AUG / $LAMBDA above apply to the transformer
# (Ours) ONLY. The other 6 models (lr, svr, ann, cnn_1d_wassem, cnn_1d_scirep, cnn_1d_simple) train with
# NO recipe overrides (cv_experiment.py's own defaults = loss mse, select val_rmse, no lds/aug/cosine -
# already "paper setting") and --lambda_ratio 1 (uniform; Tier A' upweighting is Ours' own device, not
# something the original papers ever had). The paper's new framing is "Ours = architecture + recipe,
# compared against each baseline's own original method" (decision (b)), not "same recipe, different
# architecture" - so baselines are no longer asked to run any part of Ours' recipe. cnn_1d_simple has no
# external paper, but is kept on these literal defaults too rather than growing a bespoke recipe of its own.
#
# --lambda_ratio (Tier A':Tier B loss weight, cv_experiment.py) defaults to 16.0, derived analytically from the
# participant-level rating variance (see whc2027_apt_relabeling_strategy.md, PPT 261109 slide 3/8). 'lambda_sweep'
# below is a light CV check of that value against its 90% CI bounds {8, 16, 27} - Ours only, repeats=2, no
# baseline models involved. This is distinct from the holdout comparison across all models (see run/run_holdout.sh):
# the chosen lambda from here feeds that stage, which trains every model once on the full Tier B+A' pool (no CV)
# and evaluates once on the Tier A holdout. Baselines in that stage always use --lambda_ratio 1 regardless
# of what this sweep picks for Ours (see BASELINE RECIPE note above).
#   FEATURE_MODE  original (default; Hassan et al. re-implementation, see output/review/WASEEM_CODE_REVIEW.md)
#                 | legacy (feature pipeline of the earlier results)
#   INPUT_VARIANTS  map ablations for 'components' (default "texture"; also "texture_height texture_normal")
#   LAMBDA_GRID   --lambda_ratio values for 'lambda_sweep' (default "8 16 27")
#   LAMBDA        the --lambda_ratio 'tune_epochs' trains every model with (default 16 - set this to
#                 whatever lambda_sweep actually picked, if different from the default)
#   TUNE_REPEATS  CV repeats for 'tune_epochs' (default 2, NOT 10 - see its note above)
#   MODELS, DATA_ROOT, FEAT_CACHE, OUT
set -Eeuo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, regardless of invocation cwd (this script lives in run/)
export PYTHONUNBUFFERED=1
DATA_ROOT=${DATA_ROOT:-data/trustworthy}           # Tier-based train pool root (Tier B / Tier A'; see config.yaml's data_tier_root)
FEAT_CACHE=${FEAT_CACHE:-data/features_cv}         # separate cache: the original data/features is never touched
OUT=${OUT:-output/results_cv}
LOGS=${LOGS:-output/logs/cv}
FEATURE_MODE=${FEATURE_MODE:-original}
MODELS=${MODELS:-"lr svr ann cnn_1d_wassem cnn_1d_scirep cnn_1d_simple transformer"}   # cheap -> expensive
RECIPE=${RECIPE:-"--loss l1 --select val_tex_mae --cosine --lr 3e-4"}   # final recipe = Recipe B (chosen 2026-10-24)
LAMBDA_GRID=${LAMBDA_GRID:-"8 16 27"}               # --lambda_ratio grid for 'lambda_sweep' (16.0 = current default)
LAMBDA=${LAMBDA:-16}                                # --lambda_ratio for 'tune_epochs' - set to lambda_sweep's pick
TUNE_REPEATS=${TUNE_REPEATS:-2}                     # CV repeats for 'tune_epochs' (not 10 - epoch selection only)
COMMON="--data_root ${DATA_ROOT} --feature_cache_root ${FEAT_CACHE} --feature_mode ${FEATURE_MODE} --patience 40 --amp --out ${OUT}"

stage=${1:-compare}
stage=${stage#--}                                  # accept "--all" as well as "all"
case "$stage" in
  compare|baseline|ablation|all|mae_screen|final|lambda_sweep|tune_epochs|fairness|components) ;;
  *) echo "Usage: bash run/run_cv.sh {compare|baseline|ablation|all|mae_screen|final|lambda_sweep|tune_epochs|fairness|components}"; exit 1 ;;
esac
mkdir -p "$LOGS" "$OUT"

run() { echo ">>> python -m test.cv.cv_experiment $*"; python -m test.cv.cv_experiment "$@"; }
tag() { python -m test.cv.cv_experiment $COMMON "$@" --print_tag; }   # run-folder name for a set of flags
paper_runs() {                                      # paper-setting run folders of all models for one protocol
  local p=$1 out=""                                  # --lambda_ratio 1: uniform loss, the actual paper setting
  for m in $MODELS; do out+="$(tag --model "$m" --protocol "$p" --lambda_ratio 1) "; done
  echo "$out"
}
compare() { python -m test.cv.cv_compare "$OUT" --feature_mode "$FEATURE_MODE" "$@"; }

if [[ $stage == compare || $stage == all ]]; then
  # Paper setting (MSE, Adam, best val RMSE, uniform loss) for every model; identical folds/seeds across
  # models. --lambda_ratio 1 is required here (not just the implicit cv_experiment.py default, which is
  # 16.0) - see the BASELINE RECIPE note above and cv_compare.py's "paper setting" check.
  n_models=$(wc -w <<< "$MODELS"); i=0
  for m in $MODELS; do
    i=$((i + 1))
    echo "=== [compare $i/$n_models] model=$m ==="
    run --model "$m" $COMMON --protocol cv --k 5 --repeats 10 --lambda_ratio 1 2>&1 | tee -a "$LOGS/cv_${m}.log"
  done
  compare --protocol cv --ref transformer | tee "$OUT/COMPARE_cv.md"
fi

if [[ $stage == baseline ]]; then
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 2>&1 | tee -a "$LOGS/cv_transformer.log"
fi

if [[ $stage == ablation || $stage == all ]]; then
  # DONE (2026-10-24) - this run is what selected Recipe B (see RECIPE above). Rerun only if the recipe
  # itself is being revisited; kept (not deleted) for paper reproducibility / review response.
  # under-fitting remedies for the transformer (reported in the paper): 5-fold x 10 repeats, same as all comparisons
  A="--model transformer $COMMON --protocol cv --k 5 --repeats 10"
  run $A --select val_ccc                                               2>&1 | tee -a "$LOGS/abl_select.log"
  run $A --select val_ccc --loss mse_ccc                                2>&1 | tee -a "$LOGS/abl_ccc.log"
  run $A --select val_ccc --loss mse_ccc --lds                          2>&1 | tee -a "$LOGS/abl_ccc_lds.log"
  run $A --select val_ccc --loss mse_ccc --lds --aug --cosine --lr 3e-4 2>&1 | tee -a "$LOGS/abl_full.log"
  ABL=""
  for f in "--select val_ccc" "--select val_ccc --loss mse_ccc" "--select val_ccc --loss mse_ccc --lds" \
           "--select val_ccc --loss mse_ccc --lds --aug --cosine --lr 3e-4"; do ABL+="$(tag $A $f) "; done
  compare --protocol cv --ref transformer --name compare_cv_ablation --runs $(paper_runs cv) $ABL \
    | tee "$OUT/COMPARE_cv_ablation.md"
fi

if [[ $stage == mae_screen ]]; then
  # DONE (2026-10-24) - same status as 'ablation' above: this is how the L1/val_tex_mae choice in
  # Recipe B was confirmed. Kept for reproducibility, not meant to be rerun.
  S="--model transformer $COMMON --protocol cv --k 5 --loss l1 --select val_tex_mae --aug --cosine --lr 3e-4"
  run $S --repeats 10 2>&1 | tee -a "$LOGS/mae_screen.log"
  SCREEN=$(tag $S)
  FULL=$(tag --model transformer --protocol cv --k 5 --select val_ccc --loss mse_ccc --lds --aug --cosine --lr 3e-4)
  compare --protocol cv --ref "$SCREEN" --name compare_cv_mae_screen --runs $(paper_runs cv) "$FULL" "$SCREEN" \
    | tee "$OUT/COMPARE_cv_mae_screen.md"
fi

if [[ $stage == final ]]; then
  FINAL_CV=$(tag --model transformer --protocol cv --k 5 $RECIPE --aug)
  echo "final recipe: $RECIPE --aug  ->  $FINAL_CV"
  # CV x 10 (skipped fold by fold if the same run folder already exists, e.g. from 'mae_screen' on this machine)
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 $RECIPE --aug 2>&1 | tee -a "$LOGS/final_cv.log"
  compare --protocol cv --ref "$FINAL_CV" --name compare_cv_final --runs $(paper_runs cv) "$FINAL_CV" \
    | tee "$OUT/COMPARE_cv_final.md"
fi

if [[ $stage == lambda_sweep ]]; then
  # Lambda validation only (PPT 261109 slides 4/8/9): 5-fold CV, repeats=2 (NOT 10), Ours (transformer) ONLY -
  # no other model is trained or compared here. This is a light 1-D check of the 90% CI around the
  # analytically-derived ratio (lambda*=16.3 -> grid {8,16,27}, see the RECIPE note above), not the full
  # model-comparison protocol. Every lambda in the grid, INCLUDING 16, now trains its own fold set:
  # cv_experiment.py's run-folder tag always encodes --lambda_ratio (never omitted at the 16.0 default -
  # fixed 2026-11-06, see that file's tag() comment). Before that fix, lambda=16 silently reused the
  # 'final'/'mae_screen'/'ablation' folders from before tier-weighted loss existed (pre-2026-10-30, when
  # every sample was weighted 1.0 unconditionally) as if they were real lambda_ratio=16.0 runs - they were
  # not the same experiment. If you ran lambda_sweep before this fix, delete the old un-suffixed
  # transformer_cv5..._val_tex_mae folder under $OUT so lambda=16 retrains for real under the new tag.
  RUNS=""
  PAIRS=""
  n_lam=$(wc -w <<< "$LAMBDA_GRID"); i=0
  for lam in $LAMBDA_GRID; do
    i=$((i + 1))
    echo "=== [lambda_sweep $i/$n_lam] lambda_ratio=$lam ==="
    run --model transformer $COMMON --protocol cv --k 5 --repeats 2 $RECIPE --aug --lambda_ratio "$lam" \
      2>&1 | tee -a "$LOGS/lambda_${lam}.log"
    t=$(tag --model transformer --protocol cv --k 5 $RECIPE --aug --lambda_ratio "$lam")
    RUNS+="$t "
    PAIRS+="$lam:transformer:$t "
  done
  compare --protocol cv --name compare_cv_lambda_sweep --runs $RUNS --expect_repeats 2 \
    | tee "$OUT/COMPARE_cv_lambda_sweep.md"
  # Auto-pick lambda by lowest mean tex_MAE over the sweep (see test/pick/pick_lambda.py) and write
  # $OUT/lambda.txt, so a wrapper script can do LAMBDA=$(cat "$OUT/lambda.txt") without a human
  # reading COMPARE_cv_lambda_sweep.md first.
  python -m test.pick.pick_lambda "$OUT/compare_cv_lambda_sweep/per_repeat.csv" --pairs $PAIRS --out "$OUT" \
    | tee "$OUT/LAMBDA_SELECTED.md"
fi

if [[ $stage == tune_epochs ]]; then
  # Ours (transformer): best_epoch under RECIPE + the chosen LAMBDA, criterion val_tex_mae (unchanged).
  # The 6 baselines (2026-11-18, see BASELINE RECIPE note above): NO recipe override, NO augmentation,
  # --lambda_ratio 1 - each picks its own best_epoch under its own paper-setting training (loss mse,
  # criterion val_rmse, cv_experiment.py's defaults). LR/SVR have no epoch concept and are skipped.
  # Light CV only (TUNE_REPEATS, default 2): this is hyperparameter selection, not a model comparison.
  PAIRS=""
  n_models=$(wc -w <<< "$MODELS"); i=0
  for m in $MODELS; do
    i=$((i + 1))
    [[ "$m" == "lr" || "$m" == "svr" ]] && continue
    if [[ "$m" == "transformer" ]]; then
      MRECIPE="$RECIPE"; MAUG="--aug"; MLAMBDA="$LAMBDA"
    else
      MRECIPE=""; MAUG=""; MLAMBDA="1"
    fi
    echo "=== [tune_epochs $i/$n_models] model=$m ==="
    run --model "$m" $COMMON --protocol cv --k 5 --repeats "$TUNE_REPEATS" $MRECIPE $MAUG --lambda_ratio "$MLAMBDA" \
      2>&1 | tee -a "$LOGS/tune_${m}.log"
    t=$(tag --model "$m" --protocol cv --k 5 $MRECIPE $MAUG --lambda_ratio "$MLAMBDA")
    PAIRS+="$m:$t "
  done
  python -m test.pick.pick_epochs "$OUT" --pairs $PAIRS | tee "$OUT/EPOCHS_SELECTED.md"
fi

if [[ $stage == fairness ]]; then
  # Supplementary / Discussion-only side-analysis (2026-11-18: no longer part of the main Method claim,
  # since baselines now run their own original recipe by design - see BASELINE RECIPE note above).
  # Kept only to answer "would Ours' recipe also help a baseline architecture?" if a reviewer asks;
  # NOT used for the headline comparison tables.
  FINAL_CV=$(tag --model transformer --protocol cv --k 5 $RECIPE --aug)
  FAIR=""
  for m in ann cnn_1d_wassem; do                    # feature-vector inputs: augmentation not applicable
    run --model "$m" $COMMON --protocol cv --k 5 --repeats 10 $RECIPE 2>&1 | tee -a "$LOGS/fair_${m}.log"
    FAIR+="$(tag --model "$m" --protocol cv --k 5 $RECIPE) "
  done
  # Simple 1D-CNN: same image inputs as the proposed model (texture + height + normal) -> same recipe incl. augmentation
  run --model cnn_1d_simple $COMMON --protocol cv --k 5 --repeats 10 $RECIPE --aug 2>&1 | tee -a "$LOGS/fair_cnn_1d_simple.log"
  FAIR+="$(tag --model cnn_1d_simple --protocol cv --k 5 $RECIPE --aug) "
  compare --protocol cv --ref "$FINAL_CV" --name compare_cv_fairness --runs $(paper_runs cv) "$FINAL_CV" $FAIR \
    | tee "$OUT/COMPARE_cv_fairness.md"
fi

if [[ $stage == components ]]; then
  INPUT_VARIANTS=${INPUT_VARIANTS:-texture}
  C="--model transformer $COMMON --protocol cv --k 5 --repeats 10 $RECIPE --aug"
  FINAL_CV=$(tag $C)
  RUNS=""
  for v in $INPUT_VARIANTS; do                      # map contribution: same network, maps replaced by constants
    run $C --inputs "$v" 2>&1 | tee -a "$LOGS/comp_inputs_${v}.log"
    RUNS+="$(tag $C --inputs "$v") "
  done
  run $C --no_lsfg 2>&1 | tee -a "$LOGS/comp_nolsfg.log"   # LSFG contribution: equal fixed fusion weights
  RUNS+="$(tag $C --no_lsfg) "
  compare --protocol cv --ref "$FINAL_CV" --name compare_cv_components --runs $(paper_runs cv) "$FINAL_CV" $RUNS \
    | tee "$OUT/COMPARE_cv_components.md"
fi

python -m test.cv.cv_aggregate "$OUT"/*/ | tee "$OUT/ALL_SUMMARY.md"
