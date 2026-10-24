#!/usr/bin/env bash
# Texture-level 5-fold CV / fixed-split repeats for the proposed model and all baselines.
# Finished folds are skipped on restart, so every stage can be stopped and resumed.
# All results go to output/results_cv and logs to output/logs (output/ is git-ignored).
#
# Usage:  bash run_cv.sh <stage>
#   compare    : all models (LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN, Simple 1D-CNN, Transformer),
#                paper setting, 5-fold CV x 10 + fixed split x 10, then comparison tables
#   baseline   : transformer only, 5-fold CV x 10
#   fixed      : transformer only, fixed split x 10 seeds
#   ablation   : transformer under-fitting remedies (cumulative), 5-fold CV x 10
#   all        : compare + ablation
#   mae_screen : transformer, MAE-oriented recipe (L1 + best val texture MAE + aug + cosine), 5-fold CV x 10
#   final      : transformer with the chosen RECIPE, 5-fold CV x 10 + fixed split x 10, compared with the baselines
#   fairness   : OPTIONAL supplement - same RECIPE for ANN / Hassan 1D-CNN (no aug: feature inputs)
#                and Simple 1D-CNN (with aug: same image inputs as the proposed model)
#   components : transformer component ablations with the final RECIPE, 5-fold CV x 10 each:
#                  input ablation (INPUT_VARIANTS, default "texture" = height/normal replaced by constant maps)
#                  LSFG ablation  (fusion gates frozen at equal weights 1/3)
#                then one table with the baselines, the full model and the ablations (same folds)
#
# Environment overrides
#   RECIPE        training flags of the final recipe (without --aug, which is added for image models)
#                 default: "--select val_ccc --loss mse_ccc --lds --cosine --lr 3e-4"
#                 L1 recipe: RECIPE="--loss l1 --select val_tex_mae --cosine --lr 3e-4" bash run_cv.sh final
#   FEATURE_MODE  original (default; Hassan et al. re-implementation, see output/review/WASEEM_CODE_REVIEW.md)
#                 | legacy (feature pipeline of the earlier results)
#   INPUT_VARIANTS  map ablations for 'components' (default "texture"; also "texture_height texture_normal")
#   MODELS, DATA_ROOT, FEAT_CACHE, OUT
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
DATA_ROOT=${DATA_ROOT:-data}                       # folder that contains train/ val/ test/
FEAT_CACHE=${FEAT_CACHE:-data/features_cv}         # separate cache: the original data/features is never touched
OUT=${OUT:-output/results_cv}
LOGS=${LOGS:-output/logs/cv}
FEATURE_MODE=${FEATURE_MODE:-original}
MODELS=${MODELS:-"lr svr ann cnn_1d_wassem cnn_1d_scirep cnn_1d_simple transformer"}   # cheap -> expensive
RECIPE=${RECIPE:-"--select val_ccc --loss mse_ccc --lds --cosine --lr 3e-4"}
COMMON="--data_root ${DATA_ROOT} --feature_cache_root ${FEAT_CACHE} --feature_mode ${FEATURE_MODE} --patience 40 --amp --out ${OUT}"

stage=${1:-compare}
stage=${stage#--}                                  # accept "--all" as well as "all"
case "$stage" in
  compare|baseline|fixed|ablation|all|mae_screen|final|fairness|components) ;;
  *) echo "Usage: bash run_cv.sh {compare|baseline|fixed|ablation|all|mae_screen|final|fairness|components}"; exit 1 ;;
esac
mkdir -p "$LOGS" "$OUT"

run() { echo ">>> python -m test.cv_experiment $*"; python -m test.cv_experiment "$@"; }
tag() { python -m test.cv_experiment $COMMON "$@" --print_tag; }   # run-folder name for a set of flags
paper_runs() {                                      # paper-setting run folders of all models for one protocol
  local p=$1 out=""
  for m in $MODELS; do out+="$(tag --model "$m" --protocol "$p") "; done
  echo "$out"
}
compare() { python -m test.cv_compare "$OUT" --feature_mode "$FEATURE_MODE" "$@"; }

if [[ $stage == compare || $stage == all ]]; then
  # Paper setting (MSE, Adam, best val RMSE) for every model; identical folds/seeds across models.
  for m in $MODELS; do
    run --model "$m" $COMMON --protocol cv --k 5 --repeats 10 2>&1 | tee -a "$LOGS/cv_${m}.log"
  done
  for m in $MODELS; do
    run --model "$m" $COMMON --protocol fixed --repeats 10 2>&1 | tee -a "$LOGS/fixed_${m}.log"
  done
  compare --protocol cv    --ref transformer | tee "$OUT/COMPARE_cv.md"
  compare --protocol fixed --ref transformer | tee "$OUT/COMPARE_fixed.md"
fi

if [[ $stage == baseline ]]; then
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 2>&1 | tee -a "$LOGS/cv_transformer.log"
fi

if [[ $stage == fixed ]]; then
  run --model transformer $COMMON --protocol fixed --repeats 10 2>&1 | tee -a "$LOGS/fixed_transformer.log"
fi

if [[ $stage == ablation || $stage == all ]]; then
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
  S="--model transformer $COMMON --protocol cv --k 5 --loss l1 --select val_tex_mae --aug --cosine --lr 3e-4"
  run $S --repeats 10 2>&1 | tee -a "$LOGS/mae_screen.log"
  SCREEN=$(tag $S)
  FULL=$(tag --model transformer --protocol cv --k 5 --select val_ccc --loss mse_ccc --lds --aug --cosine --lr 3e-4)
  compare --protocol cv --ref "$SCREEN" --name compare_cv_mae_screen --runs $(paper_runs cv) "$FULL" "$SCREEN" \
    | tee "$OUT/COMPARE_cv_mae_screen.md"
fi

if [[ $stage == final ]]; then
  FINAL_CV=$(tag --model transformer --protocol cv --k 5 $RECIPE --aug)
  FINAL_FX=$(tag --model transformer --protocol fixed $RECIPE --aug)
  echo "final recipe: $RECIPE --aug  ->  $FINAL_CV / $FINAL_FX"
  # 1) CV x 10 (already finished by 'ablation' (default recipe) or 'mae_screen' (L1 recipe) -> skipped)
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 $RECIPE --aug 2>&1 | tee -a "$LOGS/final_cv.log"
  compare --protocol cv --ref "$FINAL_CV" --name compare_cv_final --runs $(paper_runs cv) "$FINAL_CV" \
    | tee "$OUT/COMPARE_cv_final.md"
  # 2) fixed split x 10 seeds (original train/val/test)
  run --model transformer $COMMON --protocol fixed --repeats 10 $RECIPE --aug 2>&1 | tee -a "$LOGS/final_fixed.log"
  compare --protocol fixed --ref "$FINAL_FX" --name compare_fixed_final --runs $(paper_runs fixed) "$FINAL_FX" \
    | tee "$OUT/COMPARE_fixed_final.md"
fi

if [[ $stage == fairness ]]; then
  # supplementary only: are the recipe gains specific to the transformer? (reference implementations stay the main baselines)
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

python -m test.cv_aggregate "$OUT"/*/ | tee "$OUT/ALL_SUMMARY.md"
