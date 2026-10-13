#!/usr/bin/env bash
# Texture-level 5-fold CV / fixed-split repeats for the proposed model and all baselines.
# Finished runs are skipped on restart, so the script can be stopped and resumed.
#
# Usage:  bash run_cv.sh <stage>
#   compare   : all models (LR, SVR, ANN, Hassan 1D-CNN, Simple 1D-CNN, Transformer), 5-fold CV x 10 + fixed split x 10
#   baseline  : transformer only, 5-fold CV x 10
#   fixed     : transformer only, fixed split x 10 seeds
#   ablation  : transformer under-fitting remedies, 5-fold CV x 3
#   all       : compare + ablation
#   final     : final recipe (val_ccc + mse_ccc + LDS + aug + cosine + lr 3e-4), 5-fold CV x 10 + fixed split x 10,
#               then paired comparison of the final recipe vs the paper-setting runs (repeats 0-2 from 'ablation' are reused)
#   mae_screen: transformer, MAE-oriented recipe (L1 loss + best val texture MAE + aug + cosine), 5-fold CV x 3
#               -> compare with the paper setting and the final recipe on the same 3 repeats
#   fairness  : OPTIONAL supplement - ANN / Hassan 1D-CNN trained with the same recipe (no aug: feature inputs), CV x 10
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
DATA_ROOT=${DATA_ROOT:-data}                       # folder that contains train/ val/ test/
FEAT_CACHE=${FEAT_CACHE:-data/features_cv}         # separate cache: the original data/features is never touched
MODELS=${MODELS:-"lr svr ann cnn_1d_wassem cnn_1d_simple transformer"}   # cheap -> expensive
COMMON="--data_root ${DATA_ROOT} --feature_cache_root ${FEAT_CACHE} --patience 40 --amp --out results_cv"
stage=${1:-compare}
stage=${stage#--}                                  # accept "--all" as well as "all"
case "$stage" in
  compare|baseline|fixed|ablation|all|final|fairness|mae_screen) ;;
  *) echo "Usage: bash run_cv.sh {compare|baseline|fixed|ablation|all|final|fairness|mae_screen}"; exit 1 ;;
esac
mkdir -p logs results_cv

run() { echo ">>> python -m test.cv_experiment $*"; python -m test.cv_experiment "$@"; }

if [[ $stage == compare || $stage == all ]]; then
  # Paper setting (MSE, Adam, best val RMSE) for every model; identical folds/seeds across models.
  for m in $MODELS; do
    run --model "$m" $COMMON --protocol cv --k 5 --repeats 10 2>&1 | tee -a "logs/cv_${m}.log"
  done
  for m in $MODELS; do
    run --model "$m" $COMMON --protocol fixed --repeats 10 2>&1 | tee -a "logs/fixed_${m}.log"
  done
  python -m test.cv_compare results_cv --protocol cv    --ref transformer | tee results_cv/COMPARE_cv.md
  python -m test.cv_compare results_cv --protocol fixed --ref transformer | tee results_cv/COMPARE_fixed.md
fi

if [[ $stage == baseline ]]; then
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 2>&1 | tee -a logs/cv_transformer.log
fi

if [[ $stage == fixed ]]; then
  run --model transformer $COMMON --protocol fixed --repeats 10 2>&1 | tee -a logs/fixed_transformer.log
fi

if [[ $stage == ablation || $stage == all ]]; then
  # under-fitting remedies for the transformer, cheaper screening: 5-fold x 3 repeats
  A="--model transformer $COMMON --protocol cv --k 5 --repeats 3"
  run $A --select val_ccc                                               2>&1 | tee -a logs/abl_select.log
  run $A --select val_ccc --loss mse_ccc                                2>&1 | tee -a logs/abl_ccc.log
  run $A --select val_ccc --loss mse_ccc --lds                          2>&1 | tee -a logs/abl_ccc_lds.log
  run $A --select val_ccc --loss mse_ccc --lds --aug --cosine --lr 3e-4 2>&1 | tee -a logs/abl_full.log
  python -m test.cv_compare results_cv --protocol cv --ref transformer --include_ablations | tee results_cv/COMPARE_cv_ablation.md
fi

RECIPE="--select val_ccc --loss mse_ccc --lds --cosine --lr 3e-4"
FINAL_CV=transformer_cv5_mse_ccc_lds_aug_cos_val_ccc
FINAL_FX=transformer_fixed_mse_ccc_lds_aug_cos_val_ccc
PAPER_CV="lr_cv5_mse_val_rmse svr_cv5_mse_val_rmse ann_cv5_mse_val_rmse cnn_1d_wassem_cv5_mse_val_rmse cnn_1d_simple_cv5_mse_val_rmse transformer_cv5_mse_val_rmse"
PAPER_FX="lr_fixed_mse_val_rmse svr_fixed_mse_val_rmse ann_fixed_mse_val_rmse cnn_1d_wassem_fixed_mse_val_rmse cnn_1d_simple_fixed_mse_val_rmse transformer_fixed_mse_val_rmse"

if [[ $stage == final ]]; then
  # 1) CV x 10: repeats 0-2 already exist from the ablation stage and are skipped automatically
  run --model transformer $COMMON --protocol cv --k 5 --repeats 10 $RECIPE --aug 2>&1 | tee -a logs/final_cv.log
  python -m test.cv_compare results_cv --protocol cv --ref $FINAL_CV --runs $PAPER_CV $FINAL_CV --name compare_cv_final \
    | tee results_cv/COMPARE_cv_final.md
  # 2) fixed split x 10 seeds (original train/val/test)
  run --model transformer $COMMON --protocol fixed --repeats 10 $RECIPE --aug 2>&1 | tee -a logs/final_fixed.log
  python -m test.cv_compare results_cv --protocol fixed --ref $FINAL_FX --runs $PAPER_FX $FINAL_FX --name compare_fixed_final \
    | tee results_cv/COMPARE_fixed_final.md
fi

if [[ $stage == mae_screen ]]; then
  run --model transformer $COMMON --protocol cv --k 5 --repeats 3 --loss l1 --select val_tex_mae --aug --cosine --lr 3e-4 \
    2>&1 | tee -a logs/mae_screen.log
  python -m test.cv_compare results_cv --protocol cv --ref transformer_cv5_l1_aug_cos_val_tex_mae --name compare_cv_mae_screen \
    --runs $PAPER_CV $FINAL_CV transformer_cv5_l1_aug_cos_val_tex_mae | tee results_cv/COMPARE_cv_mae_screen.md
fi

if [[ $stage == fairness ]]; then
  # supplementary only: are the recipe gains specific to the transformer? (reference implementations stay the main baselines)
  for m in ann cnn_1d_wassem; do
    run --model "$m" $COMMON --protocol cv --k 5 --repeats 10 $RECIPE 2>&1 | tee -a "logs/fair_${m}.log"
  done
  python -m test.cv_compare results_cv --protocol cv --ref $FINAL_CV --name compare_cv_fairness \
    --runs $PAPER_CV $FINAL_CV ann_cv5_mse_ccc_lds_cos_val_ccc cnn_1d_wassem_cv5_mse_ccc_lds_cos_val_ccc \
    | tee results_cv/COMPARE_cv_fairness.md
fi

python -m test.cv_aggregate results_cv/* | tee results_cv/ALL_SUMMARY.md
