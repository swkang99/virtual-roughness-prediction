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
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
DATA_ROOT=${DATA_ROOT:-data}                       # folder that contains train/ val/ test/
FEAT_CACHE=${FEAT_CACHE:-data/features_cv}         # separate cache: the original data/features is never touched
MODELS=${MODELS:-"lr svr ann cnn_1d_wassem cnn_1d_simple transformer"}   # cheap -> expensive
COMMON="--data_root ${DATA_ROOT} --feature_cache_root ${FEAT_CACHE} --patience 40 --amp --out results_cv"
stage=${1:-compare}
mkdir -p logs

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

python -m test.cv_aggregate results_cv/* | tee results_cv/ALL_SUMMARY.md
