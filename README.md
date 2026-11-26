# Real-Time Roughness Prediction in Virtual Reality Using Transformer with Texture, Height, and Normal Maps

Paper for predicting **roughness** from texture-related visual inputs.

Venue: WHC 2027

## Input and Output

- **Texture image & PBR maps** -> Roughness

## Setup

### Python

A virtual environment satisfying `python>=3.11` is required.

### Dependencies

All dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

This project was developed under the following environment:

- GPU: RTX 5080  
- torch: 2.8.0.dev20250415+cu128  
- torchvision: 0.22.0.dev20250416+cu128  

Please install the appropriate versions based on your system configuration, referring to the environment above.

### Data
> Kim, Donghyeon, Chungman Lim, and Gunhyuk Park. "APT: Artificial Pin Twirler for Rendering Rough–Smooth Textures Using Pin Height Modulation in VR." International Journal of Human–Computer Interaction (2026): 1-25. https://doi.org/10.1080/10447318.2026.2717499

> Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6


### Configuration

The configuration values required to reproduce the model are stored in `config.yaml`.

## Reproduction

### Models compared
**Ours** (transformer) plus 6 baselines 

— LR, SVR, ANN,
Hassan 1D-CNN (`cnn_1d_wassem`), Scirep 1D-CNN (`cnn_1d_scirep`, Taye et
al. 2020), and Simple 1D-CNN (no external citation, an internal
lightweight comparator) — 7 models total.

Training and evaluation run through the CV + holdout protocol
(`test/cv/cv_experiment.py` and `test/holdout/holdout_eval.py`, driven by
the scripts under `run/`); the data lives under `data/trustworthy` (see
`data_tier_root` in `config.yaml`) rather than the old `data/train` /
`data/val` / `data/test` split folders.

### Per-model recipe branching

Only **Ours (transformer)** uses the proposed recipe end to end —
`--loss l1 --select val_tex_mae --cosine --lr 3e-4 --aug` plus the locked
`--lambda_ratio 27` . Every baseline model is trained
with its own original-paper recipe instead (this repo's un-flagged
defaults: `loss=mse`, `select=val_rmse`, no `lds`/`aug`/`cosine`,
`feature_mode=original` for the Hassan-style features) and
`--lambda_ratio 1` (no Tier A' up-weighting — that device is specific to
Ours, not part of any baseline's original method). `run/run_cv.sh` and
`run/run_holdout.sh` apply this branching automatically per model; you
only need to pass `LAMBDA=27` (or `bash run/run_holdout.sh 27`) and the
scripts route it to the transformer alone.

### Stages, in order

```bash
bash run/run_cv.sh lambda_sweep               # Ours only, light CV -> picks --lambda_ratio (locked: 27)
LAMBDA=27 bash run/run_cv.sh tune_epochs      # every model, light CV -> output/results_cv/epochs.csv
bash run/run_holdout.sh 27                    # role 2: single full-pool model per model, Tier A score for reference
LAMBDA=27 bash run/run_cv.sh confirm          # role 1: final locked-recipe 5-fold x R-repeats CV, --eval_tier_a
python -m test.cv.pre_confirm_diagnostics output/results_cv --holdout_root output/results_holdout
python -m test.cv.tier_a_confirm output/results_cv --holdout_root output/results_holdout
```

`run/run_all.sh` chains only the first three of these stages (the
pre-2026-11-25 three-stage protocol); it predates the §18 evaluation
redesign below and does not run `confirm` / the diagnostics / the
confirmatory stats script, so for the paper's headline numbers run the
full stage list above by hand (or via one background batch — see
`nohup bash run/run_all.sh > output/logs/run_all.log 2>&1 &` for the
pattern, extended with the `confirm` and analysis commands).

### Evaluation protocol

Once every model's recipe is 100% locked, the Tier A (8 native, never
trained-on APT textures) confirmatory evaluation reuses the final CV's
fold models instead of a single one-shot full-pool evaluation, so it can
report mean ± SD instead of one point estimate:

- **Role 1 (confirmatory statistics)** — `run/run_cv.sh confirm`'s 5-fold
  x R-repeats CV fold models, each evaluated on Tier A via
  `cv_experiment.py --eval_tier_a`; `test/cv/tier_a_confirm.py` aggregates
  this into the primary pre-registered test (Ours vs. Simple 1D-CNN,
  Wilcoxon signed-rank, uncorrected) and the secondary test (all 7 models,
  Friedman omnibus + Nemenyi post-hoc, gated on omnibus significance).
- **Role 2 (deployment / User Study)** — `run/run_holdout.sh`'s single
  model trained once on the full Tier B + Tier A' pool; this is the model
  actually used in User Study #1-1/#2, with its own Tier A score reported
  alongside Role 1's only for consistency, not as primary evidence.
- **`test/cv/pre_confirm_diagnostics.py`** (§10.3) runs sanity checks that
  never look at Tier A — CV repeat-to-repeat stability, best_epoch
  selection stability, and CV-vs-holdout rank agreement — and should be
  reviewed *before* reading `tier_a_confirm.py`'s output.

`run/run_cv.sh` also has older stages (`compare`, `baseline`, `ablation`,
`all`, `mae_screen`, `final`, `fairness`, `components`) kept for
reproducibility — see the comment header in `run/run_cv.sh` for what each
stage runs. Results are written to `output/results_cv` / `output/results_holdout`
and logs to `output/logs` (all git-ignored).


## Reference

In this project, performance was compared with the following studies.

> [1] Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6

> [2] Taye, G. T., Hwang, H. J., & Lim, K. M. (2020). *Application of a convolutional neural network for predicting the occurrence of ventricular tachyarrhythmia using heart rate variability features*. Scientific Reports, 10, 6769. https://doi.org/10.1038/s41598-020-63566-8