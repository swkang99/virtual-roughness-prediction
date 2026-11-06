# Real-Time Roughness Prediction in Virtual Reality Using Transformer with Texture, Height, and Normal Maps

Paper for predicting **roughness** from texture-related visual inputs.

Venue: WHC 2027

## Input and Output

- **Texture image & PBR maps** -> Roughness

## Setup

### Python

A virtual environment satisfying `python>=3.9` is required.

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


The `requirements.txt` file does not specify versions for `torch` and `torchvision`.  
Please install the appropriate versions based on your system configuration, referring to the environment above.

### Data
> Kim, Donghyeon, Chungman Lim, and Gunhyuk Park. "APT: Artificial Pin Twirler for Rendering Rough–Smooth Textures Using Pin Height Modulation in VR." International Journal of Human–Computer Interaction (2026): 1-25. https://doi.org/10.1080/10447318.2026.2717499

> Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6


### Configuration

The configuration values required to reproduce the model are stored in `config.yaml`.

## Reproduction

### Run bash script
Training and evaluation now run through the three-stage CV + holdout
protocol (`test/cv/cv_experiment.py` and `test/holdout/holdout_eval.py`,
driven by the scripts under `run/`); the data lives under
`data/trustworthy` (see `data_tier_root` in `config.yaml`) rather than the
old `data/train` / `data/val` / `data/test` split folders.

To run the full protocol (lambda selection -> per-model epoch selection ->
holdout comparison) in one go:

```bash
bash run/run_all.sh
# or, to survive an SSH disconnect:
nohup bash run/run_all.sh > output/logs/run_all.log 2>&1 &
```

Or run each stage by hand:

```bash
bash run/run_cv.sh lambda_sweep    # picks --lambda_ratio, writes output/results_cv/lambda.txt
bash run/run_cv.sh tune_epochs     # picks each model's best_epoch, writes output/results_cv/epochs.csv
bash run/run_holdout.sh            # single train+eval per model, writes output/results_holdout
```

`run/run_cv.sh` also has older stages (`compare`, `baseline`, `ablation`,
`all`, `mae_screen`, `final`, `fairness`, `components`) kept for
reproducibility — see the comment header in `run/run_cv.sh` for what each
stage runs. Results are written to `output/results_cv` / `output/results_holdout`
and logs to `output/logs` (all git-ignored).


## Reference

In this project, performance was compared with the following studies.

> [1] Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6

> [2] Taye, G. T., Hwang, H. J., & Lim, K. M. (2020). *Application of a convolutional neural network for predicting the occurrence of ventricular tachyarrhythmia using heart rate variability features*. Scientific Reports, 10, 6769. https://doi.org/10.1038/s41598-020-63566-8