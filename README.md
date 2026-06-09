# virtual-tactile-property

A computer vision project for predicting **roughness** from texture-related visual inputs.

## Input and Output

- **Texture image** -> Roughness
- **Texture PBR maps** -> Roughness

## Setup

### Python

A virtual environment satisfying `python>=3.9` is required.

### Dependencies

All dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

### Configuration

The configuration values required to reproduce the model are stored in `config.yaml`.

## Reproduction

Run `demo.ipynb` to reproduce the reported results.
Execute all cells in order, or run the entire notebook at once.

> Note: `reproduction_1d_cnn.ipynb` is **not** related to this project.

## Use of AI / Coding Agents

This project used **Copilot Free** and **Perplexity** during debugging and refactoring.

## Baseline

In this project, performance was compared with the following studies.

> Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6

> Taye, G. T., Hwang, H. J., & Lim, K. M. (2020). *Application of a convolutional neural network for predicting the occurrence of ventricular tachyarrhythmia using heart rate variability features*. Scientific Reports, 10, 6769. https://doi.org/10.1038/s41598-020-63566-8