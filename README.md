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

This project was developed under the following environment:

- GPU: RTX 5080  
- torch: 2.8.0.dev20250415+cu128  
- torchvision: 0.22.0.dev20250416+cu128  


The `requirements.txt` file does not specify versions for `torch` and `torchvision`.  
Please install the appropriate versions based on your system configuration, referring to the environment above.

### Data
> Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6


### Configuration

The configuration values required to reproduce the model are stored in `config.yaml`.

## Reproduction

### demo.ipynb
The data can be downloaded from the supplementary materials of study [1] and should be placed in the data/original directory.

Run `demo.ipynb` to reproduce the reported results.
Execute all cells in order, or run the entire notebook at once.

### reproduction_1d_cnn.ipynb
`reproduction_1d_cnn.ipynb` is reproduction of study [1], prediction of 4 haptic attributes same as the paper. 
The feature data and label were taken from the respective .csv files. 
For label data, since `ParticipantData.csv` is stored in the range of -50 to 50, we add 50 to all values when loading the data to convert them to the range of 0 to 100. 
Once the virtual environment have set up, the training can reproduce by running all the cells.


## Reference

In this project, performance was compared with the following studies.

> [1] Hassan, W., Joolee, J. B., & Jeon, S. (2023). *Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN*. Scientific Reports, 13, 11684. https://doi.org/10.1038/s41598-023-38929-6

> [2] Taye, G. T., Hwang, H. J., & Lim, K. M. (2020). *Application of a convolutional neural network for predicting the occurrence of ventricular tachyarrhythmia using heart rate variability features*. Scientific Reports, 10, 6769. https://doi.org/10.1038/s41598-020-63566-8