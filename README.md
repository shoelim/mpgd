# MPGD: Chaotic Regularization and Heavy-Tailed Limits for Deterministic Gradient Descent
This repository implements multiscale perturbed gradient descent (MPGD) and provides codes to reproduce the results of the paper. Please refer to the paper for an introduction to the optimization tasks and other details.

## Requirements
- python 3
- pyTorch 1.9.* 
- hydra 1.* (via pip install hydra-core --upgrade)
- sklearn
- numpy
- scipy 
- pandas
- math
- statistics

## Instructions and Usage

### Minimizing the widening valley loss
```
python wideningvalleyloss_minimization/minimizing_widening_valley_loss.py
```
See also the Jupyter notebook version in the folder `wideningvalleyloss_minimization` 

### Airfoil Self-Noise regression
python airfoilselfnoise_regression/train.py

### Electrocardiogram (ECG) classification
```
python ecg5000_classification/ecg_classification_mlps.py
```
See also the Jupyter notebook version in the folder `ecg5000_classification`

### CIFAR-10 classification
```
bash cifar10_classification/train.sh 
```
Scripts for training runs can be found in `train.sh`: please check and specify the parameters there appropriately before running
