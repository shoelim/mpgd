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
python minimizing_widening_valley_loss.py
```
See also the Google Colab version [here](https://colab.research.google.com/drive/1FYOurx-FNrLT1O_OLx4ImSO4OPuB1z-H?usp=sharing)


### Airfoil Self-Noise regression
```
python train.py
```


### Electrocardiogram (ECG) classification
```
python ecg_classification_mlps.py
```
See also the Google Colab version [here](https://colab.research.google.com/drive/1EibFg-F8PUeS90tpYbeZNbRHEu_EnHo2?usp=sharing)


### CIFAR-10 classification
Scripts for training runs can be found in `train.sh`. Please check and specify the parameters there appropriately before running.
