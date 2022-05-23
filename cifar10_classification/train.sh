'''ResNet-18 trained on CIFAR-10'''

# Codes are built on https://github.com/jonasgeiping/fullbatchtraining (please check the requirements there)
# Default parameters are specified in *config/hyp/_default_hyperparams.yaml*

## Baseline:
### please specify the seed for model runs in *config/cfg.yaml* (here we choose seed 1,2,3,4,5)
python train_with_gradient_descent.py name=fbaug_noclip hyp=fb1 

## GD with Gaussian perturbations:
### please specify the seed for model runs in *config/cfg.yaml* (here we choose seed 1,2,3,4,5)
### please specify the perturbation levels (mu, sigma) and the seed for the perturbation schemes in *config/hyp/fb_gaussian.yaml* (here we choose seed 1,2,3,4,5)
python train_with_gradient_descent.py name=fbaug_gaussian hyp=fb_gaussian

## MPGD:
### please specify the scheme: *perturb* or *diffperturb* in *config/hyp/fb_ours.yaml*
### please specify the perturbation levels (mu, sigma) and the seed for the perturbation schemes in *config/hyp/fb_ours.yaml* (here we choose seed 1,2,3,4,5) 
### please specify the seed for model runs in *config/cfg.yaml* (here we choose seed 1,2,3,4,5)
python train_with_gradient_descent.py name=fbaug_ours hyp=fb_ours
