'''Shallow neural networks trained on the Airfoil Self-Noise Data Set'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

from get_data import get_data
from helper import init_data, obs_v_dim, diff_obs_v_dim

# get and preprocess data
X_train, y_train, X_test, y_test = get_data()

# define the neural network model
class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super(MLP, self).__init__()
    self.h1 = nn.Linear(input_size, 16)
    self.h2 = nn.Linear(16, output_size)
    self.activation = nn.ReLU()
      
  def forward(self, x):
    x = self.activation(self.h1(x))
    output = self.h2(x)
    return output

# implement various GD schemes: vanilla GD (baseline), GD with Gaussian perturbations, MPGD, MPGD variant Eq. (11) in the paper
def train(X_train, y_train, X_test, y_test, option=0, start_perturb=0, perturb_level=[0.0,0.0], gamma=0.65, beta=0.5, epochs=100, lr=0.01, mom=0.0, seed=1, input_size=5, output_size=1):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  regressor = MLP(input_size= input_size, output_size=output_size)
  criterion = nn.MSELoss()
  optimizer = optim.SGD(regressor.parameters(), lr=lr, momentum=mom)

  p = 0.5*(1+beta)

  total_params = sum(p.numel() for p in regressor.parameters())
  print('Number of trainable parameters: ', total_params)

  count = 0
  loss_list = []
  trainRMSE = []
  testRMSE = []

  for epoch in range(epochs):
      regressor.train()

      inputs = Variable(torch.from_numpy(X_train))
      targets = Variable(torch.from_numpy(y_train))
      outputs = regressor(inputs)
      
      if epoch >= start_perturb: 
        # simulate perturbations
        if option == 1 and epoch == start_perturb:
          p_vec_list = []
          for param in regressor.parameters():
            p_vec = obs_v_dim(p, epochs, gamma, param.data, seed=123456+seed)
            p_vec_list.append(p_vec)
        if option == 3 and epoch == start_perturb:
          p_vec_list = []
          for param in regressor.parameters():
            diff_p_vec = diff_obs_v_dim(p, epochs, gamma, param.data, seed=123456+seed)
            p_vec_list.append(diff_p_vec)
       

        # adding the perturbations to GD
        reg = 0.0
        if option == 0: #baseline
          loss = criterion(outputs, targets)
        elif option == 1 or option == 3: #inject perturbations (different for all parameters in all layers)
          param_count = 0
          for param in regressor.parameters():
            if param.requires_grad:
              p_perturb = p_vec_list[param_count]
              reg += (perturb_level[0] * (p_perturb[count] * param).sum() + perturb_level[1] * (p_perturb[count] * param * param).sum()/2) 
              param_count += 1
          loss = criterion(outputs, targets) + (lr**gamma) * reg / lr
        elif option == 2: #inject uncorrelated Gaussians for all parameters
          param_count = 0
          for param in regressor.parameters():
            if param.requires_grad:
              reg += perturb_level[0] * torch.matmul(torch.randn(param.shape), param.T).sum() + perturb_level[1] * (torch.randn(param.shape) * param * param).sum()/2
              param_count += 1
          loss = criterion(outputs, targets) + reg
        else:
          print('invalid option!')

        count += 1

      else:
        loss = criterion(outputs, targets)
          
      # collecting performance data at each epoch
      loss_list.append(criterion(outputs, targets).detach().numpy())

      y_pred = regressor(Variable(torch.from_numpy(X_test))).data.numpy()
      testScore = math.sqrt(np.mean((y_test - y_pred)**2))
      testRMSE.append(testScore)

      y_pred_train = regressor(Variable(torch.from_numpy(X_train))).data.numpy()
      trainScore = math.sqrt(np.mean((y_train - y_pred_train)**2))
      trainRMSE.append(trainScore)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

  return loss_list, trainRMSE, testRMSE

# run the experiments 
def run_exp(option=0, lr=0.1, epochs=3000, start_perturb=0, gamma=0.6, beta=0.5, sigma=0.0, mu=0.0, n_run=5):
    list0 = []
    diff0 = []
    for i in range(n_run):
        loss_list0, trainRMSE0, testRMSE0 = train(X_train, y_train, X_test, y_test, start_perturb=start_perturb, gamma=gamma, beta=beta, perturb_level=[sigma, mu], epochs=epochs, option=option, lr=lr, mom=0.0, seed=1+i)
        list0.append(testRMSE0[-1]) 
        diff0.append(testRMSE0[-1]-loss_list0[-1])
    print('===========> Option: ', option)
    if option == 1 or option == 3:
        print('gamma: ', gamma, ', beta: ', beta, ', mu: ', mu, ', sigma: ', sigma)
    elif option == 2:
        print('sigma: ', sigma, ', mu: ', mu)
    print('test RMSEs: ', list0)
    print('RMSE gaps: ', diff0)
    print('Average test RMSE: ', statistics.mean(list0) )
    print('Average RMSE gap: ', statistics.mean(diff0) )
    print('Std dev of test RMSE: ', statistics.stdev(list0))
    print('Std dev of RMSE gap: ', statistics.stdev(diff0))
    print('Maximum test RMSE: ', max(list0))
    print('Minimum test RMSE: ', min(list0))
    print('Maximum RMSE gap: ', max(diff0))
    print('Minimum RMSE gap: ', min(diff0))
    print('--------------------------------------')


########################################## main ###########################################
import statistics

lr = 0.1
epochs = 3000
n_run = 5

print('=============== Using lr of ', lr, ' and run for ', epochs, ' epochs ============== ')

# baseline
print('==== baseline (vanilla GD) ===== ')
run_exp(option=0, lr=lr, epochs=epochs, start_perturb=0, n_run=n_run)


# GD with uncorrelated Gaussian perturbations
print('==== GD with Gaussian perturbations ===== ')
sigma = 0.02 
mu = 0.01 
run_exp(option=2, lr=lr, epochs=epochs, start_perturb=0, sigma=sigma, mu=mu, n_run=n_run)


# MPGD with beta = 0.0
print('==================== MPGD with beta = 0.0 ================== ')
sigma = 0.02
mu = 0.01 
beta = 0.0
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)

# MPGD with perturbations of the form v(y1)-v(y2), where v is the observable and y1 and y2 are independent Thaler iterates
print('==== Another MPGD variant ===== ')
sigma = 0.01 
mu = 0.01
beta = 0.0
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)


# MPGD with beta = 0.25
print('==================== MPGD with beta = 0.25 ================== ')
sigma = 0.01 
mu = 0.01 
beta = 0.25
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)

# MPGD with perturbations of the form v(y1)-v(y2), where v is the observable and y1 and y2 are independent Thaler iterates
print('==== Another MPGD variant ===== ')
sigma = 0.01 
mu = 0.01 
beta = 0.25
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)


# MPGD with beta = 0.5
print('==================== MPGD with beta = 0.5 ================== ')
sigma = 0.01 
mu = 0.01
beta = 0.5
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=1, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)

# MPGD with perturbations of the form v(y1)-v(y2), where v is the observable and y1 and y2 are independent Thaler iterates
print('==== Another MPGD variant ===== ')
sigma = 0.01
mu = 0.01
beta = 0.5
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.55, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.60, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.65, beta=beta, sigma=sigma, mu=mu, n_run=n_run)
run_exp(option=3, lr=lr, epochs=epochs, start_perturb=0, gamma=0.70, beta=beta, sigma=sigma, mu=mu, n_run=n_run)

