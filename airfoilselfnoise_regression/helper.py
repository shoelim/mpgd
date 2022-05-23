'''Helper functions for implementing the perturbations used in MPGD (see Section 3 of the paper for full details) '''

import numpy as np
import torch
from scipy.optimize import fsolve
from scipy.special import gamma as Gamma

# define the Thaler map
def Thaler_map(x, gamma):
  return ((x**(1-gamma) + (1+x)**(1-gamma) - 1)**(1/(1-gamma))) % 1

# define the chi_j 
def chi_func(x, x_star, p, seed):
  np.random.seed(seed)
  if x <= x_star:
    return 1.0
  else: 
    return 2*np.random.binomial(1,p,1)[0] - 1

# define v
def v_func(x, x_star, eta, gamma):
  alpha = 1/gamma
  if x <= x_star:
    return eta * ((1-2**(gamma-1))**(-gamma)) * ( (alpha**alpha)*(1-gamma)*Gamma(1-alpha)*np.cos(alpha*np.pi/2)/(2**(1-gamma)-1) )**(-gamma)
  else:
    return (eta * ((1-2**(gamma-1))**(-gamma)) * ( (alpha**alpha)*(1-gamma)*Gamma(1-alpha)*np.cos(alpha*np.pi/2)/(2**(1-gamma)-1) )**(-gamma))/(1-2**(1-gamma))

# define the observables v^{(k)} 
def obs_v(x_0, p, T, gamma, eta=1.0, seed=1234):
  y = x_0
  def func(x, gamma=gamma):
   return x**(1-gamma) + (1+x)**(1-gamma) - 2
  x_star =  fsolve(func, (0.01))[0]
  chi_i = chi_func(x=x_0, x_star=x_star, p=p, seed=seed) 
  v_0 = chi_i * v_func(y, x_star=x_star, eta=eta, gamma=gamma)
  v_list = []
  v_list.append(v_0)

  for i in range(T):
    y = Thaler_map(y, gamma=gamma)
    chi_i = chi_i * chi_func(x=y, x_star=x_star, p=p, seed=seed)
    v_new = v_func(x = y, x_star=x_star, eta=eta, gamma=gamma)
    v = chi_i * v_new
    v_list.append(v)
  return v_list

# define initial conditions following Gottwald-Melbourne (2021)
def init_data(gamma, seed=1234):
  np.random.seed(seed)
  x = np.random.uniform()
  for i in range(10000):
    x = Thaler_map(x, gamma=gamma)
  x_0 = x
  return x_0

# construct higher dimensional versions of the v^{(k)} (to be compatible with the shape of the parameters)
def obs_v_dim(p, T, gamma, param, eta=1.0, seed=1234):
  dim = torch.numel(param)
  obs_vec = torch.zeros((dim,T+1))
  for i in range(dim):
    x0 = init_data(gamma,i+seed) 
    obs_vec[i, :] = torch.tensor(obs_v(x0, p, T, gamma, eta=eta, seed=i+seed))
  if len(param.size()) == 1:
    return obs_vec.view(T+1, param.shape[0]) 
  elif len(param.size()) == 2:
    return obs_vec.view(T+1, param.shape[0], param.shape[1])
  elif len(param.size()) == 3:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2])
  elif len(param.size()) == 4:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2], param.shape[3])
  elif len(param.size()) == 5:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])
  else:
    print('invalid shape!')

# construct higher dimensional versions of the observables for the MPGD variant (see Eq. (11) in the paper)
def diff_obs_v_dim(p, T, gamma, param, eta=1.0, seed=1234):
  dim = torch.numel(param)
  obs_vec = torch.zeros((dim,T+1))
  for i in range(dim):
    x0 = init_data(gamma,i+seed) 
    x00 = init_data(gamma,i+100*seed) 
    obs_vec[i, :] = torch.tensor( np.array(obs_v(x0, p, T, gamma, eta=eta, seed=i+seed)) - np.array(obs_v(x00, p, T, gamma, eta=eta, seed=i+seed)) )
  if len(param.size()) == 1:
    return obs_vec.view(T+1, param.shape[0]) 
  elif len(param.size()) == 2:
    return obs_vec.view(T+1, param.shape[0], param.shape[1])
  elif len(param.size()) == 3:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2])
  elif len(param.size()) == 4:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2], param.shape[3])
  elif len(param.size()) == 5:
    return obs_vec.view(T+1, param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])
  else:
    print('invalid shape!')

