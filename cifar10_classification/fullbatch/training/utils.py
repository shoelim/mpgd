"""Utility functions for training."""
import torch

import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma as Gamma


def Thaler_map(x, gamma):
  return ((x**(1-gamma) + (1+x)**(1-gamma) - 1)**(1/(1-gamma))) % 1

def chi_func(x, x_star, p, seed):
  np.random.seed(seed)
  if x <= x_star:
    return 1.0
  else: 
    return 2*np.random.binomial(1,p,1)[0] - 1

def v_func(x, x_star, eta, gamma):
  alpha = 1/gamma
  if x <= x_star:
    return eta * ((1-2**(gamma-1))**(-gamma)) * ( (alpha**alpha)*(1-gamma)*Gamma(1-alpha)*np.cos(alpha*np.pi/2)/(2**(1-gamma)-1) )**(-gamma)
  else:
    return (eta * ((1-2**(gamma-1))**(-gamma)) * ( (alpha**alpha)*(1-gamma)*Gamma(1-alpha)*np.cos(alpha*np.pi/2)/(2**(1-gamma)-1) )**(-gamma))/(1-2**(1-gamma))

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


#choose IC following Gottwald-Melbourne (2021)
def init_data(gamma, seed=1234):
  np.random.seed(seed)
  x = np.random.uniform()
  for i in range(10000):
    x = Thaler_map(x, gamma=gamma)
  x_0 = x
  return x_0

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
    
    
@torch.no_grad()
def _clip_gradient_list(grads, scaled_clip, cfg, eps=1e-6):
    """Clip inplace with the same implementation as torch.nn.utils.clip, but use _foreach_ functions.

    Returns 1 if the batch was clipped.
    """
    if cfg.hyp.grad_clip_norm == float('inf'):
        grad_norm = max(g.abs().max() for g in grads)
    else:
        grad_norm = torch.norm(torch.stack([torch.norm(g, cfg.hyp.grad_clip_norm) for g in grads]),
                               cfg.hyp.grad_clip_norm)  # same norm as pytorch clipping
    if grad_norm > scaled_clip:
        torch._foreach_mul_(grads, scaled_clip / (grad_norm + eps))
        return 1
    else:
        return 0


def _update_ema(model, ema_model, momentum=0.995):
    """Update exponential moving average in parameters and buffers."""
    with torch.no_grad():
        for param_source, param_target in zip(model.parameters(), ema_model.parameters()):
            param_target.copy_(momentum * param_target.data + (1 - momentum) * param_source.data)
        for buffer_source, buffer_target in zip(model.buffers(), ema_model.buffers()):
            buffer_target.copy_(momentum * buffer_target.data + (1 - momentum) * buffer_source.data)


@torch.no_grad()
def _allreduce_coalesced(model, average_grads):
    """This helps a tiny bit in multi-node settings and doesn't appear to hurt single-node performance."""
    concat_grad = torch.cat([g.reshape(-1) for g in average_grads])
    torch.distributed.all_reduce(concat_grad, async_op=False)

    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.grad = concat_grad[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

@torch.no_grad()
def _save_to_checkpoint(model, optimizer, scheduler, scaler, counter, file='checkpoints/fb.pth'):
    optim_state = optimizer.state_dict()
    model_state = model.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict() if scaler is not None else None
    step = counter.step

    torch.save([optim_state, model_state, scheduler_state, scaler_state, step], file)

@torch.no_grad()
def _load_from_checkpoint(model, optimizer, scheduler, scaler, counter, max_steps, device=None, file='checkpoints/fb.pth'):
    try:
        optim_state, model_state, scheduler_state, scaler_state, step = torch.load(file, map_location=device)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        scheduler.load_state_dict(scheduler_state)
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
            counter.scale = scaler.get_scale()
        counter.step = step
        if step >= max_steps:
            raise ValueError('Maximum step size reached. Terminating computations.')
        else:
            print(f'Existing checkpoint loaded successfully. Continuing to train from step {step}.')
    except FileNotFoundError:
        print('No existing checkpoint found. Starting to train from step 0.')

@torch.no_grad()
def _save_state_for_visualization(model, optimizer, cfg, path='models/viz.pth'):
    grads = [p.grad.detach() for p in model.parameters()]
    momentum_terms = [optimizer.state[p]['momentum_buffer'] for group in optimizer.param_groups for p in group['params']]
    # Nesterov:
    if cfg.hyp.optim.nesterov:
        update_directions = [grad.add(mom, alpha=cfg.hyp.optim.momentum) for mom, grad in zip(momentum_terms, grads)]
    else:
        update_directions = momentum_terms
    payload = dict(state_dict=model.state_dict(),
                   model_cfg=cfg.model,
                   grads=grads,
                   update_directions=update_directions)
    torch.save(payload, path)
