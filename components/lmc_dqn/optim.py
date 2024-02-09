import math
import torch
from torch import Tensor
from typing import List, Optional

from torch.optim.sgd import *
from torch.optim.adam import *
from torch.optim.rmsprop import *


class aSGLD(Adam):
  """
  Implementation of Adam SGLD based on: http://arxiv.org/abs/2009.09535
  Built on PyTorch Adam implementation.
  Note that there is no bias correction in the original description of Adam SGLD.
  """
  def __init__(
    self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0, amsgrad=False,
    noise_scale=0.01, a=1.0
  ):
    defaults = dict(
      lr=lr, betas=betas, eps=eps, 
      weight_decay=weight_decay, amsgrad=amsgrad
    )
    super(aSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
          
          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      adam_sgld(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=group['amsgrad'],
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
        noise_scale=self.noise_scale,
        a=self.a
      )
    return loss


def adam_sgld(
  params: List[Tensor],
  grads: List[Tensor],
  exp_avgs: List[Tensor],
  exp_avg_sqs: List[Tensor],
  max_exp_avg_sqs: List[Tensor],
  state_steps: List[int],
  *,
  amsgrad: bool,
  beta1: float,
  beta2: float,
  lr: float,
  weight_decay: float,
  eps: float,
  noise_scale: float,
  a: float
):
  """Functional API that performs Adam SGLD algorithm computation.
  See :class:`~torch.optim.Adam` for details.
  """
  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = max_exp_avg_sqs[i].sqrt().add_(eps)
    else:
      denom = exp_avg_sq.sqrt().add_(eps)
    
    # Add pure gradient
    param.add_(grad, alpha=-lr)
    # Add the adaptive bias term
    am = a * exp_avg
    param.addcdiv_(am, denom, value=-lr)
    # Add noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    param.add_(noise_scale * math.sqrt(2.0*lr) * grad_perturb)