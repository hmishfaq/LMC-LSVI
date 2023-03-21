import math
import torch
from torch import Tensor
from typing import List, Optional

from torch.optim.adam import *


class LangevinAdam(Adam):
  def __init__(
    self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0, amsgrad=False,
    noise_scale=0.01
  ):
    super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
    self.noise_scale = noise_scale

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
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

      langevin_adam(
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
        noise_scale=self.noise_scale
      )

    return loss


def langevin_adam(
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
  noise_scale: float
):
  """Functional API that performs Langevin Adam algorithm computation.
  See :class:`~torch.optim.Adam` for details.
  """

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
    else:
      denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    step_size = lr / bias_correction1
    param.addcdiv_(exp_avg, denom, value=-step_size)

    # Add noise to gradient as grad_perturb
    scaled_lr = 2 * lr * bias_correction2_sqrt / bias_correction1
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device) / exp_avg_sq.sqrt().add_(eps).sqrt()
    param.add_(noise_scale * math.sqrt(scaled_lr) * grad_perturb)