import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.utils.net.common import MLP

from components.dqn.network import DQNNet


class NoisyLinear(nn.Module):
  """Implementation of Noisy Networks. arXiv:1706.10295.
  :param int in_features: the number of input features.
  :param int out_features: the number of output features.
  :param float noisy_std: initial standard deviation of noisy linear layers.
  .. note::
    Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
    /fqf_iqn_qrdqn/network.py .
  """

  def __init__(
    self, in_features: int, out_features: int, noisy_std: float = 0.5
  ) -> None:
    super().__init__()

    # Learnable parameters.
    self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
    self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

    # Factorized noise parameters.
    self.register_buffer('eps_p', torch.FloatTensor(in_features))
    self.register_buffer('eps_q', torch.FloatTensor(out_features))

    self.in_features = in_features
    self.out_features = out_features
    self.sigma = noisy_std

    self.reset()
    self.sample()

  def reset(self) -> None:
    bound = 1 / np.sqrt(self.in_features)
    self.mu_W.data.uniform_(-bound, bound)
    self.mu_bias.data.uniform_(-bound, bound)
    self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
    self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

  def f(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.randn(x.size(0), device=x.device)
    return x.sign().mul_(x.abs().sqrt_())

  def sample(self) -> None:
    self.eps_p.copy_(self.f(self.eps_p))  # type: ignore
    self.eps_q.copy_(self.f(self.eps_q))  # type: ignore

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.training:
      weight = self.mu_W + self.sigma_W * (
        self.eps_q.ger(self.eps_p)  # type: ignore
      )
      bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()  # type: ignore
    else:
      weight = self.mu_W
      bias = self.mu_bias

    return F.linear(x, weight, bias)


class NoisyNet(DQNNet):
  """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
  For advanced usage (how to customize the network), please refer to
  :ref:`build_the_network`.
  """

  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    device: Union[str, int, torch.device] = "cpu",
    output_dim: Optional[int] = None,
    noisy_std: float = 0.5,
    is_noisy: bool = True,
  ) -> None:
    super().__init__(c, h, w, action_shape, device, features_only=True, output_dim=output_dim)
    
    def linear(x, y):
      if is_noisy:
        return NoisyLinear(x, y, noisy_std)
      else:
        return nn.Linear(x, y)
    
    self.action_num = np.prod(action_shape)
    self.Q = nn.Sequential(
      linear(self.output_dim, 512), nn.ReLU(inplace=True),
      linear(512, self.action_num)
    )
    self.output_dim = self.action_num

  def forward(
    self,
    x: Union[np.ndarray, torch.Tensor],
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: x -> Z(x, \*)."""
    x, state = super().forward(x)
    q = self.Q(x)
    return q, state