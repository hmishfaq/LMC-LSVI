import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence


class BootstrappedDQNNet(nn.Module):
  """
  Bootstrapped DQN Net.
  """
  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    device: Union[str, int, torch.device] = "cpu",
    num_ensemble: int = 10,
    prior_scale: float = 0.,
  ) -> None:
    super().__init__()
    self.device = device
    self.num_ensemble = num_ensemble
    self.prior_scale = prior_scale

    self.prior_model = EnsembleDQNNet(c, h, w, action_shape, device, num_ensemble)
    self.model = EnsembleDQNNet(c, h, w, action_shape, device, num_ensemble)

    self.core = self.model.net

    for parameter in self.prior_model.parameters():
      parameter.requires_grad = False

  def forward(
    self,
    obs: Union[np.ndarray, torch.Tensor],
    head: int = None,
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
    model: str = "all"
  ) -> Tuple[torch.Tensor, Any]:
    r"""Mapping: s -> Q(s, \*)."""
    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
    if model == "all":
      prior_q, _ = self.prior_model(obs, head, state, info)
      q, _ = self.model(obs, head, state, info)
      q_value = prior_q * self.prior_scale + q
    elif model == "prior":
      prior_q, _ = self.prior_model(obs, head, state, info)
      q_value = prior_q * self.prior_scale
    else:
      raise ValueError(model)
    return q_value, state

  def compute_feature_rank(self, x: Union[torch.Tensor, np.ndarray], delta=0.01) -> int:
    return self.model.compute_feature_rank(x, delta)

  def get_weight_norm(self):
    return self.model.get_weight_norm()


class EnsembleDQNNet(nn.Module):
  def __init__(
    self,
    c: int,
    h: int,
    w: int,
    action_shape: Sequence[int],
    device: Union[str, int, torch.device] = "cpu",
    num_ensemble: int = 10,
  ) -> None:
    super().__init__()
    self.device = device
    self.num_ensemble = num_ensemble

    self.net = nn.Sequential(
      nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
      nn.Flatten()
    )
    with torch.no_grad():
      cnn_output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
    self.net = nn.Sequential(
      self.net,
      nn.Linear(cnn_output_dim, 512),
      nn.ReLU(inplace=True)
    )
    self.output_dim = int(np.prod(action_shape))

    self.head_list = nn.ModuleList([
      nn.Linear(512, self.output_dim)
      for _ in range(num_ensemble)
    ])

  def forward(
    self,
    obs: Union[np.ndarray, torch.Tensor],
    head: int = None,
    state: Optional[Any] = None,
    info: Dict[str, Any] = {},
  ) -> Tuple[torch.Tensor, Any]:
    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
    feature = self.net(obs)
    if head is not None:
      q_value = self.head_list[head](feature)
    else:
      q_value = [self.head_list[k](feature) for k in range(self.num_ensemble)]
      q_value = torch.stack(q_value, dim=1)
    return q_value, state

  def compute_feature_rank(self, x: Union[torch.Tensor, np.ndarray], delta=0.01) -> int:
    x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    h = self.net(x)
    _, s, _ = torch.svd(h, compute_uv=False)
    z = torch.cumsum(s, dim=0)

    rank = torch.nonzero(z >= z[-1] * (1. - delta))[0][0]
    return rank.item()

  def get_weight_norm(self):
    norm = 0.0
    for param in self.parameters():
      norm += torch.norm(param).item()
    return norm