import torch
import numpy as np
from typing import Any, Dict, Optional, Union

from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from components.noisy_net_dqn.network import NoisyLinear


def sample_noise(model: torch.nn.Module) -> bool:
  """Sample the random noises of NoisyLinear modules in the model.
  :param model: a PyTorch module which may have NoisyLinear submodules.
  :returns: True if model has at least one NoisyLinear submodule;
    otherwise, False.
  """
  done = False
  for m in model.modules():
    if isinstance(m, NoisyLinear):
      m.sample()
      done = True
  return done

class NoisyNetPolicy(DQNPolicy):
  """
  Implementation of Noisy Net DQN.
  """
  def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
    sample_noise(self.model)
    if self._target and sample_noise(self.model_old):
      self.model_old.train()  # So that NoisyLinear takes effect
    return super().learn(batch, **kwargs)

  def forward(
    self,
    batch: Batch,
    state: Optional[Union[dict, Batch, np.ndarray]] = None,
    model: str = "model",
    input: str = "obs",
    **kwargs: Any,
  ) -> Batch:
    if not self.updating:   # Sample noise at each interaction step
      sample_noise(self.model)
      self.model.train()
    return super().forward(batch, state, model, input, **kwargs)