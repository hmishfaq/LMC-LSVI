import torch
from typing import Any, Dict, Optional

from tianshou.data import ReplayBuffer
from tianshou.policy import DQNPolicy


class LMCDQNPolicy(DQNPolicy):
  """Implementation of LMC DQN.
  
  :param torch.nn.Module model: a model following the rules in
    :class:`~tianshou.policy.BasePolicy`. (s -> logits)
  :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
  :param float discount_factor: in [0, 1].
  :param int estimation_step: the number of steps to look ahead. Default to 1.
  :param int target_update_freq: the target network update frequency (0 if
    you do not use the target network). Default to 0.
  :param bool reward_normalization: normalize the reward to Normal(0, 1).
    Default to False.
  :param bool is_double: use double dqn. Default to True.
  :param bool clip_loss_grad: clip the gradient of the loss in accordance
    with nature14236; this amounts to using the Huber loss instead of
    the MSE loss. Default to False.
  :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
    optimizer in each policy.update(). Default to None (no lr_scheduler).

  .. seealso::

    Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
    explanation.
  """
  def __init__(
    self,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    discount_factor: float = 0.99,
    estimation_step: int = 1,
    target_update_freq: int = 0,
    reward_normalization: bool = False,
    is_double: bool = False,
    clip_loss_grad: bool = False,
    update_num: int = 1,
    **kwargs: Any,
  ) -> None:
    super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
    self.update_num = update_num

  def update(self, 
    sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
    """Update the policy network and replay buffer.
    It includes 3 function steps: process_fn, learn, and post_process_fn. In
    addition, this function will change the value of ``self.updating``: it will be
    False before this function and will be True when executing :meth:`update`.
    Please refer to :ref:`policy_state` for more detailed explanation.
    :param int sample_size: 0 means it will extract all the data from the buffer,
        otherwise it will sample a batch with given sample_size.
    :param ReplayBuffer buffer: the corresponding replay buffer.
    :return: A dict, including the data needed to be logged (e.g., loss) from
        ``policy.learn()``.
    """
    if buffer is None:
        return {}
    # Perform multiple updates
    self.updating = True
    for _ in range(self.update_num):
      batch, indices = buffer.sample(sample_size)
      batch = self.process_fn(batch, buffer, indices)
      result = self.learn(batch, **kwargs)
      self.post_process_fn(batch, buffer, indices)
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()
    self.updating = False
    return result