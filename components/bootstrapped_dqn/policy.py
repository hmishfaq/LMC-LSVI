import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Any, Dict, Union, Optional, List, Callable

from tianshou.policy.base import _nstep_return
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

from components.bootstrapped_dqn.network import BootstrappedDQNNet


class BootstrappedDQNPolicy(DQNPolicy):
  """
  Implementation of Bootstrapped DQN.
  ref: https://arxiv.org/abs/1602.04621, https://arxiv.org/abs/1806.03335
  """
  def __init__(
    self,
    model: BootstrappedDQNNet,
    optim: torch.optim.Optimizer,
    discount_factor: float = 0.99,
    estimation_step: int = 1,
    target_update_freq: int = 0,
    reward_normalization: bool = False,
    is_double: bool = False,
    clip_loss_grad: bool = False,
    mask_prob: float = 0.,
    normalize_grad: bool = False,
    evaluation_mode: str = "sample",
    compute_rank_interval: int = 0,
    rank_batch_size: int = 1024,
    **kwargs: Any,
  ) -> None:
    super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
    self._normalize_grad = normalize_grad
    self._evaluation_mode = evaluation_mode
    self._num_ensemble = model.num_ensemble
    self._mask_prob = mask_prob
    self._compute_rank_interval = compute_rank_interval
    self._rank_batch_size = rank_batch_size
    self._feature_rank = 0
    self._active_head_train = None
    self._active_head_test = None

  def forward(
    self,
    batch: Batch,
    state: Optional[Union[dict, Batch, np.ndarray]] = None,
    model: str = "model",
    input: str = "obs",
    **kwargs: Any,
  ) -> Batch:
    if self.training:
      if self._active_head_train is None or (len(batch.done.shape) == 0 or batch.done[0]):
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        assert len(obs_) == 1, (
          "Current only support one actor mode."
        )
        self._active_head_train = np.random.randint(low=0, high=self._num_ensemble)
      return self._model_forward(batch, self._active_head_train, state, model, input)
    else:
      if self._evaluation_mode == "vote":
        return self._model_forward(batch, None, state, model, input, mode="vote")
      else:
        if self._active_head_test is None or (len(batch.done.shape) == 0 or batch.done[0]):
          obs = batch[input]
          obs_ = obs.obs if hasattr(obs, "obs") else obs
          assert len(obs_) == 1, (
            "Current only support one actor mode."
          )
          self._active_head_test = np.random.randint(low=0, high=self._num_ensemble)
        return self._model_forward(batch, self._active_head_test, state, model, input)

  def _target_q(self, buffer: ReplayBuffer, indice: np.ndarray) -> torch.Tensor:
    batch = buffer[indice]  # batch.obs_next: s_{t+n}
    if self._target:
      # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
      with torch.no_grad():
        batch_result = self._model_forward(
          batch, active_head=None, model="model_old", input="obs_next"
        )  # (None, num_ensemble, num_action)
      target_q = batch_result.logits
      if self._is_double:
        a = self._model_forward(batch, active_head=None, input="obs_next").act  # (None, num_ensemble)
      else:
        a = batch_result.act
      a_one_hot = F.one_hot(torch.as_tensor(a, device=target_q.device), self.max_action_num).to(torch.float32)
      target_q = torch.sum(target_q * a_one_hot, dim=-1)  # (None, num_ensemble)
    else:
      with torch.no_grad():
        target_q = self._model_forward(batch, active_head=None, input="obs_next").logits.max(dim=-1)[0]
    return target_q

  def _model_forward(
    self,
    batch: Batch,
    active_head: int = None,
    state: Optional[Union[dict, Batch, np.ndarray]] = None,
    model: str = "model",
    input: str = "obs",
    mode: str = "sample",
  ):
    model = getattr(self, model)
    obs = batch[input]
    obs_ = obs.obs if hasattr(obs, "obs") else obs

    logits, h = model(obs_, active_head, state=state, info=batch.info)
    q = self.compute_q_value(logits, getattr(obs, "mask", None))
    if not hasattr(self, "max_action_num"):
      self.max_action_num = q.shape[-1]

    if mode == "sample":
      # q: (batch_size, num_action) or (batch_size, num_ensemble, num_action)
      act = to_numpy(q.max(dim=-1)[1])
    elif mode == "vote":
      assert not self.training
      # q: (batch_size, num_ensemble, num_action)
      act = to_numpy(q.max(dim=-1)[1])   # (batch_size, num_ensemble)
      assert act.ndim == 2 and act.shape[1] == self._num_ensemble
      act = stats.mode(act, axis=1)[0]  # (batch_size, 1)
      act = act.flatten()  # (batch_size, )
    else:
      raise ValueError(mode)
    return Batch(logits=logits, act=act, state=h)

  def update(
    self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
  ) -> Dict[str, Any]:
    """Update the policy network and replay buffer.
    """
    if buffer is None or len(buffer) < sample_size:
      return {}
    batch, indice = buffer.sample(sample_size)
    self.updating = True
    batch = self.process_fn(batch, buffer, indice)
    result = self._learn(buffer, batch, **kwargs)
    self.post_process_fn(batch, buffer, indice)
    self.updating = False
    return result

  def _learn(self, buffer: ReplayBuffer, batch: Batch, **kwargs: Any) -> Dict[str, float]:
    assert hasattr(batch, "ensemble_mask"), (
      "You need to add ensemble masks before learning."
    )

    if self._target and self._iter % self._freq == 0:
      self.sync_weight()

    # Optimize
    self.optim.zero_grad()
    q = self._model_forward(batch, active_head=None).logits  # (None, num_ensemble, num_action)
    a_one_hot = F.one_hot(
      torch.as_tensor(batch.act, device=q.device), self.max_action_num
    ).to(torch.float32)  # (None, num_actions)
    q = torch.einsum('bka,ba->bk', q, a_one_hot)  # (None, num_ensemble)
    r = to_torch_as(batch.returns, q)  # (None, num_ensemble)
    td = r - q

    masks = to_torch_as(batch.ensemble_mask, q)
    loss = torch.mean(td.pow(2) * masks, dim=0).sum()
    loss.backward()

    if self._normalize_grad:
      assert self._mask_prob == 1.
      assert hasattr(self.model, "core")
      for param in self.model.core.parameters():
        param.grad.data *= 1. / self._num_ensemble
    self.optim.step()

    if self._compute_rank_interval and (self._iter % self._compute_rank_interval == 0) and \
        len(buffer) >= self._rank_batch_size:
      batch, _ = buffer.sample(self._rank_batch_size)
      self._feature_rank = self.model.compute_feature_rank(batch.obs)

    self._iter += 1

    weight_norm = 0.
    if hasattr(self.model, "get_weight_norm"):
      weight_norm = self.model.get_weight_norm()
    q_prior = self.model(batch.obs, None, model="prior")[0]

    result = {
      "boot/loss": loss.item() / self._num_ensemble if self._normalize_grad else loss.item(),
      "boot/q": q.mean().item(),
      "boot/q_std": q.std(dim=1).mean().item(),
      "boot/prior_value": q_prior.mean().item(),
      "boot/prior_std": q_prior.std(dim=1).mean().item(),
      "boot/td_value": td.mean().item(),
      "boot/weight_norm": weight_norm,
      "rank_feat": self._feature_rank
    }
    return result

  @staticmethod
  def compute_nstep_return(
    batch: Batch,
    buffer: ReplayBuffer,
    indice: np.ndarray,
    target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
    gamma: float = 0.99,
    n_step: int = 1,
    rew_norm: bool = False,
  ) -> Batch:
    assert not rew_norm, \
      "Reward normalization in computing n-step returns is unsupported now."
    rew = buffer.rew
    bsz = len(indice)
    indices = [indice]
    for _ in range(n_step - 1):
      indices.append(buffer.next(indices[-1]))
    indices = np.stack(indices)
    # Terminal indicates buffer indexes nstep after 'indice',
    # and are truncated at the end of each episode
    terminal = indices[-1]

    # TODO: return contains all target_q for ensembles
    target_qs = []
    with torch.no_grad():
      target_q_torch = target_q_fn(buffer, terminal)  # (None, num_ensemble)
    end_flag = buffer.done.copy()
    end_flag[buffer.unfinished_index()] = True
    for k in range(target_q_torch.shape[1]):
      target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1))
      target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
      target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
      target_qs.append(target_q.flatten())

    target_qs = np.array(target_qs)   # (num_ensemble, None)
    target_qs = target_qs.transpose([1, 0])  # (None, num_ensemble)
    batch.returns = to_torch_as(target_qs, target_q_torch)
    if hasattr(batch, "weight"):  # prio buffer update
      batch.weight = to_torch_as(batch.weight, target_q_torch)
    return batch
