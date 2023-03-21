from envs.env import *
from utils.helper import *
from agents.BaseAgent import *

from tianshou.policy import DQNPolicy
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer

from components.dqn.network import DQNNet


class DQN(BaseAgent):
  '''
  Implementation of DQN.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Create Q network
    self.net = self.createNN()
    # Set optimizer
    self.optimizer = getattr(torch.optim, self.cfg['optimizer']['name'])(self.net.parameters(), **self.cfg['optimizer']['kwargs'])
    # Set replay buffer: `save_last_obs` and `stack_num` can be removed when you have enough RAM
    self.buffer = VectorReplayBuffer(
      total_size = self.cfg['buffer_size'],
      buffer_num = self.cfg['env']['train_num'],
      ignore_obs_next = True,
      save_only_last_obs = self.save_only_last_obs,
      stack_num = self.cfg['frames_stack']
    )
    # Define policy
    self.policy = DQNPolicy(
      model = self.net,
      optim = self.optimizer,
      discount_factor = self.discount,
      estimation_step = self.cfg['n_step'],
      target_update_freq = self.cfg['target_update_steps'],
      reward_normalization = False,
      is_double = self.cfg['agent']['is_double'],
      clip_loss_grad = self.cfg['clip_loss_grad'] # if True, use huber loss
    )
    # Set Collectors
    self.collectors =  {
      'Train': Collector(self.policy, self.envs['Train'], self.buffer, exploration_noise=True),
      'Test': Collector(self.policy, self.envs['Test'], exploration_noise=True)
    }
    # Load checkpoint
    if self.cfg['resume_from_log']:
      self.load_checkpoint()

  def createNN(self):
    NN = DQNNet(
      *self.state_shape,
      action_shape = self.action_shape, 
      device = self.device
    )
    return NN.to(self.device)

  def run_steps(self):
    # Test train_collector and start filling replay buffer
    self.collectors['Train'].collect(n_step=self.batch_size * self.cfg['env']['train_num'])
    # Trainer
    result = offpolicy_trainer(
      policy = self.policy,
      train_collector = self.collectors['Train'],
      test_collector = self.collectors['Test'],
      max_epoch = self.cfg['epoch'],
      step_per_epoch = self.cfg['step_per_epoch'],
      step_per_collect = self.cfg['step_per_collect'],
      episode_per_test = self.cfg['env']['test_num'],
      batch_size = self.batch_size,
      update_per_step = self.cfg['update_per_step'],
      train_fn = self.train_fn,
      test_fn = self.test_fn,
      save_best_fn = self.save_model if self.cfg['save_model'] else None,
      logger = self.logger,
      verbose = True,
      # Set it to True to show speed, etc.
      show_progress = self.cfg['show_progress'],
      test_in_train = True,
      # Resume training setting
      resume_from_log = self.cfg['resume_from_log'],
      save_checkpoint_fn = self.save_checkpoint,
    )
    for k, v in result.items():
      self.logger.info(f'{k}: {v}')

  def train_fn(self, epoch, env_step):
    # Linear decay epsilon in the first eps_steps
    if env_step <= self.cfg['agent']['eps_steps']:
      eps = self.cfg['agent']['eps_start'] - env_step / self.cfg['agent']['eps_steps'] * \
        (self.cfg['agent']['eps_start'] - self.cfg['agent']['eps_end'])
    else:
      eps = self.cfg['agent']['eps_end']
    self.policy.set_eps(eps)

  def test_fn(self, epoch, env_step):
    self.policy.set_eps(self.cfg['agent']['eps_test'])