from envs.env import *
from utils.helper import *
from agents.BaseAgent import *

from tianshou.data import VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer

from components.noisy_net_dqn.network import NoisyNet
from components.noisy_net_dqn.policy import NoisyNetPolicy


class NoisyNetDQN(BaseAgent):
  '''
  Implementation of NoisyNet DQN
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
    self.policy = NoisyNetPolicy(
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
      'Train': Collector(self.policy, self.envs['Train'], self.buffer, exploration_noise=False),
      'Test': Collector(self.policy, self.envs['Test'], exploration_noise=False)
    }
    # Load checkpoint
    if self.cfg['resume_from_log']:
      self.load_checkpoint()

  def createNN(self):
    NN = NoisyNet(
      *self.state_shape,
      action_shape = self.action_shape, 
      device = self.device,
      noisy_std = self.cfg['agent']['noisy_std'],
      is_noisy = True
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
      train_fn = None,
      test_fn = None,
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