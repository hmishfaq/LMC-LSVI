"""
Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License")
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_context("notebook")
sns.set_style("ticks")

plt.rcParams['ytick.right'] = True
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


GAME_NAMES = [
  ('alien', 'Alien'),
  ('amidar', 'Amidar'),
  ('assault', 'Assault'),
  ('asterix', 'Asterix'),
  ('asteroids', 'Asteroids'),
  ('atlantis', 'Atlantis'),
  ('bank_heist', 'Bank Heist'),
  ('battle_zone', 'Battlezone'),
  ('beam_rider', 'Beam Rider'),
  ('berzerk', 'Berzerk'),
  ('bowling', 'Bowling'),
  ('boxing', 'Boxing'),
  ('breakout', 'Breakout'),
  ('centipede', 'Centipede'),
  ('chopper_command', 'Chopper Command'),
  ('crazy_climber', 'Crazy Climber'),
  ('defender', 'Defender'),
  ('demon_attack', 'Demon Attack'),
  ('double_dunk', 'Double Dunk'),
  ('enduro', 'Enduro'),
  ('fishing_derby', 'Fishing Derby'),
  ('freeway', 'Freeway'),
  ('frostbite', 'Frostbite'),
  ('gopher', 'Gopher'),
  ('gravitar', 'Gravitar'),
  ('hero', 'H.E.R.O.'),
  ('ice_hockey', 'Ice Hockey'),
  ('jamesbond', 'James Bond 007'),
  ('kangaroo', 'Kangaroo'),
  ('krull', 'Krull'),
  ('kung_fu_master', 'Kung-Fu Master'),
  ('montezuma_revenge', 'Montezuma’s Revenge'),
  ('ms_pacman', 'Ms. Pac-Man'),
  ('name_this_game', 'Name This Game'),
  ('phoenix', 'Phoenix'),
  ('pitfall', 'Pitfall!'),
  ('pong', 'Pong'),
  ('private_eye', 'Private Eye'),
  ('qbert', 'Q*bert'),
  ('riverraid', 'River Raid'),
  ('road_runner', 'Road Runner'),
  ('robotank', 'Robotank'),
  ('seaquest', 'Seaquest'),
  ('skiing', 'Skiing'),
  ('solaris', 'Solaris'),
  ('space_invaders', 'Space Invaders'),
  ('star_gunner', 'Stargunner'),
  ('surround', 'Surround'),
  ('tennis', 'Tennis'),
  ('time_pilot', 'Time Pilot'),
  ('tutankham', 'Tutankham'),
  ('up_n_down', 'Up’n Down'),
  ('venture', 'Venture'),
  ('video_pinball', 'Video Pinball'),
  ('wizard_of_wor', 'Wizard of Wor'),
  ('yars_revenge', 'Yars’ Revenge'),
  ('zaxxon', 'Zaxxon'),
]
GAME_NAME_MAP = dict(GAME_NAMES)
selected_games = ['alien', 'freeway', 'gravitar', 'hero', 'pitfall', 'solaris', 'qbert', 'venture']

def load_experiment_data_from_results_csv_dir(experiment_details, results_dir):
  df_exps = []
  for ed in experiment_details:
    csv_file = os.path.join(results_dir, ed['agent_id'] + '.csv')
    with open(csv_file, 'r') as f:
      df = pd.read_csv(f, index_col=0)
    df = df.assign(agent_id=ed['agent_id'], agent_name=ed['agent_name'])
    # Cut into 50M frames
    df = df[df['frame'] <= 50e6]
    # Select games
    df = df[df['environment_name'].isin(selected_games)]
    df_exps.append(df)
  df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
  return df_exp

def moving_average(values, window_size):
  # numpy.convolve uses zero for initial missing values, so is not suitable.
  numerator = np.nancumsum(values)
  # The sum of the last window_size values.
  numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
  denominator = np.ones(len(values)) * window_size
  denominator[:window_size] = np.arange(1, window_size + 1)
  smoothed = numerator / denominator
  assert values.shape == smoothed.shape
  return smoothed

def smooth(df, smoothing_window, index_columns, columns):
  dfg = df.groupby(index_columns)
  for col in columns:
    df[col] = dfg[col].transform(lambda s: moving_average(s.values, smoothing_window))
  return df

def environment_pretty(row):
  return GAME_NAME_MAP[row['environment_name']]

def add_columns(df):
  df['environment'] = df.apply(environment_pretty, axis=1)
  df['frame_millions'] = df['frame'] / int(1e6)
  return df

def smooth_dataframe(df):
  return smooth(
      df,
      smoothing_window=10,
      index_columns=['agent_id', 'environment_name', 'seed'],
      columns=[
        'normalized_return',
        'eval_episode_return',
      ])

def make_agent_hue_kws(experiments):
  pairs = [(exp['agent_name'], exp['color']) for exp in experiments]
  agent_names, colors = zip(*pairs)
  hue_kws = dict(color=colors)
  return list(agent_names), hue_kws

def plot_individual(df, agent_names, hue_kws):
  g = sns.FacetGrid(
    df.query('agent_name == %s' % agent_names),
    row=None,
    col='environment',
    hue='agent_name',
    height=2.5,
    aspect=1.35,
    col_wrap=4,
    hue_order=agent_names,
    sharey=False,
    hue_kws=hue_kws,
  )

  g = g.map(
    sns.lineplot,
    'frame_millions',
    'eval_episode_return',
    estimator='median',
    errorbar='se',
    # alpha=0.5,
    linewidth=3,
  )
  g.despine(left=False, top=True, right=False, bottom=False)
  g.set_titles(col_template='{col_name}', row_template='{row_name}')
  g.set_axis_labels('Frame (millions)', 'Return')

  # Create legend from the final axes.
  legend_x_margin = 0.03
  legend_y_offset_inches = 0
  legend_y_offset = legend_y_offset_inches / g.fig.get_figheight()
  g.axes[-1].legend(
    bbox_to_anchor=(legend_x_margin, legend_y_offset, 1 - 2 * legend_x_margin, 0),
    bbox_transform=g.fig.transFigure,
    mode='expand',
    ncol=10,
    borderaxespad=0,
    loc='lower left',
    frameon=False,
  )
  g.fig.subplots_adjust(bottom=0.2)
  g.fig.subplots_adjust(bottom=0.16)
  return g

# Plot
agent_ids = ['AdamLMCDQN_double', 'AdamLMCDQN_nodouble', 'LangevinAdam_double', 'LangevinAdam_nodouble']
for agent_id in agent_ids:
  print(f'Plot for agent_id={agent_id}')
  if agent_id == 'AdamLMCDQN_double':
    experiments = [
      dict(agent_id='AdamLMCDQN_double', agent_name='Adam LMCDQN', color='tab:red'),
      dict(agent_id='NoisyNet_double', agent_name='NoisyNet DQN', color='tab:grey'),
      dict(agent_id='double_q', agent_name='Double DQN', color='tab:purple'),
      dict(agent_id='prioritized', agent_name='Prioritized DQN', color='tab:olive'),
      dict(agent_id='c51', agent_name='C51', color='tab:orange'),
      dict(agent_id='qrdqn', agent_name='QR-DQN', color='tab:blue'),
      dict(agent_id='BootDQN_double', agent_name='Bootstrapped DQN', color='black'),
      dict(agent_id='iqn', agent_name='IQN', color='tab:green'),
    ]
  elif agent_id == 'AdamLMCDQN_nodouble':
    experiments = [
      dict(agent_id='AdamLMCDQN_double', agent_name='Adam LMCDQN (with Double Q)', color='tab:red'),
      dict(agent_id='AdamLMCDQN_nodouble', agent_name='Adam LMCDQN (w.o. Double Q)', color='tab:blue')
    ]
  elif agent_id == 'LangevinAdam_double':
    experiments = [
      dict(agent_id='AdamLMCDQN_double', agent_name='Adam LMCDQN', color='tab:red'),
      dict(agent_id='LangevinAdam_double', agent_name='Langevin DQN', color='tab:blue')
    ]
  elif agent_id == 'LangevinAdam_nodouble':
    experiments = [
      dict(agent_id='LangevinAdam_double', agent_name='Langevin DQN (with Double Q)', color='tab:red'),
      dict(agent_id='LangevinAdam_nodouble', agent_name='Langevin DQN (w.o. Double Q)', color='tab:blue')
    ]
  df_exp_raw = load_experiment_data_from_results_csv_dir(experiments, '.')
  df_exp = df_exp_raw.pipe(add_columns).pipe(smooth_dataframe)
  df = df_exp.sort_values(by=['agent_id', 'environment_name', 'seed', 'frame'])
  g = plot_individual(df, *make_agent_hue_kws(experiments))
  g.fig.savefig(f'atari_{agent_id}.pdf')
  plt.close(g.fig)