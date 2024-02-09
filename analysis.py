import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, memory_info
from utils.helper import set_one_thread


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-10:].mean() if mode=='Train' else result['Return'][-5:].mean(),
    'Return (max)': result['Return'].max() if mode=='Train' else result['Return'].max()
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (max)': result['Return (max)'].max()
  }
  return result_dict

cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Return',
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None, 
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'upper left',
  'sweep_keys': ['optimizer/name', 'optimizer/kwargs/noise_scale', 'optimizer/kwargs/a', 'agent/update_num', 'agent/is_double', 'agent/eps_start'],
  'sort_by': ['Return (mean)', 'Return (max)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Test', indexes='all')

if __name__ == "__main__":
  exp, runs = 'atari8_lmc', 5
  # exp, runs = 'atari8_boot', 5
  # exp, runs = 'atari8_noisynet', 5
  unfinished_index(exp, runs=runs)
  memory_info(exp, runs=runs)
  analyze(exp, runs=runs)