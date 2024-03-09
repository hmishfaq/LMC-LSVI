# Adam LMCDQN

This is the official implementation of *Adam LMCDQN* algorithm, introduced in our ICLR 2024 paper [Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo](https://arxiv.org/abs/2305.18246).


## Installation

- Python: >=3.8
- [PyTorch](https://pytorch.org/): GPU version
- [Tianshou](https://github.com/thu-ml/tianshou): ==0.4.10
- [Envpool](https://github.com/sail-sg/envpool): ==0.6.6
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `atari8_lmc.json` and configuration index `1`:

```python main.py --config_file ./configs/atari8_lmc.json --config_idx 1```


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `atari8_lmc.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in atari8_lmc.json: 72`

Then we run through all configuration indexes from `1` to `72`. The simplest way is using a bash script:

``` bash
for index in {1..72}
do
  python main.py --config_file ./configs/atari8_lmc.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/atari8_lmc.json --config_idx {1} ::: $(seq 1 72)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should have the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 73 145 217 289
do
  python main.py --config_file ./configs/atari8_lmc.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/atari8_lmc.json --config_idx {1} ::: $(seq 1 72 360)
```


### Analysis (Optional)

To analyze the experimental results, just run:

`python analysis.py`

Inside `analysis.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in directory `logs/atari8_lmc/0`. `analyze` will generate `csv` files that store training and test results. Please check `analysis.py` for more details. More functions are available in `utils/plotter.py`.

Enjoy!


## Citation

If you find this work useful to your research, please cite our paper.

```bibtex
@inproceedings{ishfaq2024provable,
  title={Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo},
  author={Ishfaq, Haque and Lan, Qingfeng and Xu, Pan and Mahmood, A Rupam and Precup, Doina and Anandkumar, Anima and Azizzadenesheli, Kamyar},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Acknowledgement

We thank the following projects which provide great references:

- [Tianshou](https://github.com/thu-ml/tianshou)
- [Explorer](https://github.com/qlan3/Explorer)
