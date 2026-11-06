# DexDICE: Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning

Official code for the NeurIPS 2026 paper *DexDICE: Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning*.

DexDICE (**D**ual-**ex**pert offline imitation learning via stationary **DI**stribution **C**orrection **E**stimation) learns a policy from three demonstration sets without reward or cost labels:

- `D_S` (safe demonstrations): satisfy the safety constraint, possibly suboptimal in return.
- `D_P` (performant demonstrations): achieve high return, possibly unsafe.
- `D_U` (unlabeled demonstrations): a larger offline dataset that provides transition coverage.

The policy matches the stationary distribution of `D_P` while being restricted to the safe support estimated from `D_S` (Section 4 of the paper).

## Repository layout

```
code/
├── dsrl_code/              # Simulation benchmarks: PyTorch
│   ├── preprocess.py       #   DSRL dataset -> re-curated trajectory pickles (D_S / D_P / D_U)
│   ├── train.py            #   training + evaluation entry point 
│   ├── configs/
│   │   ├── dexdice.yaml    #   training configuration
│   │   └── dsrl_task_info.yaml  # per-task cost / return thresholds used for re-curation
│   ├── imitation/          #   DexDICE and baselines 
│   ├── core/               #   policies, replay buffers, dataset loading
│   ├── rlkit/              #   minimal vendored utilities (networks, distributions)
│   ├── dataset/            #   output of preprocess.py (generated locally; git-ignored)
│   └── README.md           #   detailed instructions for this part
└── real_code_data_demo/    # Real mobile-robot navigation: JAX
    ├── train_dexdice.py    #   single-file DexDICE trainer with rollout-based checkpoint selection
    ├── mobile_env_2d.py    #   calibrated 2D rollout environment 
    ├── teleoperation_data/ #   teleoperation demonstrations (D_P, D_S, D_U) and arena layout
    ├── demo.gif            #   a successful real-robot trial of DexDICE
    └── README.md           #   detailed instructions for this part
```

Each subdirectory is self-contained with its own environment specification and README. See
[`dsrl_code/README.md`](dsrl_code/README.md) and
[`real_code_data_demo/README.md`](real_code_data_demo/README.md) for step-by-step commands.

## Quick start

### Simulation benchmarks (`dsrl_code/`)

```bash
cd dsrl_code
conda env create -f environment.yml && conda activate dexdice

# 1) re-curate a DSRL task into D_S / D_P / D_U trajectory pickles
python preprocess.py --task SafetyCarCircle-v0 --class_type safe       --output_dir ./dataset
python preprocess.py --task SafetyCarCircle-v0 --class_type performant --output_dir ./dataset
python preprocess.py --task SafetyCarCircle-v0 --class_type shuffle    --output_dir ./dataset

# 2) train and evaluate (method, task, and hyperparameters are set in the YAML config)
python train.py --pid 0 --config ./configs/dexdice.yaml
```

The `method` field of the config selects `DexDICE`, `BC`, `FBC`, `JFBC`, `DemoDICE-S`, `DemoDICE-U`, or `DWBC-S`;
`--pid` sets the random seed. Evaluation runs 50 rollouts every `eval_freq` iterations;
the paper reports the last checkpoint (`bc_actor_last.pt` when `train.save_policy_path` is set).

### Real mobile robot (`real_code_data_demo/`)

```bash
cd real_code_data_demo
uv sync                         # installs jax, flax, optax (see pyproject.toml)
uv run python train_dexdice.py  # trains on the bundled teleoperation data
```

Training hyperparameters are the constants at the top of `train_dexdice.py`.
Checkpoints are selected by rollouts in the calibrated 2D simulator (`mobile_env_2d.py`) before
deployment on the physical robot, as described in Appendix H.

## Datasets

### Re-curated DSRL datasets

`preprocess.py` downloads the public DSRL datasets and labels every trajectory along two binary axes
using the thresholds in `configs/dsrl_task_info.yaml`:

- safe: episode cost `<= cost_threshold` (the DSRL cost limit of the task);
- performant: episode return `>= reward_threshold` (the 90th percentile of safe-trajectory returns).

A task must be listed under `environments:` in `configs/dsrl_task_info.yaml`; for a task that is not
listed, `preprocess.py` falls back to a cost threshold of 40 and a return threshold of 0, which labels
every trajectory as performant. Add the task's thresholds before preprocessing.

The resulting pickles contain lists of trajectory dictionaries with keys
`observations`, `actions`, `rewards`, `costs`, `next_observations`, `terminals`, and `trajectory_id`.
`rewards` and `costs` are kept only for dataset construction and evaluation; no training method in
this repository reads them. `trajectory_id` is the index of the trajectory in the original DSRL
dataset of the task and is shared across the `safe`, `performance`, and `shuffled` files, so the same
trajectory can be matched across subsets.

### Teleoperation data (real robot)

`real_code_data_demo/teleoperation_data/` contains 114 joystick teleoperation episodes recorded by a
single operator on a Robotnik SUMMIT-XL-Steel with a 2D LiDAR:

| File | Role | Episodes |
|---|---|---|
| `dataset_aggressive.pkl` | `D_P` (fast, possibly unsafe) | 10 |
| `dataset_safe.pkl` | `D_S` (safe, slow) | 10 |
| `dataset_unlabeled.pkl` | remaining episodes; `D_U` is this file plus `D_P` and `D_S` | 94 |
| `layout.json` | arena, start pose, goal pose, obstacles used for checkpoint selection | |

Each pickle is a flat dictionary with per-step arrays `scan` (370 LiDAR ranges in metres),
`pose` (x, y, yaw), `vel` (vx, vy, wz), `goal` (x, y, yaw), `cmd` (commanded vx, vy, wz),
`done`, `timeout`, `time`, together with `episode_starts`, `episode_lengths`, `episode_settings`,
`hz`, and `terminal_thresholds`. The state and action used for learning are derived from these
arrays in `train_dexdice.py` (`build_state`).



## Citation

```bibtex
@inproceedings{seo2026dexdice,
  title     = {Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning},
  author    = {Seo, Seokin and Shyn, Sung Kuk and Seo, Minji and Lee, Yisoo and Lee, Jongmin and Yang, Hongseok and Kim, Kee-Eung},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
