# Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning (NeurIPS 2026)

> Safety-critical control tasks on a public simulation benchmark (Sections 5.1–5.2)

This code contains the official implementation of **DexDICE** (**D**ual-**ex**pert offline imitation learning via **DICE**) and several baselines for the DSRL benchmark experiments on offline imitation learning from dual-expert demonstrations.

## Quickstart

```bash
conda env create -f environment.yml
conda activate dexdice

# 1) preprocess offline dataset into trajectory pkl files
python preprocess.py --task SafetyCarCircle-v0 --class_type safe --output_dir ./dataset
python preprocess.py --task SafetyCarCircle-v0 --class_type performant --output_dir ./dataset
python preprocess.py --task SafetyCarCircle-v0 --class_type unsafe --output_dir ./dataset
python preprocess.py --task SafetyCarCircle-v0 --class_type shuffle --output_dir ./dataset

# 2) train (config-driven)
python train.py --pid 0 --config ./configs/dexdice.yaml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate dexdice
```

### (Optional) MetaDrive environments

If you want to run MetaDrive tasks, install the clean MetaDrive fork used by this code:

```bash
pip install "git+https://github.com/HenryLHH/metadrive_clean.git@main"
```

## Dataset preprocessing (trajectory classification)

Run `preprocess.py` to convert DSRL offline datasets into trajectory pickles (`.pkl`) and summary statistics (`.yaml`).

- The original DSRL dataset is downloaded/cached by DSRL on first run (typically under `~/.dsrl/datasets`).
- Outputs are saved under `--output_dir/<task>/` (default: `./dataset/<task>/`).
- Thresholds (cost/reward) are read from `./configs/dsrl_task_info.yaml` by default. The file lists all 38 re-curated DSRL tasks of the paper (cost limit of the task and the 90th percentile of safe-trajectory returns). A task that is not listed falls back to a cost threshold of 40 and a return threshold of 0, which labels every trajectory as performant, so add the task first.

### Command

```bash
python preprocess.py \
  --task SafetyCarCircle-v0 \
  --class_type safe \
  --output_dir ./dataset \
  --random_seed 42 \
  --task_info ./configs/dsrl_task_info.yaml
```

### `class_type` options and output files

The `--class_type` flag controls which subset is written.

- **`safe`**: all safe trajectories (safe-only + safe&performant) → `dsrl-safe-v0.pkl`
- **`performant`**: all performant trajectories (performant-only + safe&performant) → `dsrl-performance-v0.pkl`
- **`unsafe`**: trajectories that violate safety (neither + performant-only) → `dsrl-nonpreferred-v0.pkl`
- **`safe-performant`**: trajectories that are both safe and performant → `dsrl-safe-performant-v0.pkl`
- **`shuffle`**: all trajectories shuffled → `dsrl-shuffled-v0.pkl`

Each run also writes a matching stats file under `stats/` (e.g., `stats/dsrl-safe-v0-stats.yaml`).

### Example output layout

```
dataset/
└── SafetyCarCircle-v0/
    ├── dsrl-nonpreferred-v0.pkl
    ├── dsrl-performance-v0.pkl
    ├── dsrl-safe-performant-v0.pkl
    ├── dsrl-safe-v0.pkl
    ├── dsrl-shuffled-v0.pkl
    └── stats/
        ├── dsrl-nonpreferred-v0-stats.yaml
        ├── dsrl-performance-v0-stats.yaml
        ├── dsrl-safe-performant-v0-stats.yaml
        ├── dsrl-safe-v0-stats.yaml
        └── dsrl-shuffled-v0-stats.yaml
```

## Training & evaluation

Training is config-driven via `train.py`:

```bash
python train.py --pid 0 --config ./configs/dexdice.yaml
```

- `--pid`: process id, also used as the random seed for Python, NumPy, and PyTorch (overrides `train.seed` in the config).
- `--task_info`: per-task threshold table (default `./configs/dsrl_task_info.yaml`); the task's cost limit is read from it for evaluation.
- `--config`: YAML config (see `./configs/dexdice.yaml`).
- Device selection is automatic (`cuda` if available, otherwise `cpu`).
- Evaluation prints the episode return normalized to $[0,1]$ by the task's return range and the episode cost normalized by the task's cost limit (both from `configs/dsrl_task_info.yaml`); a method is safe when the normalized cost is $\le 1$.
- **Checkpoints and resuming.** Every evaluation also writes a resumable checkpoint under `./checkpoints/<method>_<env>_<seed>_cid_<cid>_pid_<pid>/`, and `train.py` automatically resumes from the latest one in that directory. Re-running a finished configuration with the same `--pid` therefore exits immediately without training; delete the directory (or change `--pid`) to start a fresh run.

### Config pointers (`configs/dexdice.yaml`)

Key fields you will typically edit:

- **`method`**: one of `DexDICE`, `BC`, `FBC`, `JFBC`, `DemoDICE-S`, `DemoDICE-U`, `DWBC-S`
- **`env_id`**: environment id (e.g., `SafetyCarCircle-v0`)
- **`dataset.dataset_path`**: directory containing the preprocessed `.pkl` files (e.g., `./dataset/SafetyCarCircle-v0`)
- **`dataset.<split>.types`**: dataset file stems **without** `.pkl` (e.g., `dsrl-safe-v0` refers to `dsrl-safe-v0.pkl`)
- **`train.save_policy_path`**: set a directory path to save checkpoints; if set, trainers write:
  - `bc_actor_best.pt`
  - `bc_actor_last.pt`

## Repository structure (high-level)

- `preprocess.py`: DSRL dataset → trajectory pickle classification utility
- `train.py`: training entrypoint (DexDICE + baselines)
- `configs/dexdice.yaml`: training configuration
- `configs/dsrl_task_info.yaml`: per-task cost/reward thresholds used during preprocessing and the cost limit used during evaluation
