"""DexDICE training (JAX/Flax).

Algorithm (DICE-style off-policy IL with safety constraint):
  1. (Pretrain) reward disc D(s, a)  : expert vs mixed
                uc_disc D_hat(s, a)   : safe   vs mixed
  2. Threshold tau = quantile_q( D_hat(safe) ) — periodically refreshed.
  3. Per training step:
     - r(s, a) = log D / (1 - D)
     - safe_indicator = 1[ D_hat(s, a) > tau ]
     - A = r + lambda * safe_indicator + gamma * nu(s') - nu(s)
     - nu gradient   <- grad_nu     L_tilde(nu, lambda)   (numerically stable form)
     - lambda gradient <- grad_lambda L(nu, lambda)
     - actor grad    <- grad -E_d_U[ softmax(A / (alpha + 1) - 1) * log pi(a | s) ]
                                                                      weighted BC

Stack: jax + flax (linen) + optax.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import json
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


# -----------------------------------------------------------------------------
# Repo paths
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR  = _REPO_ROOT / 'teleoperation_data'
_CKPT_DIR  = _REPO_ROOT / 'checkpoints'

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUTPUT_DIR = _CKPT_DIR / 'dexdice'

# SCAN_MODE: 'none' | 'sparse'.
#   'none'   : ignore scan (state-only).
#   'sparse' : top-K nearest beams, encoded as [range, cos(angle), sin(angle)] per
#              beam. Output dim = 3K. No learnable parameters.
SCAN_MODE       = 'sparse'
N_SCAN_SPARSE_K = 10

# Train-time Gaussian noise injection on (scan, state).
USE_INPUT_NOISE       = True
TRAIN_SCAN_NOISE_STD  = 0.001
TRAIN_STATE_NOISE_STD = 0.001

SEED = 0

# Dataset paths.
DATASET_PATHS_EXPERT = [_DATA_DIR / 'dataset_aggressive.pkl']
DATASET_PATHS_SAFE   = [_DATA_DIR / 'dataset_safe.pkl']
DATASET_PATHS_MIXED  = [_DATA_DIR / 'dataset_unlabeled.pkl'] \
                          + DATASET_PATHS_EXPERT + DATASET_PATHS_SAFE

# Optimization
TOTAL_STEPS = 1_000_000
BATCH_SIZE  = 512
LR_ACTOR    = 3e-5
LR_NU       = 3e-5
LR_DISC     = 3e-5
LR_LAMBDA   = 3e-4

GAMMA       = 0.99
ALPHA       = 0.0      # KL regularization
LAMBDA_INIT = 5.0      # initial Lagrange multiplier

# Gradient penalty (WGAN-GP style)
DISC_GP_WEIGHT     = 10.0
UC_DISC_GP_WEIGHT  = 0.0
NU_GP_WEIGHT       = 10.0

# Discriminator inner-loop and pretrain
DISC_STEPS_PER_MAIN         = 1
PRIOR_LEARNING_DURING_TRAIN = True
PRETRAIN_DISC_STEPS         = 0

# Safety threshold (tau): quantile of D_hat over the safe pool.
DISC_QUANTILE         = 0.2
THRESHOLD_UPDATE_FREQ = 1000

# Logging / save
LOG_EVERY  = 1_000
SAVE_EVERY = 10_000

# Rollout-based best-policy selection (calibrated layout).
EVAL_EVERY        = 10_000
EVAL_EPISODES     = 50
EVAL_MAX_STEPS    = 500
EVAL_LAYOUT_PATH  = str(_DATA_DIR / 'layout.json')
EVAL_W_COLLIDE    = 100.
EVAL_W_SV         = 100.
EVAL_W_END_DIST   = 0.3
EVAL_INIT_PERTURB_XY  = 0.20
EVAL_INIT_PERTURB_YAW = 0.20

# Network sizes
STATE_FEAT_DIM = 32
HIDDEN_DIM     = 256
DISC_HIDDEN    = (256, 256)
NU_HIDDEN      = (256, 256)

SCAN_NORMALIZE = 30.0
LOG_STD_INIT   = -1.0
LOG_STD_CLAMP  = (-5.0, 1.0)

_SCAN_NOISE_EFF  = TRAIN_SCAN_NOISE_STD  if USE_INPUT_NOISE else 0.0
_STATE_NOISE_EFF = TRAIN_STATE_NOISE_STD if USE_INPUT_NOISE else 0.0

# State is 7-dim body-frame delta-to-goal:
#   [vel(3), dx_body, dy_body, cos(dyaw), sin(dyaw)]
STATE_DIM = 7


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def build_state(vel, pose, goal):
    """7-dim body-frame delta-to-goal state."""
    pose_yaw = pose[..., 2]
    dx_w = goal[..., 0] - pose[..., 0]
    dy_w = goal[..., 1] - pose[..., 1]
    c, s = np.cos(pose_yaw), np.sin(pose_yaw)
    dx_b =  dx_w * c + dy_w * s
    dy_b = -dx_w * s + dy_w * c
    dyaw = np.arctan2(np.sin(goal[..., 2] - pose_yaw),
                       np.cos(goal[..., 2] - pose_yaw))
    return np.stack([
        vel[..., 0], vel[..., 1], vel[..., 2],
        dx_b, dy_b, np.cos(dyaw), np.sin(dyaw),
    ], axis=-1).astype(np.float32)


def load_one_dataset_with_next(path):
    """Load a flat dataset and build per-step next_idx + done mask using
    episode boundaries. Terminal steps self-loop and are masked via (1 - done)."""
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    n = len(ds['scan'])
    next_idx = np.arange(n, dtype=np.int64)
    done     = np.zeros(n, dtype=np.float32)
    starts  = np.asarray(ds['episode_starts'],  dtype=np.int64)
    lengths = np.asarray(ds['episode_lengths'], dtype=np.int64)
    for s, L in zip(starts, lengths):
        if L > 1:
            next_idx[s:s + L - 1] = np.arange(s + 1, s + L)
        next_idx[s + L - 1] = s + L - 1
        done[s + L - 1] = 1.0
    info = {'path': str(path), 'steps': n,
            'episodes': len(starts), 'meta': ds.get('meta', {})}
    return ds, next_idx, done, starts, info


def load_and_concat_with_next(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    flat_fields = ['scan', 'pose', 'vel', 'goal', 'cmd']
    accum = {f: [] for f in flat_fields}
    accum_next = []
    accum_done = []
    accum_starts = []
    info_list = []
    cur_offset = 0
    for p in paths:
        ds, next_idx, done, starts, info = load_one_dataset_with_next(p)
        info_list.append(info)
        for f in flat_fields:
            accum[f].append(ds[f])
        accum_next.append(next_idx + cur_offset)
        accum_done.append(done)
        accum_starts.append(starts + cur_offset)
        cur_offset += len(ds['scan'])
    out = {f: np.concatenate(accum[f], axis=0) for f in flat_fields}
    out['next_idx']       = np.concatenate(accum_next)
    out['done']           = np.concatenate(accum_done)
    out['episode_starts'] = np.concatenate(accum_starts)
    return out, info_list


def make_replay_buffer(paths):
    ds, info = load_and_concat_with_next(paths)
    scan   = ds['scan'].astype(np.float32) / SCAN_NORMALIZE
    state  = build_state(ds['vel'], ds['pose'], ds['goal'])
    action = ds['cmd'].astype(np.float32)
    next_idx = ds['next_idx']
    next_state = state[next_idx]
    next_scan  = scan[next_idx]
    init_idx   = ds['episode_starts']
    return {
        'scan': scan, 'state': state, 'action': action,
        'next_scan': next_scan, 'next_state': next_state,
        'done':       ds['done'].astype(np.float32),
        'init_idx':   init_idx,
        'init_scan':  scan[init_idx],
        'init_state': state[init_idx],
        'size': len(action),
        'info': info,
    }


def random_batch(buf, batch_size, rng):
    idx = rng.integers(0, buf['size'], size=batch_size)
    return {
        'scan':       buf['scan'][idx],
        'state':      buf['state'][idx],
        'action':     buf['action'][idx],
        'next_scan':  buf['next_scan'][idx],
        'next_state': buf['next_state'][idx],
        'done':       buf['done'][idx],
    }


def random_init_batch(buf, batch_size, rng):
    idx = rng.integers(0, len(buf['init_state']), size=batch_size)
    return buf['init_scan'][idx], buf['init_state'][idx]


# -----------------------------------------------------------------------------
# Rollout-based evaluation
# -----------------------------------------------------------------------------

def make_eval_env_and_predict(actor_apply, layout_path, max_steps,
                                 n_beams_default=370):
    from mobile_env_2d import MobileEnv2D, EnvConfig
    cfg = EnvConfig(fixed_layout_path=layout_path,
                     n_beams=n_beams_default, max_steps=max_steps)
    env = MobileEnv2D(cfg)

    @jax.jit
    def predict(params, scan, state):
        mean, _ = actor_apply({'params': params}, scan, state)
        return mean

    return env, predict


def _perturb_init(env, seed, perturb_xy, perturb_yaw):
    if perturb_xy <= 0 and perturb_yaw <= 0:
        return
    rng = np.random.default_rng(int(seed))
    dx   = rng.uniform(-perturb_xy,  perturb_xy)  if perturb_xy  > 0 else 0.0
    dy   = rng.uniform(-perturb_xy,  perturb_xy)  if perturb_xy  > 0 else 0.0
    dyaw = rng.uniform(-perturb_yaw, perturb_yaw) if perturb_yaw > 0 else 0.0
    env.pose = (env.pose + np.array([dx, dy, dyaw], dtype=np.float32))
    env.pose[2] = np.arctan2(np.sin(env.pose[2]), np.cos(env.pose[2]))
    env.start_pose = env.pose.copy()


def evaluate_policy(env, predict, params, episodes, max_steps,
                     perturb_xy=0.0, perturb_yaw=0.0):
    results = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=int(seed))
        if perturb_xy > 0 or perturb_yaw > 0:
            _perturb_init(env, seed, perturb_xy, perturb_yaw)
            obs = env._obs()
        info = {}
        for _ in range(max_steps):
            scan_in  = (obs['scan'] / SCAN_NORMALIZE).astype(np.float32)
            # env returns 11-dim state in start frame; convert to 7-dim body-frame delta.
            full = obs['state']
            vx, vy, wz = full[0], full[1], full[2]
            px, py     = full[3], full[4]
            cp, sp     = full[5], full[6]
            gx, gy     = full[7], full[8]
            cg, sg     = full[9], full[10]
            pyaw = np.arctan2(sp, cp); gyaw = np.arctan2(sg, cg)
            dx_w = gx - px; dy_w = gy - py
            dx_b =  dx_w * cp + dy_w * sp
            dy_b = -dx_w * sp + dy_w * cp
            dyaw = np.arctan2(np.sin(gyaw - pyaw), np.cos(gyaw - pyaw))
            state_in = np.array([vx, vy, wz, dx_b, dy_b,
                                  np.cos(dyaw), np.sin(dyaw)],
                                 dtype=np.float32)
            a = np.asarray(predict(params,
                                     jnp.asarray(scan_in)[None],
                                     jnp.asarray(state_in)[None]))[0]
            obs, _, done, info = env.step(a)
            if done:
                break
        end_dist = float(np.hypot(env.pose[0] - env.goal[0],
                                    env.pose[1] - env.goal[1]))
        results.append({
            'success':                bool(info.get('reached', False)),
            'collide':                bool(info.get('collided', False)),
            'safety_violation_count': int(info.get('safety_violation_count', 0)),
            'end_dist':               end_dist,
            'length':                 int(info.get('step', 0)),
        })
    n = len(results)
    n_succ = sum(r['success'] for r in results)
    n_coll = sum(r['collide'] for r in results)
    sv_total = sum(r['safety_violation_count'] for r in results)
    avg_end = float(np.mean([r['end_dist'] for r in results]))
    avg_len = float(np.mean([r['length']   for r in results]))
    score = (n_succ
              - EVAL_W_COLLIDE * n_coll
              - EVAL_W_SV       * sv_total
              - EVAL_W_END_DIST * avg_end)
    summary = {
        'episodes':   n,        'success': n_succ,    'collide': n_coll,
        'safety_viol': sv_total, 'avg_end_dist': avg_end,
        'avg_length': avg_len,  'score': float(score),
    }
    return results, float(score), summary


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class StateEncoder(nn.Module):
    feat_dim: int = STATE_FEAT_DIM

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x); x = nn.relu(x)
        x = nn.Dense(self.feat_dim)(x); x = nn.relu(x)
        return x


def sparse_top_k_scan_jnp(scan, k):
    """(..., L) -> (..., 3k): the K nearest beams encoded as
    [range, cos(angle), sin(angle)]. Beams 0..L-1 are assumed evenly spaced
    over a full 2*pi field of view."""
    L = scan.shape[-1]
    if k > L:
        raise ValueError(f"k={k} > L={L}")
    # top_k returns the LARGEST k; negate to get the SMALLEST k ranges.
    neg_top, idx = jax.lax.top_k(-scan, k)
    smallest = -neg_top
    angles = idx.astype(jnp.float32) * (2.0 * jnp.pi / L)
    return jnp.concatenate([smallest, jnp.cos(angles), jnp.sin(angles)], axis=-1)


class GaussianPolicy(nn.Module):
    action_dim:     int
    state_feat_dim: int   = STATE_FEAT_DIM
    hidden_dim:     int   = HIDDEN_DIM
    log_std_init:   float = LOG_STD_INIT
    scan_mode:      str   = 'sparse'
    n_sparse_k:     int   = 10

    @nn.compact
    def __call__(self, scan, state):
        st = StateEncoder(self.state_feat_dim)(state)
        if self.scan_mode == 'sparse':
            sf = sparse_top_k_scan_jnp(scan, self.n_sparse_k)
            h = jnp.concatenate([sf, st], axis=-1)
        else:    # 'none'
            h = st
        h = nn.Dense(self.hidden_dim)(h); h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h); h = nn.relu(h)
        mean = nn.Dense(self.action_dim)(h)
        log_std = self.param('log_std',
                              nn.initializers.constant(self.log_std_init),
                              (self.action_dim,))
        log_std = jnp.clip(log_std, LOG_STD_CLAMP[0], LOG_STD_CLAMP[1])
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, log_std


class DiscriminatorNetwork(nn.Module):
    """(scan?) + state + action -> sigmoid scalar."""
    hidden_dims: tuple = DISC_HIDDEN
    scan_mode:   str   = 'sparse'
    n_sparse_k:  int   = 10

    @nn.compact
    def __call__(self, scan, state, action):
        if self.scan_mode == 'sparse':
            sf = sparse_top_k_scan_jnp(scan, self.n_sparse_k)
            x = jnp.concatenate([sf, state, action], axis=-1)
        else:    # 'none'
            x = jnp.concatenate([state, action], axis=-1)
        for h in self.hidden_dims:
            x = nn.Dense(h)(x); x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jax.nn.sigmoid(x)


class StateValue(nn.Module):
    """(scan?) + state -> scalar (state value)."""
    hidden_dims: tuple = NU_HIDDEN
    scan_mode:   str   = 'sparse'
    n_sparse_k:  int   = 10

    @nn.compact
    def __call__(self, scan, state):
        if self.scan_mode == 'sparse':
            sf = sparse_top_k_scan_jnp(scan, self.n_sparse_k)
            x = jnp.concatenate([sf, state], axis=-1)
        else:    # 'none'
            x = state
        for h in self.hidden_dims:
            x = nn.Dense(h)(x); x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def gaussian_log_prob(mean, log_std, action):
    var = jnp.exp(2.0 * log_std)
    return (-0.5 * (((action - mean) ** 2) / var
                      + 2.0 * log_std + jnp.log(2.0 * jnp.pi))).sum(axis=-1)


# -----------------------------------------------------------------------------
# Train steps (jit-able)
# -----------------------------------------------------------------------------

def _gp_disc(disc_apply_fn, params, interp_scan, interp_state, interp_act):
    def _sum_d(scan, s, a):
        return disc_apply_fn({'params': params}, scan, s, a).sum()
    g_scan  = jax.grad(_sum_d, argnums=0)(interp_scan, interp_state, interp_act)
    g_state = jax.grad(_sum_d, argnums=1)(interp_scan, interp_state, interp_act)
    g_act   = jax.grad(_sum_d, argnums=2)(interp_scan, interp_state, interp_act)
    grad_norm = jnp.sqrt(
        jnp.sum(g_scan  ** 2, axis=-1)
        + jnp.sum(g_state ** 2, axis=-1)
        + jnp.sum(g_act   ** 2, axis=-1) + 1e-12)
    return jnp.mean((grad_norm - 1.0) ** 2)


def _gp_nu(nu_apply_fn, params, interp_scan, interp_state):
    def _sum_nu(scan, s):
        return nu_apply_fn({'params': params}, scan, s).sum()
    g_scan  = jax.grad(_sum_nu, argnums=0)(interp_scan, interp_state)
    g_state = jax.grad(_sum_nu, argnums=1)(interp_scan, interp_state)
    grad_norm = jnp.sqrt(
        jnp.sum(g_scan  ** 2, axis=-1)
        + jnp.sum(g_state ** 2, axis=-1) + 1e-12)
    return jnp.mean((grad_norm - 1.0) ** 2)


def make_disc_step(disc_apply_fn, uc_disc_apply_fn,
                    disc_gp_weight, uc_disc_gp_weight,
                    scan_noise_std=0.0, state_noise_std=0.0):
    @jax.jit
    def disc_step(disc_state, uc_disc_state,
                  e_scan, e_state, e_act,
                  m_scan, m_state, m_act,
                  s_scan, s_state, s_act, rng):
        batch_size = m_state.shape[0]
        rng, perm_rng, coef_rng = jax.random.split(rng, 3)
        perm = jax.random.permutation(perm_rng, batch_size)
        coef = jax.random.uniform(coef_rng, (batch_size, 1))

        if scan_noise_std > 0 or state_noise_std > 0:
            rng, kn = jax.random.split(rng)
            ke, km, ks = jax.random.split(kn, 3)
            ke_sc, ke_st = jax.random.split(ke); km_sc, km_st = jax.random.split(km)
            ks_sc, ks_st = jax.random.split(ks)
            if scan_noise_std > 0:
                e_scan = e_scan + jax.random.normal(ke_sc, e_scan.shape) * scan_noise_std
                m_scan = m_scan + jax.random.normal(km_sc, m_scan.shape) * scan_noise_std
                s_scan = s_scan + jax.random.normal(ks_sc, s_scan.shape) * scan_noise_std
            if state_noise_std > 0:
                e_state = e_state + jax.random.normal(ke_st, e_state.shape) * state_noise_std
                m_state = m_state + jax.random.normal(km_st, m_state.shape) * state_noise_std
                s_state = s_state + jax.random.normal(ks_st, s_state.shape) * state_noise_std
        interp_scan  = coef * m_scan  + (1 - coef) * m_scan[perm]
        interp_state = coef * m_state + (1 - coef) * m_state[perm]
        interp_act   = coef * m_act   + (1 - coef) * m_act[perm]

        # Reward disc: expert vs mixed.
        def disc_loss_fn(params):
            d_e = disc_apply_fn({'params': params}, e_scan, e_state, e_act)
            d_m = disc_apply_fn({'params': params}, m_scan, m_state, m_act)
            bce = -jnp.mean(jnp.log(d_e + 1e-10)) - jnp.mean(jnp.log(1 - d_m + 1e-10))
            if disc_gp_weight > 0:
                gp = _gp_disc(disc_apply_fn, params,
                                interp_scan, interp_state, interp_act)
                return bce + disc_gp_weight * gp, (bce, gp)
            return bce, (bce, jnp.float32(0.0))

        (d_loss, (d_bce, d_gp)), d_grads = jax.value_and_grad(
            disc_loss_fn, has_aux=True
        )(disc_state.params)
        disc_state = disc_state.apply_gradients(grads=d_grads)

        # Uc disc: safe vs mixed.
        def uc_loss_fn(params):
            d_s = uc_disc_apply_fn({'params': params}, s_scan, s_state, s_act)
            d_m = uc_disc_apply_fn({'params': params}, m_scan, m_state, m_act)
            bce = -jnp.mean(jnp.log(d_s + 1e-10)) - jnp.mean(jnp.log(1 - d_m + 1e-10))
            if uc_disc_gp_weight > 0:
                gp = _gp_disc(uc_disc_apply_fn, params,
                                interp_scan, interp_state, interp_act)
                return bce + uc_disc_gp_weight * gp, (bce, gp)
            return bce, (bce, jnp.float32(0.0))

        (uc_loss, (uc_bce, uc_gp)), uc_grads = jax.value_and_grad(
            uc_loss_fn, has_aux=True
        )(uc_disc_state.params)
        uc_disc_state = uc_disc_state.apply_gradients(grads=uc_grads)

        return disc_state, uc_disc_state, {
            'disc_loss':    d_loss,  'disc_bce':    d_bce,  'disc_gp':    d_gp,
            'uc_disc_loss': uc_loss, 'uc_disc_bce': uc_bce, 'uc_disc_gp': uc_gp,
        }
    return disc_step


def make_main_step(actor_apply_fn, nu_apply_fn,
                    disc_apply_fn, uc_disc_apply_fn,
                    gamma, alpha, nu_gp_weight,
                    scan_noise_std=0.0, state_noise_std=0.0):
    @jax.jit
    def main_step(actor_state, nu_state, lambda_state,
                  disc_state, uc_disc_state,
                  init_scan, init_state,
                  m_scan, m_state, m_act,
                  m_next_scan, m_next_state, m_done,
                  threshold, rng):
        batch_size = m_state.shape[0]

        if scan_noise_std > 0 or state_noise_std > 0:
            rng, kn = jax.random.split(rng)
            ki, km, kn2 = jax.random.split(kn, 3)
            ki_sc, ki_st = jax.random.split(ki)
            km_sc, km_st = jax.random.split(km)
            kn_sc, kn_st = jax.random.split(kn2)
            if scan_noise_std > 0:
                init_scan   = init_scan   + jax.random.normal(ki_sc, init_scan.shape)   * scan_noise_std
                m_scan      = m_scan      + jax.random.normal(km_sc, m_scan.shape)      * scan_noise_std
                m_next_scan = m_next_scan + jax.random.normal(kn_sc, m_next_scan.shape) * scan_noise_std
            if state_noise_std > 0:
                init_state   = init_state   + jax.random.normal(ki_st, init_state.shape)   * state_noise_std
                m_state      = m_state      + jax.random.normal(km_st, m_state.shape)      * state_noise_std
                m_next_state = m_next_state + jax.random.normal(kn_st, m_next_state.shape) * state_noise_std

        # disc reward r = log D / (1 - D)
        d_out = jax.lax.stop_gradient(
            disc_apply_fn({'params': disc_state.params}, m_scan, m_state, m_act)
        )
        disc_r = (jnp.log(d_out + 1e-10) - jnp.log(1 - d_out + 1e-10)).reshape(-1)

        uc_out = jax.lax.stop_gradient(
            uc_disc_apply_fn({'params': uc_disc_state.params}, m_scan, m_state, m_act)
        )
        safe_indicator = (uc_out > threshold).astype(jnp.float32).reshape(-1)

        rng, perm_rng, coef_rng = jax.random.split(rng, 3)
        perm = jax.random.permutation(perm_rng, batch_size)
        coef = jax.random.uniform(coef_rng, (batch_size, 1))
        interp_scan  = coef * m_scan  + (1 - coef) * m_scan[perm]
        interp_state = coef * m_state + (1 - coef) * m_state[perm]

        lambda_val = lambda_state.params['value']
        not_done = 1.0 - m_done

        def nu_loss_fn(nu_params, lam):
            init_nu       = nu_apply_fn({'params': nu_params}, init_scan, init_state).reshape(-1)
            mixed_nu      = nu_apply_fn({'params': nu_params}, m_scan,    m_state).reshape(-1)
            mixed_nu_next = nu_apply_fn({'params': nu_params}, m_next_scan, m_next_state).reshape(-1)
            mixed_adv = disc_r + lam * safe_indicator + gamma * not_done * mixed_nu_next - mixed_nu
            lse = jax.nn.logsumexp(mixed_adv / (1 + alpha))
            nu_loss = (1 - gamma) * jnp.mean(init_nu) + (1 + alpha) * lse - lam
            if nu_gp_weight > 0:
                gp = _gp_nu(nu_apply_fn, nu_params, interp_scan, interp_state)
                nu_loss = nu_loss + nu_gp_weight * gp
            return nu_loss, (mixed_adv, jnp.mean(init_nu))

        (nu_loss_val, aux), nu_grads = jax.value_and_grad(
            nu_loss_fn, argnums=0, has_aux=True
        )(nu_state.params, lambda_val)
        mixed_adv, init_nu_mean = aux

        def lam_loss_fn(lam, nu_params):
            mixed_nu      = nu_apply_fn({'params': nu_params}, m_scan,    m_state).reshape(-1)
            mixed_nu_next = nu_apply_fn({'params': nu_params}, m_next_scan, m_next_state).reshape(-1)
            A = disc_r + lam * safe_indicator + gamma * not_done * mixed_nu_next - mixed_nu
            log_mean = jax.nn.logsumexp(A - 1) - jnp.log(A.shape[0])
            return jnp.exp(log_mean) - lam

        lam_loss_val, lam_grad = jax.value_and_grad(lam_loss_fn, argnums=0)(
            lambda_val, nu_state.params
        )

        nu_state     = nu_state.apply_gradients(grads=nu_grads)
        lambda_state = lambda_state.apply_gradients(grads={'value': lam_grad})

        adv = jax.lax.stop_gradient(mixed_adv)
        def actor_loss_fn(actor_params):
            log_w = adv / (alpha + 1) - 1
            log_w = log_w - jax.nn.logsumexp(log_w)
            w = jnp.exp(log_w)
            mean, log_std = actor_apply_fn({'params': actor_params}, m_scan, m_state)
            log_prob = gaussian_log_prob(mean, log_std, m_act)
            return -jnp.sum(w * log_prob), (mean, log_std)

        (actor_loss, _), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        log_N = jnp.log(adv.shape[0])
        e_exp_total = jnp.exp(jax.nn.logsumexp(adv - 1) - log_N)
        safe_log_sum = jax.nn.logsumexp(adv - 1, b=safe_indicator)
        e_exp_safe   = jnp.exp(safe_log_sum - log_N)

        return actor_state, nu_state, lambda_state, {
            'nu_loss':       nu_loss_val,
            'lambda_loss':   lam_loss_val,
            'policy_loss':   actor_loss,
            'lambda':        lambda_state.params['value'],
            'init_nu':       init_nu_mean,
            'safe_rate':     safe_indicator.mean(),
            'E_exp_A_total': e_exp_total,
            'E_exp_A_safe':  e_exp_safe,
        }
    return main_step


# -----------------------------------------------------------------------------
# Threshold computation
# -----------------------------------------------------------------------------

def compute_threshold(uc_disc_state, safe_buf, quantile, chunk=4096):
    n = safe_buf['size']
    vals = []
    for i in range(0, n, chunk):
        sc = jnp.asarray(safe_buf['scan'][i:i + chunk])
        s  = jnp.asarray(safe_buf['state'][i:i + chunk])
        a  = jnp.asarray(safe_buf['action'][i:i + chunk])
        d = uc_disc_state.apply_fn({'params': uc_disc_state.params}, sc, s, a)
        vals.append(np.asarray(d).reshape(-1))
    return float(np.quantile(np.concatenate(vals), quantile))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    jax_key = jax.random.PRNGKey(SEED)

    print("=== Loading buffers ===")
    print(f"[expert] paths: {[str(p) for p in DATASET_PATHS_EXPERT]}")
    expert = make_replay_buffer(DATASET_PATHS_EXPERT)
    print(f"[safe]   paths: {[str(p) for p in DATASET_PATHS_SAFE]}")
    safe   = make_replay_buffer(DATASET_PATHS_SAFE)
    print(f"[mixed]  paths: {[str(p) for p in DATASET_PATHS_MIXED]}")
    mixed  = make_replay_buffer(DATASET_PATHS_MIXED)
    print(f"  expert: {expert['size']} steps")
    print(f"  safe  : {safe['size']} steps")
    print(f"  mixed : {mixed['size']} steps  ({len(mixed['init_idx'])} init obs)")

    scan_dim   = expert['scan'].shape[1]
    state_dim  = expert['state'].shape[1]
    action_dim = expert['action'].shape[1]
    print(f"scan_dim={scan_dim} state_dim={state_dim} action_dim={action_dim}")

    actor_def   = GaussianPolicy(action_dim=action_dim,
                                   scan_mode=SCAN_MODE,
                                   n_sparse_k=N_SCAN_SPARSE_K)
    nu_def      = StateValue(scan_mode=SCAN_MODE, n_sparse_k=N_SCAN_SPARSE_K)
    disc_def    = DiscriminatorNetwork(scan_mode=SCAN_MODE,
                                          n_sparse_k=N_SCAN_SPARSE_K)
    uc_disc_def = DiscriminatorNetwork(scan_mode=SCAN_MODE,
                                          n_sparse_k=N_SCAN_SPARSE_K)

    dummy_scan  = jnp.zeros((1, scan_dim),  dtype=jnp.float32)
    dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
    dummy_act   = jnp.zeros((1, action_dim), dtype=jnp.float32)

    jax_key, *keys = jax.random.split(jax_key, 5)
    actor_params = actor_def.init(keys[0],   dummy_scan, dummy_state)['params']
    nu_params    = nu_def.init(keys[1],      dummy_scan, dummy_state)['params']
    disc_params  = disc_def.init(keys[2],    dummy_scan, dummy_state, dummy_act)['params']
    uc_params    = uc_disc_def.init(keys[3], dummy_scan, dummy_state, dummy_act)['params']

    actor_state = train_state.TrainState.create(
        apply_fn=actor_def.apply, params=actor_params,
        tx=optax.adam(learning_rate=LR_ACTOR))
    nu_state    = train_state.TrainState.create(
        apply_fn=nu_def.apply, params=nu_params,
        tx=optax.adam(learning_rate=LR_NU))
    disc_state  = train_state.TrainState.create(
        apply_fn=disc_def.apply, params=disc_params,
        tx=optax.adam(learning_rate=LR_DISC))
    uc_disc_state = train_state.TrainState.create(
        apply_fn=uc_disc_def.apply, params=uc_params,
        tx=optax.adam(learning_rate=LR_DISC))
    lambda_state = train_state.TrainState.create(
        apply_fn=lambda p, x=None: p['value'],
        params={'value': jnp.array(LAMBDA_INIT, dtype=jnp.float32)},
        tx=optax.adam(learning_rate=LR_LAMBDA))

    n_actor = sum(p.size for p in jax.tree_util.tree_leaves(actor_params))
    n_nu    = sum(p.size for p in jax.tree_util.tree_leaves(nu_params))
    n_disc  = sum(p.size for p in jax.tree_util.tree_leaves(disc_params))
    print(f"Actor: {n_actor:,}  nu: {n_nu:,}  disc: {n_disc:,}  uc_disc: {n_disc:,}")

    disc_step = make_disc_step(disc_def.apply, uc_disc_def.apply,
                                 DISC_GP_WEIGHT, UC_DISC_GP_WEIGHT,
                                 scan_noise_std=_SCAN_NOISE_EFF,
                                 state_noise_std=_STATE_NOISE_EFF)
    main_step = make_main_step(actor_def.apply, nu_def.apply,
                                 disc_def.apply, uc_disc_def.apply,
                                 GAMMA, ALPHA, NU_GP_WEIGHT,
                                 scan_noise_std=_SCAN_NOISE_EFF,
                                 state_noise_std=_STATE_NOISE_EFF)

    suffix = ''
    if SCAN_MODE == 'none':     suffix += '_noscan'
    elif SCAN_MODE == 'sparse': suffix += f'_sparse{N_SCAN_SPARSE_K}'
    if USE_INPUT_NOISE: suffix += '_noise'
    out_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + suffix) if suffix else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path  = out_dir / 'policy_best.pkl'
    final_path = out_dir / 'policy_final.pkl'

    # ----- Pretrain disc + uc_disc -----
    print(f"\n=== Pretrain discriminators ({PRETRAIN_DISC_STEPS} steps) ===")
    for u in range(PRETRAIN_DISC_STEPS):
        e = random_batch(expert, BATCH_SIZE, rng)
        m = random_batch(mixed,  BATCH_SIZE, rng)
        s = random_batch(safe,   BATCH_SIZE, rng)
        jax_key, k = jax.random.split(jax_key)
        disc_state, uc_disc_state, info = disc_step(
            disc_state, uc_disc_state,
            jnp.asarray(e['scan']), jnp.asarray(e['state']), jnp.asarray(e['action']),
            jnp.asarray(m['scan']), jnp.asarray(m['state']), jnp.asarray(m['action']),
            jnp.asarray(s['scan']), jnp.asarray(s['state']), jnp.asarray(s['action']),
            k,
        )
        if (u + 1) % max(1, PRETRAIN_DISC_STEPS // 10) == 0:
            print(f"  pretrain {u+1:5d}/{PRETRAIN_DISC_STEPS}: "
                  f"disc total={float(info['disc_loss']):.3f} "
                  f"(BCE={float(info['disc_bce']):.3f} "
                  f"GP={float(info['disc_gp']):.3f})  "
                  f"uc_disc total={float(info['uc_disc_loss']):.3f} "
                  f"(BCE={float(info['uc_disc_bce']):.3f} "
                  f"GP={float(info['uc_disc_gp']):.3f})")

    threshold = compute_threshold(uc_disc_state, safe, DISC_QUANTILE)
    print(f"=== Initial threshold (q={DISC_QUANTILE}): {threshold:.4f} ===")

    # ----- Training loop -----
    history = []
    eval_history = []
    best_score = float('-inf')
    eval_env, eval_predict = make_eval_env_and_predict(
        actor_def.apply, EVAL_LAYOUT_PATH, EVAL_MAX_STEPS,
        n_beams_default=scan_dim,
    )
    print(f"\n=== Eval env loaded — rollout best select "
          f"(every {EVAL_EVERY} steps, {EVAL_EPISODES} eps) ===")

    print(f"\n=== Training ({TOTAL_STEPS} steps) ===")
    for step in range(1, TOTAL_STEPS + 1):
        if PRIOR_LEARNING_DURING_TRAIN:
            for _ in range(DISC_STEPS_PER_MAIN):
                e = random_batch(expert, BATCH_SIZE, rng)
                m = random_batch(mixed,  BATCH_SIZE, rng)
                s = random_batch(safe,   BATCH_SIZE, rng)
                jax_key, k = jax.random.split(jax_key)
                disc_state, uc_disc_state, _ = disc_step(
                    disc_state, uc_disc_state,
                    jnp.asarray(e['scan']), jnp.asarray(e['state']), jnp.asarray(e['action']),
                    jnp.asarray(m['scan']), jnp.asarray(m['state']), jnp.asarray(m['action']),
                    jnp.asarray(s['scan']), jnp.asarray(s['state']), jnp.asarray(s['action']),
                    k,
                )

        init_scan, init_state = random_init_batch(mixed, BATCH_SIZE, rng)
        mb         = random_batch(mixed, BATCH_SIZE, rng)
        jax_key, k = jax.random.split(jax_key)
        actor_state, nu_state, lambda_state, info = main_step(
            actor_state, nu_state, lambda_state,
            disc_state, uc_disc_state,
            jnp.asarray(init_scan),  jnp.asarray(init_state),
            jnp.asarray(mb['scan']), jnp.asarray(mb['state']),
            jnp.asarray(mb['action']),
            jnp.asarray(mb['next_scan']), jnp.asarray(mb['next_state']),
            jnp.asarray(mb['done']),
            jnp.asarray(threshold, dtype=jnp.float32),
            k,
        )

        if (PRIOR_LEARNING_DURING_TRAIN
                and step % THRESHOLD_UPDATE_FREQ == 0):
            threshold = compute_threshold(uc_disc_state, safe, DISC_QUANTILE)

        if step % LOG_EVERY == 0 or step == 1:
            log_std_now = np.asarray(actor_state.params['log_std']).tolist()
            print(f"step {step:6d}/{TOTAL_STEPS}  "
                  f"policy={float(info['policy_loss']):+.3f}  "
                  f"nu={float(info['nu_loss']):+.3f}  "
                  f"lambda={float(info['lambda']):+.3f}  "
                  f"E[exp(A-1)]={float(info['E_exp_A_total']):.3f}  "
                  f"safe={float(info['safe_rate']):.2f}  tau={threshold:.3f}  "
                  f"log_std=[{log_std_now[0]:+.2f},{log_std_now[1]:+.2f},{log_std_now[2]:+.2f}]")
            history.append({
                'step': step,
                'policy_loss':   float(info['policy_loss']),
                'nu_loss':       float(info['nu_loss']),
                'lambda_loss':   float(info['lambda_loss']),
                'lambda':        float(info['lambda']),
                'init_nu':       float(info['init_nu']),
                'safe_rate':     float(info['safe_rate']),
                'E_exp_A_total': float(info['E_exp_A_total']),
                'E_exp_A_safe':  float(info['E_exp_A_safe']),
                'threshold':     threshold,
                'log_std':       log_std_now,
            })

        # Rollout-based evaluation and best-policy save.
        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
            _, score, esum = evaluate_policy(
                eval_env, eval_predict, actor_state.params,
                EVAL_EPISODES, EVAL_MAX_STEPS,
                perturb_xy=EVAL_INIT_PERTURB_XY,
                perturb_yaw=EVAL_INIT_PERTURB_YAW,
            )
            esum_log = {**esum, 'step': step}
            eval_history.append(esum_log)
            improved = score > best_score
            tag = ' *NEW BEST*' if improved else ''
            print(f"  [eval @ {step}] score={score:+.3f}  "
                   f"succ={esum['success']}/{esum['episodes']}  "
                   f"coll={esum['collide']}  sv={esum['safety_viol']}  "
                   f"end_dist={esum['avg_end_dist']:.2f}m  "
                   f"avg_len={esum['avg_length']:.0f}{tag}")
            if improved:
                best_score = score
                with open(best_path, 'wb') as f:
                    pickle.dump({
                        'params':         jax.tree_util.tree_map(np.asarray, actor_state.params),
                        'nu_params':      jax.tree_util.tree_map(np.asarray, nu_state.params),
                        'disc_params':    jax.tree_util.tree_map(np.asarray, disc_state.params),
                        'uc_disc_params': jax.tree_util.tree_map(np.asarray, uc_disc_state.params),
                        'lambda':         float(lambda_state.params['value']),
                        'threshold':      threshold, 'step': step,
                        'scan_mode':      SCAN_MODE,
                        'n_sparse_k':     N_SCAN_SPARSE_K,
                        'eval_score':     score,
                        'eval_summary':   esum,
                    }, f)

        if step % SAVE_EVERY == 0:
            ckpt_path = out_dir / f'policy_step{step}.pkl'
            with open(ckpt_path, 'wb') as f:
                pickle.dump({
                    'params':         jax.tree_util.tree_map(np.asarray, actor_state.params),
                    'nu_params':      jax.tree_util.tree_map(np.asarray, nu_state.params),
                    'disc_params':    jax.tree_util.tree_map(np.asarray, disc_state.params),
                    'uc_disc_params': jax.tree_util.tree_map(np.asarray, uc_disc_state.params),
                    'lambda':         float(lambda_state.params['value']),
                    'threshold':      threshold, 'step': step,
                    'scan_mode':      SCAN_MODE,
                    'n_sparse_k':     N_SCAN_SPARSE_K,
                }, f)

    with open(final_path, 'wb') as f:
        pickle.dump({
            'params':         jax.tree_util.tree_map(np.asarray, actor_state.params),
            'nu_params':      jax.tree_util.tree_map(np.asarray, nu_state.params),
            'disc_params':    jax.tree_util.tree_map(np.asarray, disc_state.params),
            'uc_disc_params': jax.tree_util.tree_map(np.asarray, uc_disc_state.params),
            'lambda':         float(lambda_state.params['value']),
            'threshold':      threshold,
            'scan_mode':      SCAN_MODE,
            'n_sparse_k':     N_SCAN_SPARSE_K,
        }, f)

    meta = {
        'algorithm': 'DexDICE',
        'config': {
            'scan_mode': SCAN_MODE, 'n_sparse_k': N_SCAN_SPARSE_K,
            'seed': SEED,
            'use_input_noise': USE_INPUT_NOISE,
            'scan_noise_std':  TRAIN_SCAN_NOISE_STD if USE_INPUT_NOISE else 0.0,
            'state_noise_std': TRAIN_STATE_NOISE_STD if USE_INPUT_NOISE else 0.0,
            'total_steps': TOTAL_STEPS, 'batch_size': BATCH_SIZE,
            'lr_actor': LR_ACTOR, 'lr_nu': LR_NU,
            'lr_disc': LR_DISC, 'lr_lambda': LR_LAMBDA,
            'gamma': GAMMA, 'alpha': ALPHA, 'lambda_init': LAMBDA_INIT,
            'disc_gp_weight': DISC_GP_WEIGHT,
            'uc_disc_gp_weight': UC_DISC_GP_WEIGHT,
            'nu_gp_weight': NU_GP_WEIGHT,
            'disc_steps_per_main': DISC_STEPS_PER_MAIN,
            'prior_learning_during_train': PRIOR_LEARNING_DURING_TRAIN,
            'pretrain_disc_steps': PRETRAIN_DISC_STEPS,
            'disc_quantile': DISC_QUANTILE,
            'threshold_update_freq': THRESHOLD_UPDATE_FREQ,
            'state_feat_dim': STATE_FEAT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'disc_hidden': list(DISC_HIDDEN), 'nu_hidden': list(NU_HIDDEN),
            'log_std_init': LOG_STD_INIT, 'log_std_clamp': list(LOG_STD_CLAMP),
        },
        'datasets': {
            'expert': expert['info'],
            'safe':   safe['info'],
            'mixed':  mixed['info'],
        },
        'shapes': {'scan_dim': scan_dim, 'state_dim': state_dim, 'action_dim': action_dim},
        'final_threshold': threshold,
        'final_lambda':    float(lambda_state.params['value']),
        'best_eval_score': best_score,
        'eval_config': {
            'every':            EVAL_EVERY,
            'episodes':         EVAL_EPISODES,
            'max_steps':        EVAL_MAX_STEPS,
            'layout':           EVAL_LAYOUT_PATH,
            'w_collide':        EVAL_W_COLLIDE,
            'w_sv':             EVAL_W_SV,
            'w_end_dist':       EVAL_W_END_DIST,
            'init_perturb_xy':  EVAL_INIT_PERTURB_XY,
            'init_perturb_yaw': EVAL_INIT_PERTURB_YAW,
        },
        'eval_history':  eval_history,
        'history':       history,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. saved best  -> {best_path}")
    print(f"      saved final -> {final_path}")
    print(f"      saved meta  -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
