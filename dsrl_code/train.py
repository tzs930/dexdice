import os
import random

import yaml

import numpy as np
import torch

from imitation.bc import BC
from imitation.fbc import FBC
from imitation.jfbc import JFBC
from imitation.demodice_S import DemoDICE_S
from imitation.demodice_U import DemoDICE_U
from imitation.dwbc_S import DWBC_S
from imitation.dexdice import DexDICE

from argparse import ArgumentParser

from core.policy import TanhGaussianPolicy, GaussianPolicy
from core.replay_buffer import InitObsBuffer, MDPReplayBuffer
from core.preprocess import preprocess_dataset
from rlkit.envs.wrappers import NormalizedBoxEnv

METADRIVE_ENVS = ['SafeMetaDrive-easysparse-v0', 'SafeMetaDrive-easymean-v0', 'SafeMetaDrive-easydense-v0',
                  'SafeMetaDrive-mediumsparse-v0', 'SafeMetaDrive-mediummean-v0', 'SafeMetaDrive-mediumdense-v0',
                  'SafeMetaDrive-hardsparse-v0', 'SafeMetaDrive-hardmean-v0', 'SafeMetaDrive-harddense-v0']
BULLET_ENVS = ['SafetyAntCircle-v0', 'SafetyAntRun-v0', 'SafetyBallCircle-v0', 'SafetyBallRun-v0', 'SafetyDroneCircle-v0', 'SafetyDroneRun-v0', 'SafetyCarCircle-v0', 'SafetyCarRun-v0']

def load_task_info(env_id, task_info_path):
    """Return the task entry of configs/dsrl_task_info.yaml (cost limit and return range), or None if not listed."""
    try:
        with open(task_info_path, 'r') as f:
            task_info = yaml.safe_load(f)
        return task_info['environments'][env_id]
    except (OSError, KeyError, TypeError):
        return None


def train(configs):
    # Seed every RNG before any network or environment is created.
    seed = int(configs['train']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # The evaluation cost limit (20, 40, or 80 in DSRL) and the return-normalization range are task-specific;
    # take them from the task table so that evaluation reports normalized return and normalized cost as in the paper.
    task_entry = load_task_info(configs['env_id'], configs.get('task_info', './configs/dsrl_task_info.yaml'))
    if task_entry is not None:
        configs['train']['cost_threshold'] = float(task_entry['cost_threshold'])
        configs['train']['max_return'] = float(task_entry['max_return'])
        configs['train']['min_return'] = float(task_entry['min_return'])
    else:
        print(f"** Warning: {configs['env_id']} not found in the task table; using cost_threshold={configs['train']['cost_threshold']} "
              f"from the config and reporting unnormalized returns")

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = configs.get('train', {}).get('save_policy_path', './checkpoints')
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'** Created checkpoint directory: {checkpoint_dir}')
    
    if configs['env_id'] in BULLET_ENVS:
        import bullet_safety_gym
        import gymnasium as gym
        env = gym.make(configs['env_id'])
    elif configs['env_id'] in METADRIVE_ENVS:
        import gym as gym_old
        import dsrl.offline_metadrive
        modified_env_id = configs['env_id'].replace('SafeMetaDrive-','OfflineMetadrive-')
        env = gym_old.make(modified_env_id)
    else:
        import safety_gymnasium as safety_gym
        env = safety_gym.make(configs['env_id'])
    
    if configs['replay_buffer']['use_absorbing_state']:
        obs_dim = env.observation_space.low.size + 1
    else:
        obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    # if torch.backends.mps.is_available():
    #     device = 'mps'
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    configs['device'] = device

    
    envname = configs['env_id']
    print(f'** Preprocessing dataset... ({envname})')
    expert_data_dict, n_expert = preprocess_dataset(configs['dataset']['dataset_path'],
                                                   configs['dataset']['expert']['types'], 
                                                   configs['dataset']['expert']['start_indices'], 
                                                   num_rollouts=configs['dataset']['expert']['num_trajs'],
                                                   use_absorbing_state=configs['replay_buffer']['use_absorbing_state'],
                                                   max_episode_len = env.spec.max_episode_steps)
    
    init_obs_list = [expert_data_dict['initial_observations']]
    if configs['dataset']['safe']['types'] is not None:
        if len(configs['dataset']['safe']['types']) == 0:
            safe_data_dict = None
            n_safe = 0
        else:
            safe_data_dict, n_safe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                    configs['dataset']['safe']['types'], 
                                                    configs['dataset']['safe']['start_indices'], 
                                                    num_rollouts=configs['dataset']['safe']['num_trajs'],
                                                    use_absorbing_state=configs['replay_buffer']['use_absorbing_state'],
                                                    max_episode_len = env.spec.max_episode_steps)
            init_obs_list.append(safe_data_dict['initial_observations'])
    else:
        safe_data_dict = None
        n_safe = 0
    
    print(f'** Number of expert data points: {n_expert}')
    print(f'** Number of safe data points: {n_safe}')

    if configs['dataset']['mixed']['types'] is not None:
        if len(configs['dataset']['mixed']['types']) == 0:
            mixed_data_dict = None
            n_mixed = 0
        else:
            mixed_data_dict, n_mixed = preprocess_dataset(configs['dataset']['dataset_path'],
                                                    configs['dataset']['mixed']['types'], 
                                                    configs['dataset']['mixed']['start_indices'], 
                                                    num_rollouts=configs['dataset']['mixed']['num_trajs'],
                                                    use_absorbing_state=configs['replay_buffer']['use_absorbing_state'],
                                                    max_episode_len = env.spec.max_episode_steps)
            init_obs_list.append(mixed_data_dict['initial_observations'])
    else:
        mixed_data_dict = None
        n_mixed = 0

    init_obs_datapoints = np.concatenate(init_obs_list)

    print(f'** num. of expert data points : {n_expert}')
    print(f'** num. of safe data points : {n_safe}')
    print(f'** num. of mixed data points : {n_mixed}')
    n_train = n_expert + n_safe + n_mixed
    print(f'** num. of total train data points : {n_train}')

    expert_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    expert_replay_buffer.add_path(expert_data_dict)

    if safe_data_dict is not None:
        safe_replay_buffer = MDPReplayBuffer(
            configs['replay_buffer']['max_replay_buffer_size'],
            env,
            obs_dim=obs_dim
        )
        
        safe_replay_buffer.add_path(safe_data_dict)

    if mixed_data_dict is not None:
        mixed_replay_buffer = MDPReplayBuffer(
            configs['replay_buffer']['max_replay_buffer_size'],
            env,
            obs_dim=obs_dim
        )
        mixed_replay_buffer.add_path(mixed_data_dict)
    
    init_obs_buffer = InitObsBuffer(
        env, init_obs_datapoints
    )
    
    if configs['replay_buffer']['standardize_obs'] or configs['replay_buffer']['standardize_act']:
        obs_mean, obs_std, act_mean, act_std = expert_replay_buffer.calculate_statistics(
            standardize_obs=configs['replay_buffer']['standardize_obs'],
            standardize_act=configs['replay_buffer']['standardize_act']
        )
        init_obs_buffer.set_statistics(obs_mean, obs_std)

        if safe_data_dict is not None:
            safe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if mixed_data_dict is not None:
            mixed_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
    else:
        obs_mean, obs_std, act_mean, act_std = None, None, None, None
    
    if 'GaussianPolicy' == configs['policy']['class']:
        policy = GaussianPolicy(
                hidden_sizes=configs['policy']['layer_sizes'],
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device
            )
            
        best_policy = GaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )

    elif 'TanhGaussianPolicy' == configs['policy']['class']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )

        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )
    else:
        raise NotImplementedError
    
    if 'BC' == configs['method']:
        trainer = BC(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            replay_buffer=expert_replay_buffer,
            n_train=n_train,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'FBC' == configs['method']:
        trainer = FBC(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            expert_replay_buffer=expert_replay_buffer,
            safe_replay_buffer=safe_replay_buffer,
            mixed_replay_buffer=mixed_replay_buffer,
            n_train=n_train,
        )
        
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'JFBC' == configs['method']:
        trainer = JFBC(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            expert_replay_buffer=expert_replay_buffer,
            safe_replay_buffer=safe_replay_buffer,
            mixed_replay_buffer=mixed_replay_buffer,
            n_train=n_train,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])
    
    elif 'DemoDICE-S' == configs['method']:
        
        trainer = DemoDICE_S(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'DemoDICE-U' == configs['method']:
        trainer = DemoDICE_U(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])
    
    elif 'DWBC-S' == configs['method']:
        trainer = DWBC_S(
            policy = policy,
            env = env,
            configs = configs,
            best_policy = best_policy,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'DexDICE' == configs['method']:
        trainer = DexDICE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])
            

    else: 
        raise NotImplementedError       

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process id; also used as the random seed", default=0, type=int)
    parser.add_argument("--config", help="config file", default="./configs/dexdice.yaml")
    parser.add_argument("--task_info", help="per-task threshold table", default="./configs/dsrl_task_info.yaml")
    args = parser.parse_args()
    pid = args.pid

    configs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    configs['seed'] = pid
    configs['pid'] = pid
    configs['train']['seed'] = pid
    configs['task_info'] = args.task_info

    configs['wandb']['settings'] = {
        'save_code': False,
        'disable_git': True,
        'disable_code': True,
        'x_disable_stats': True,
    }

    train(configs)
