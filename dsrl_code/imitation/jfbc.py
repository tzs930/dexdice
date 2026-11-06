"""Jointly Filtered BC (JFBC): behavioral cloning restricted to samples accepted by
both the safe-support estimator (D_S vs D_U) and the performant-support estimator
(D_P vs D_U) (Section 5 of the paper).
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml

def copy_nn_module(source, target):
    """Copy parameters from `source` module into `target` module (in-place)."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class JFBC(nn.Module):
    """BC trainer with separate expert/safe discriminators and joint thresholding."""
    def __init__(self, policy, env, configs, best_policy=None,
                expert_replay_buffer=None, safe_replay_buffer=None, mixed_replay_buffer=None, n_train=1):
        
        seed = configs['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(JFBC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.expert_replay_buffer = expert_replay_buffer
        self.safe_replay_buffer = safe_replay_buffer
        self.mixed_replay_buffer = mixed_replay_buffer

        self.device = configs['device']
        self.add_absorbing_state = configs['replay_buffer']['use_absorbing_state']
        
        self.n_train = n_train

        self.checkpoint_config = configs.get('checkpoint', {})
        method = configs['method']
        envname = configs['env_id']
        seed = configs['train']['seed']
        dirname = f'{method}_{envname}_{seed}_cid_{configs["cid"]}_pid_{configs["pid"]}'
        self.checkpoint_save_dir = os.path.join(self.checkpoint_config.get('save_dir', './checkpoints'), dirname)
        if not os.path.exists(self.checkpoint_save_dir):
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)
            print(f'** Created checkpoint directory: {self.checkpoint_save_dir}')

        self.checkpoint_save_freq = self.checkpoint_config.get('save_freq', 10000)
        self.max_checkpoints = self.checkpoint_config.get('max_checkpoints', 5)
        self.pid = self.checkpoint_config.get('pid', 0)

        try:
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        except PermissionError:
            self.checkpoint_save_dir = f'./checkpoints_pid_{self.pid}'
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)
            print(f"** Using fallback checkpoint directory: {self.checkpoint_save_dir}")
    
        if self.add_absorbing_state:
            self.obs_dim = env.observation_space.low.size + 1
        else:
            self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=float(configs['train']['lr']))
        
        self.num_eval_iteration = configs['train']['num_evals']
        self.envname = configs['env_id']

        self.use_wandb = configs['use_wandb']
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=configs['wandb']['project'], 
                            entity=configs['wandb']['entity'], 
                            config=configs,
                            settings=configs['wandb']['settings'])
        else:
            self.wandb = None

        self.save_policy_path = configs['train']['save_policy_path']        

        self.disc_hidden_size = configs['train']['disc_hidden_size']
        self.disc_gp_weight = configs['train']['uc_disc_gp_weight']
        self.uc_disc_gp_weight = configs['train']['uc_disc_gp_weight']

        self.expert_disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        self.expert_disc_optimizer = optim.Adam(self.expert_disc_network.parameters(), lr=float(configs['train']['lr_disc']))

        self.safe_disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        self.safe_disc_optimizer = optim.Adam(self.safe_disc_network.parameters(), lr=float(configs['train']['lr_disc']))

        self.safe_disc_threshold = 0.
        self.safe_disc_quantile = float(configs['train']['disc_quantile'])
        self.expert_disc_threshold = 0.
        self.expert_disc_quantile = float(configs['train']['disc_quantile'])
        self.threshold_update_freq = int(configs['train']['threshold_update_freq'])
        self.disc_learning_during_train = configs['train']['disc_learning_during_train']
        
        self.current_expert_disc_gradient_penalty = 0.
        self.current_safe_disc_gradient_penalty = 0.
        
        self.obs_standardize = configs['replay_buffer']['standardize_obs']
        self.act_standardize = configs['replay_buffer']['standardize_act']

        if self.obs_standardize:
            self.obs_mean = self.mixed_replay_buffer.obs_mean
            self.obs_std = self.mixed_replay_buffer.obs_std 
            self.obs_mean_tt = torch.tensor(self.obs_mean, device=self.device)
            self.obs_std_tt = torch.tensor(self.obs_std, device=self.device)
        else:
            self.obs_mean = None
            self.obs_std = None
            self.obs_mean_tt = None
            self.obs_std_tt = None
            
        if self.act_standardize:
            self.act_mean = self.mixed_replay_buffer.act_mean
            self.act_std = self.mixed_replay_buffer.act_std
            self.act_mean_tt = torch.tensor(self.mixed_replay_buffer.act_mean, device=self.device)
            self.act_std_tt = torch.tensor(self.mixed_replay_buffer.act_std, device=self.device)
        else:
            self.act_mean = None
            self.act_std = None
            self.act_mean_tt = None
            self.act_std_tt = None

        self.dataset_path = configs['dataset']['dataset_path']
        # Normalization range of the task (max/min episode return of the full dataset, Appendix G.1) and its cost limit,
        # filled in by train.py from configs/dsrl_task_info.yaml. Returns are reported normalized to [0, 1] when available.
        self.max_score = configs['train'].get('max_return', None)
        self.min_score = configs['train'].get('min_return', None)
        self.cost_threshold = configs['train']['cost_threshold']
        
        print(f'** max_score: {self.max_score}, cost_threshold: {self.cost_threshold}, min_score: {self.min_score}')

    def save_checkpoint(self, iteration, max_score):
        import glob
        
        checkpoint_path = os.path.join(self.checkpoint_save_dir, f'checkpoint_pid_{self.pid}_iter_{iteration}.pt')
        
        checkpoint = {
            'iteration': iteration,
            'max_score': max_score,
            'policy_state_dict': self.policy.state_dict(),
            'best_policy_state_dict': self.best_policy.state_dict() if self.best_policy else None,
            'expert_disc_network_state_dict': self.expert_disc_network.state_dict(),
            'safe_disc_network_state_dict': self.safe_disc_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'expert_disc_optimizer_state_dict': self.expert_disc_optimizer.state_dict(),
            'safe_disc_optimizer_state_dict': self.safe_disc_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f'** Checkpoint saved at iteration {iteration}: {checkpoint_path}')
        
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self):
        import glob
        
        checkpoint_pattern = os.path.join(self.checkpoint_save_dir, f'checkpoint_pid_{self.pid}_iter_*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            print(f'** No checkpoint found for pid {self.pid}. Starting from scratch.')
            return 0, -100000.
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_iter_')[1].split('.pt')[0]))
        
        print(f'** Loading checkpoint: {latest_checkpoint}')
        checkpoint = torch.load(latest_checkpoint, weights_only=False, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if self.best_policy and checkpoint['best_policy_state_dict']:
            self.best_policy.load_state_dict(checkpoint['best_policy_state_dict'])

        self.expert_disc_network.load_state_dict(checkpoint['expert_disc_network_state_dict'])
        self.safe_disc_network.load_state_dict(checkpoint['safe_disc_network_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.expert_disc_optimizer.load_state_dict(checkpoint['expert_disc_optimizer_state_dict'])
        self.safe_disc_optimizer.load_state_dict(checkpoint['safe_disc_optimizer_state_dict'])
        
        iteration = checkpoint['iteration'] + 1
        max_score = checkpoint['max_score']
        
        print(f'** Checkpoint loaded: iteration={iteration}, max_score={max_score:.2f}')
        return iteration, max_score
    
    def _cleanup_old_checkpoints(self):
        import glob
        
        checkpoint_pattern = os.path.join(self.checkpoint_save_dir, f'checkpoint_pid_{self.pid}_iter_*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        checkpoint_files.sort(key=lambda x: int(x.split('_iter_')[1].split('.pt')[0]))
        
        for checkpoint_file in checkpoint_files[:-self.max_checkpoints]:
            os.remove(checkpoint_file)
            print(f'** Removed old checkpoint: {checkpoint_file}')
    
    def calculate_gradient_penalty(self, network, expert_s_a, mixed_s_a):
        
        rand_coef = torch.rand(expert_s_a.size(0), 1, device=self.device)
        rand_coef = rand_coef.expand(expert_s_a.size())
        
        interpolated_s_a = rand_coef * expert_s_a + (1 - rand_coef) * mixed_s_a
        interpolated_s_a.requires_grad_(True)
        
        interpolated_output = network(interpolated_s_a)
                        
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated_s_a,
            grad_outputs=torch.ones_like(interpolated_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(-1, gradients.size(-1))
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

        return gradient_penalty
    
    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        
        start_iteration, max_score = self.load_checkpoint()

        safe_n = self.safe_replay_buffer._size
        safe_obs_total = torch.tensor(self.safe_replay_buffer._observations[:safe_n], dtype=torch.float32, device=self.device)
        safe_actions_total = torch.tensor(self.safe_replay_buffer._actions[:safe_n], dtype=torch.float32, device=self.device)
        safe_s_a_total = torch.cat([safe_obs_total, safe_actions_total], dim=-1)

        safe_disc_total = self.safe_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
        safe_disc_threshold = np.quantile(safe_disc_total, self.safe_disc_quantile)
        self.safe_disc_threshold = safe_disc_threshold

        expert_n = self.expert_replay_buffer._size
        expert_obs_total = torch.tensor(self.expert_replay_buffer._observations[:expert_n], dtype=torch.float32, device=self.device)
        expert_actions_total = torch.tensor(self.expert_replay_buffer._actions[:expert_n], dtype=torch.float32, device=self.device)
        expert_s_a_total = torch.cat([expert_obs_total, expert_actions_total], dim=-1)

        expert_disc_total = self.expert_disc_network(expert_s_a_total).reshape(-1).cpu().detach().numpy()
        expert_disc_threshold = np.quantile(expert_disc_total, self.expert_disc_quantile)
        self.expert_disc_threshold = expert_disc_threshold

        for num in range(start_iteration, int(total_iteration)+1):
            self.safe_disc_optimizer.zero_grad()
                    
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)

            safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

            safe_disc = self.safe_disc_network(safe_s_a)
            mixed_disc = self.safe_disc_network(mixed_s_a)

            disc_loss = -torch.log(safe_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()
            
            if self.uc_disc_gp_weight > 0:
                randperm = torch.randperm(batch_size)
                self.current_safe_disc_gradient_penalty = self.calculate_gradient_penalty(self.safe_disc_network, mixed_s_a, mixed_s_a[randperm])
                disc_loss += self.uc_disc_gp_weight * self.current_safe_disc_gradient_penalty
            else:
                self.current_safe_disc_gradient_penalty = 0.

            disc_loss.backward()
            self.safe_disc_optimizer.step()

            self.expert_disc_optimizer.zero_grad()
                    
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)

            expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

            expert_disc = self.expert_disc_network(expert_s_a)
            mixed_disc = self.expert_disc_network(mixed_s_a)

            disc_loss = -torch.log(expert_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()
            
            if self.disc_gp_weight > 0:
                randperm = torch.randperm(batch_size)
                self.current_expert_disc_gradient_penalty = self.calculate_gradient_penalty(self.expert_disc_network, mixed_s_a, mixed_s_a[randperm])
                disc_loss += self.disc_gp_weight * self.current_expert_disc_gradient_penalty
            else:
                self.current_expert_disc_gradient_penalty = 0.

            disc_loss.backward()
            self.expert_disc_optimizer.step()

            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)
            mixed_expert_disc = self.expert_disc_network(mixed_s_a)
            mixed_safe_disc = self.safe_disc_network(mixed_s_a)

            expert_indicator = (mixed_expert_disc > self.expert_disc_threshold).reshape(-1).detach().float()
            safe_indicator = (mixed_safe_disc > self.safe_disc_threshold).reshape(-1).detach().float()

            train_loss = torch.mean(expert_indicator * safe_indicator * -self.policy.log_prob(mixed_obs, mixed_actions))
            
            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num) % self.threshold_update_freq == 0  and self.disc_learning_during_train:
                safe_disc_total = self.safe_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
                safe_disc_threshold = np.quantile(safe_disc_total, self.safe_disc_quantile)
                self.safe_disc_threshold = safe_disc_threshold
                
                expert_disc_total = self.expert_disc_network(expert_s_a_total).reshape(-1).cpu().detach().numpy()
                expert_disc_threshold = np.quantile(expert_disc_total, self.expert_disc_quantile)
                self.expert_disc_threshold = expert_disc_threshold

            if (num) % eval_freq == 0:
                self.policy.eval()
                eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean, eval_feasible_ret_mean, eval_feasible_ret_std = \
                    self.evaluate(self.env, self.policy, num_evaluation=self.num_eval_iteration)

                if self.use_wandb:
                    self.wandb.log({'train/policy_loss':       train_loss.item(), 
                                    'eval/episode_return':     eval_ret_mean,
                                    'eval/episode_cost':       eval_cost_mean,
                                    'eval/violation_rate':     eval_violation_rate,
                                    'eval/episode_length':     eval_length_mean,
                                    'eval/feasible_return':     eval_feasible_ret_mean,
                                    'eval/feasible_return_std': eval_feasible_ret_std,
                               }, step=num+1)

                print(f'** iter{num}: policy_loss={train_loss.item():.2f}, ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}, feasible_ret={eval_feasible_ret_mean:.2f}+-{eval_feasible_ret_std:.2f}, cost={eval_cost_mean:.2f}+-{eval_cost_std:.2f}, violation_rate={eval_violation_rate:.2f}, length={eval_length_mean:.2f}, expert_disc_threshold={self.expert_disc_threshold:.2f}, safe_disc_threshold={self.safe_disc_threshold:.2f}')
                
                if eval_ret_mean > max_score:
                    print(f'** max score record! ')
                    max_score = eval_ret_mean
                    copy_nn_module(self.policy, self.best_policy)
                    
                if num % self.checkpoint_save_freq == 0 or num % eval_freq == 0:
                    self.save_checkpoint(num, max_score)
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
                    
                    
    def evaluate(self, env, policy, num_evaluation=5, deterministic=True):
        rets = []
        costs = []
        lengths = []
        rets_until_violation = []
        cost_threshold = self.cost_threshold
        max_score = self.max_score
        min_score = self.min_score

        for num in range(0, num_evaluation):
            obs_, info_ = env.reset()
            
            done = False
            t = 0
            ret = 0.
            cum_cost = 0.
            violation = False
            ret_until_violation = 0
            
            while not done:
                if self.add_absorbing_state:
                    obs = np.concatenate([obs_, [0.]])
                else:
                    obs = obs_
                    
                if self.obs_standardize:
                    obs = (obs - self.obs_mean[0]) / (self.obs_std[0] + 1e-8)

                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if deterministic:
                    action = policy(obs).mean.cpu().detach().numpy()
                else:
                    action = policy(obs).sample().cpu().detach().numpy()
                
                interactions = env.step(action)
                if len(interactions) == 6:
                    next_obs, rew, cost, terminate, truncated, info = interactions
                else:
                    next_obs, rew, terminate, truncated, info = interactions
                    cost = info['cost']

                ret += rew
                cum_cost += cost
                
                if cum_cost > cost_threshold:
                    violation = True
                if not violation:
                    ret_until_violation += rew
                
                obs_ = next_obs 
                done = terminate or truncated
                    
                t += 1

            if max_score is not None and min_score is not None:
                normalized_ret = (ret - min_score) / (max_score - min_score)
                normalized_ret_until_violation = (ret_until_violation - min_score) / (max_score - min_score)
            else:
                normalized_ret = ret
                normalized_ret_until_violation = ret_until_violation

            rets.append(normalized_ret)
            costs.append(cum_cost / cost_threshold)   # normalized cost: the constraint is satisfied when <= 1
            lengths.append(t)
            rets_until_violation.append(normalized_ret_until_violation)

        violation_rate = np.mean(np.array(costs) > 1.0)

        return np.mean(rets), np.std(rets)/np.sqrt(num_evaluation), np.mean(costs), np.std(costs)/np.sqrt(num_evaluation), violation_rate, np.mean(lengths), np.mean(rets_until_violation), np.std(rets_until_violation)/np.sqrt(num_evaluation)
    
