#!/usr/bin/env python3

"""DSRL offline dataset preprocessing / trajectory classification script.

This script:
- loads an offline dataset from DSRL (and related env wrappers),
- splits it into trajectories,
- classifies trajectories by safety (cost) and performance (return) thresholds,
- saves selected subsets + summary statistics.

Edits in this file are limited to comment/docstring cleanup and small hygiene
improvements (no algorithmic behavior changes).
"""

import dsrl

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pickle5 as pickle
import yaml

BULLET_ENVS = ['SafetyAntCircle-v0', 'SafetyAntRun-v0', 'SafetyBallCircle-v0',
               'SafetyBallRun-v0', 'SafetyCarCircle-v0', 'SafetyCarRun-v0',
               'SafetyDroneCircle-v0', 'SafetyDroneRun-v0']
def split_into_trajectories(data):
    done_flags = np.zeros(len(data["rewards"]), dtype=bool)
    if data["terminals"] is not None:
        done_flags |= data["terminals"].astype(bool)
    if data["timeouts"] is not None:
        done_flags |= data["timeouts"].astype(bool)

    done_indices = np.where(done_flags)[0]
    if done_indices.size == 0:
        raise ValueError("No True values found in terminals/timeouts.")

    trajectories = []
    start = 0
    for end in done_indices:
        end += 1  # include the terminal step
        traj = {
            "observations": data["observations"][start:end],
            "actions": data["actions"][start:end],
            "rewards": data["rewards"][start:end],
            "costs": data["costs"][start:end],
            "next_observations": data["next_observations"][start:end],
            "terminals": done_flags[start:end].astype(bool),
        }
        trajectories.append(traj)
        start = end

    if start != len(done_flags):
        raise ValueError("Not all steps are included in trajectories. Please check the done flag.")
    return trajectories


def load_thresholds(env_name: str, task_info_path: str = "./configs/dsrl_task_info.yaml") -> Tuple[float, float]:
    """Load safety (cost) and performance (return) thresholds for an environment."""
    try:
        with open(task_info_path, 'r') as f:
            task_info = yaml.safe_load(f)
        print(env_name in task_info['environments'])
        if 'environments' in task_info and env_name in task_info['environments']:
            env_config = task_info['environments'][env_name]
            cost_threshold = env_config.get('cost_threshold', 20.0)
            return_threshold = env_config.get('reward_threshold', 0.0)
            return cost_threshold, return_threshold
        else:
            print(f"Warning: Environment {env_name} not found in {task_info_path}")
            return 40.0, 0.0
    except Exception as e:
        print(f"Warning: Could not load thresholds from {task_info_path}: {e}")
        return 40.0, 0.0


def calculate_trajectory_metrics(trajectory: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate return and cost for a trajectory."""
    return_val = np.sum(trajectory['rewards'])
    cost_val = np.sum(trajectory['costs'])
    return return_val, cost_val


def classify_trajectories(trajectories: List[Dict[str, Any]], 
                        cost_threshold: float, 
                        return_threshold: float) -> Dict[str, List[int]]:
    """
    Classify trajectories into safe and performant sets.
    
    Returns:
        Dictionary with trajectory indices for each category:
        - 'safe_only': trajectories that are safe but not performant
        - 'performant_only': trajectories that are performant but not safe  
        - 'both': trajectories that are both safe and performant
        - 'neither': trajectories that are neither safe nor performant
    """
    print(f"Classifying trajectories with cost_threshold={cost_threshold}, return_threshold={return_threshold}")
    
    safe_only = []
    performant_only = []
    both = []
    neither = []
    
    for idx, trajectory in enumerate(trajectories):
        return_val, cost_val = calculate_trajectory_metrics(trajectory)
        
        is_safe = cost_val <= cost_threshold
        is_performant = return_val >= return_threshold
        
        if is_safe and is_performant:
            both.append(idx)
        elif is_safe and not is_performant:
            safe_only.append(idx)
        elif not is_safe and is_performant:
            performant_only.append(idx)
        elif not is_safe and not is_performant:
            neither.append(idx)
        else:
            neither.append(idx)
    
    print(f"Classification results:")
    print(f"  Safe only: {len(safe_only)}")
    print(f"  Performant only: {len(performant_only)}")
    print(f"  Both safe & performant: {len(both)}")
    print(f"  unsafe & not performant: {len(neither)}")
    
    return {
        'safe_only': safe_only,
        'performant_only': performant_only,
        'both': both,
        'neither': neither
    }

def create_final_sets(classified: Dict[str, List[int]], 
                     both_split: Tuple[List[int], List[int]],
                     random_seed: int = 42) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Create final safe and performant trajectory sets.
    
    Returns:
        Tuple of (safe_indices, performant_indices, tracking_info)
    """
    safe_half, performant_half = both_split
    
    safe_indices = classified['safe_only'] + safe_half
    performant_indices = classified['performant_only'] + performant_half
    
    tracking_info = {
        'safe_only_count': len(classified['safe_only']),
        'performant_only_count': len(classified['performant_only']),
        'both_safe_half_count': len(safe_half),
        'both_performant_half_count': len(performant_half),
        'safe_only_indices': classified['safe_only'],
        'performant_only_indices': classified['performant_only'],
        'both_safe_half_indices': safe_half,
        'both_performant_half_indices': performant_half,
        'total_safe': len(safe_indices),
        'total_performant': len(performant_indices)
    }
    
    print(f"Final set sizes:")
    print(f"  Safe set: {len(safe_indices)} trajectories")
    print(f"  Performant set: {len(performant_indices)} trajectories")
    
    return safe_indices, performant_indices, tracking_info


def shuffle_with_demo_first(trajectory_indices: List[int], 
                           trajectories: List[Dict[str, Any]],
                           cost_threshold: float,
                           return_threshold: float,
                           is_safe_set: bool,
                           random_seed: int = 42) -> List[int]:
    """
    Shuffle trajectory indices with ONLY ONE safe & performant demo first.
    
    Args:
        trajectory_indices: List of trajectory indices to shuffle
        trajectories: All trajectories
        cost_threshold: Cost threshold for safety
        return_threshold: Return threshold for performance
        is_safe_set: True if this is the safe set, False if performant set
        random_seed: Random seed for reproducibility
    
    Returns:
        Shuffled list of indices with ONE demo first, rest randomly shuffled
    """
    if not trajectory_indices:
        return []
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    demo_indices = []
    regular_indices = []
    
    for idx in trajectory_indices:
        trajectory = trajectories[idx]
        return_val, cost_val = calculate_trajectory_metrics(trajectory)
        
        is_safe = cost_val <= cost_threshold
        is_performant = return_val >= return_threshold
        
        if is_safe and is_performant:
            demo_indices.append(idx)
        else:
            regular_indices.append(idx)
    
    # Shuffle all remaining trajectories, leaving exactly one demo at the front (if present).
    all_other_indices = demo_indices[1:] + regular_indices
    random.shuffle(all_other_indices)
    
    if demo_indices:
        final_indices = [demo_indices[0]] + all_other_indices
        print(f"  {'Safe' if is_safe_set else 'Performant'} set shuffling:")
        print(f"    First demo trajectory (safe & performant): 1")
        print(f"    Other trajectories (randomly shuffled): {len(all_other_indices)}")
    else:
        final_indices = all_other_indices
        print(f"  {'Safe' if is_safe_set else 'Performant'} set shuffling:")
        print(f"    No demo trajectories found")
        print(f"    All trajectories (randomly shuffled): {len(all_other_indices)}")
    
    return final_indices


def calculate_trajectory_statistics(trajectories: List[Dict[str, Any]], 
                                   cost_threshold: float, 
                                   return_threshold: float, 
                                   env_name: str = None) -> Dict[str, Any]:
    """Calculate comprehensive statistics for trajectories."""
    if not trajectories:
        return {
            'total_trajectories': 0,
            'safe_trajectories': 0,
            'high_return_trajectories': 0,
            'safe_high_return_trajectories': 0,
            'safe_ratio': 0.0,
            'high_return_ratio': 0.0,
            'safe_high_return_ratio': 0.0,
            'mean_return': 0.0,
            'mean_cost': 0.0,
            'mean_length': 0.0,
            'feasible_return': 0.0,
            'violation_rate': 0.0
        }
    
    returns = []
    costs = []
    lengths = []
    feasible_returns = []
    safe_trajectories = 0
    high_return_trajectories = 0
    safe_high_return_trajectories = 0
    total_transitions = 0
    
    for traj in trajectories:
        traj_return = np.sum(traj['rewards'])
        traj_cost = np.sum(traj['costs'])
        traj_length = len(traj['observations'])
        
        returns.append(traj_return)
        costs.append(traj_cost)
        lengths.append(traj_length)
        total_transitions += traj_length
        
        feasible_return = 0.0
        cumulative_cost = 0.0
        
        for i, (reward, cost) in enumerate(zip(traj['rewards'], traj['costs'])):
            cumulative_cost += cost
            if cumulative_cost <= cost_threshold:
                feasible_return += reward
            else:
                break
        
        feasible_returns.append(feasible_return)
        
        is_safe = traj_cost <= cost_threshold
        is_high_return = traj_return >= return_threshold
        
        if is_safe:
            safe_trajectories += 1
        
        if is_high_return:
            high_return_trajectories += 1
        if is_safe and is_high_return:
            safe_high_return_trajectories += 1
    
    total_trajectories = len(trajectories)
    violation_rate = (total_trajectories - safe_trajectories) / total_trajectories if total_trajectories > 0 else 0.0
    
    return {
        # Basic info (similar to existing *-stats.yaml files)
        'env_name': env_name if env_name is not None else 'Unknown',
        'num_episodes': int(total_trajectories),
        'total_transitions': int(total_transitions),
        
        'average_return': float(np.mean(returns) if returns else 0.0),
        'std_return': float(np.std(returns) if returns else 0.0),
        'min_return': float(np.min(returns) if returns else 0.0),
        'max_return': float(np.max(returns) if returns else 0.0),
        
        'average_cost': float(np.mean(costs) if costs else 0.0),
        'std_cost': float(np.std(costs) if costs else 0.0),
        'min_cost': float(np.min(costs) if costs else 0.0),
        'max_cost': float(np.max(costs) if costs else 0.0),
        
        'average_episode_length': float(np.mean(lengths) if lengths else 0.0),
        'std_episode_length': float(np.std(lengths) if lengths else 0.0),
        'min_episode_length': float(np.min(lengths) if lengths else 0.0),
        'max_episode_length': float(np.max(lengths) if lengths else 0.0),
        
        'average_return_until_violation': float(np.mean(feasible_returns) if feasible_returns else 0.0),
        'std_return_until_violation': float(np.std(feasible_returns) if feasible_returns else 0.0),
        'min_return_until_violation': float(np.min(feasible_returns) if feasible_returns else 0.0),
        'max_return_until_violation': float(np.max(feasible_returns) if feasible_returns else 0.0),
        
        'violation_rate': float(violation_rate),
        'safe_trajectories': int(safe_trajectories),
        'safe_ratio': float(safe_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        
        'high_return_trajectories': int(high_return_trajectories),
        'high_return_ratio': float(high_return_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        'safe_high_return_trajectories': int(safe_high_return_trajectories),
        'safe_high_return_ratio': float(safe_high_return_trajectories / total_trajectories if total_trajectories > 0 else 0.0),
        
        'cost_threshold': float(cost_threshold),
        'return_threshold': float(return_threshold),
        
        # Legacy fields for compatibility
        'total_trajectories': int(total_trajectories),
        'mean_return': float(np.mean(returns) if returns else 0.0),
        'mean_cost': float(np.mean(costs) if costs else 0.0),
        'mean_length': float(np.mean(lengths) if lengths else 0.0),
        'feasible_return': float(np.mean(feasible_returns) if feasible_returns else 0.0)
    }


def add_trajectory_ids(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add trajectory IDs based on original order to each trajectory."""
    for i, traj in enumerate(trajectories):
        traj['trajectory_id'] = i
    return trajectories


def save_trajectories(trajectories: List[Dict[str, Any]], 
                     indices: List[int], 
                     output_path: str,
                     tracking_info: Dict[str, Any] = None) -> None:
    """Save selected trajectories to a pickle file."""
    selected_trajectories = [trajectories[i] for i in indices]
    
    # Add tracking information to the first trajectory if provided
    if tracking_info and selected_trajectories:
        selected_trajectories[0]['_tracking_info'] = tracking_info
    
    with open(output_path, 'wb') as f:
        pickle.dump(selected_trajectories, f)
    
    print(f"Saved {len(selected_trajectories)} trajectories to: {output_path}")




def save_statistics(stats: Dict[str, Any], 
                   output_dir: str, 
                   dataset_name: str) -> None:
    """Save trajectory statistics to a YAML file."""
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    stats_file = os.path.join(stats_dir, f'{dataset_name}-stats.yaml')
    
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved statistics to: {stats_file}")


def shuffle_trajectories(indices: List[Any], random_seed : int=42) -> List[Any]:
    """Shuffle unsafe trajectories randomly."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    shuffled = indices.copy()
    random.shuffle(shuffled)
    return shuffled

def shuffle_all_trajectories(trajectories: List[Dict[str, Any]], 
                           random_seed: int = 42) -> List[Dict[str, Any]]:
    """Shuffle all trajectories randomly."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    shuffled = trajectories.copy()
    random.shuffle(shuffled)
    
    print(f"Shuffled all {len(shuffled)} trajectories")
    return shuffled


def main():
    parser = argparse.ArgumentParser(description='Classify trajectories from dsrl')
    parser.add_argument('--task', type=str, default='SafetyPointCircle1-v0',
                        choices=['SafetyPointCircle1-v0', 'SafetyPointCircle2-v0', 'SafetyPointGoal1-v0', 
                        'SafetyPointGoal2-v0', 'SafetyPointButton1-v0', 'SafetyPointButton2-v0',
                        'SafetyAntVelocity-v1', 'SafetyHopperVelocity-v1', 'SafetyWalker2dVelocity-v1', 'SafetyHalfCheetahVelocity-v1', 
                        'SafeMetaDrive-hardsparse-v0', 'SafeMetaDrive-mediummean-v0','SafeMetaDrive-easydense-v0',
                        'SafetyAntCircle-v0', 'SafetyAntRun-v0', 'SafetyBallCircle-v0', 'SafetyBallRun-v0',
                        'SafetyCarCircle-v0', 'SafetyCarRun-v0', 'SafetyDroneCircle-v0', 'SafetyDroneRun-v0'],
                        help='Task name')
    parser.add_argument('--class_type', type=str, default='safe',
                       choices=['safe', 'performant', 'unsafe', 'shuffle', 'safe-performant'],
                       help='Type of classification: safe, performant, unsafe, shuffle')
    parser.add_argument('--output_dir', type=str, 
                       default='./dataset',
                       help='Output directory')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--task_info', type=str, default='./configs/dsrl_task_info.yaml',
                       help='Path to task_info.yaml file')
    
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    env_name = args.task
    if env_name is None:
        env_name = "SafetyPointCircle1"

    # Download the offline dataset from DSRL
    if env_name.startswith('SafeMetaDrive'):
        import gym as gym_old  # MetaDrive environments
        offline_env_name = env_name.replace('SafeMetaDrive-', 'OfflineMetadrive-')
        env = gym_old.make(offline_env_name)
    elif env_name in BULLET_ENVS: 
        import gymnasium as gym  # bullet safety gym
        offline_env_name = env_name.replace('Safety', 'Offline')
        env = gym.make(offline_env_name)
    else:
        import safety_gymnasium as safety_gym  # safety-gymnasium
        offline_env_name = env_name.replace('Safety', 'Offline')
        env = safety_gym.make(offline_env_name)
    data = env.get_dataset()

    output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)

    trajectories = split_into_trajectories(data)

    
    print(f"Environment: {env_name}")
    print(f"Output directory: {output_dir}")
    print(f"Classification type: {args.class_type}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 60)
    
    trajectories = add_trajectory_ids(trajectories)
    
    # Process based on class_type
    if args.class_type == 'shuffle':
        # Simply shuffle all trajectories
        print("Shuffling all trajectories...")
        all_shuffled = shuffle_all_trajectories(trajectories, args.random_seed)
        shuffled_output = os.path.join(output_dir, 'dsrl-shuffled-v0.pkl')
        
        with open(shuffled_output, 'wb') as f:
            pickle.dump(all_shuffled, f)
        
        print(f"Saved {len(all_shuffled)} shuffled trajectories to: {shuffled_output}")
        
        # Calculate and save statistics for shuffled set
        cost_threshold, return_threshold = load_thresholds(env_name, args.task_info)
        shuffled_stats = calculate_trajectory_statistics(all_shuffled, cost_threshold, return_threshold, env_name)
        save_statistics(shuffled_stats, output_dir, 'dsrl-shuffled-v0')
        
        print("=" * 60)
        print("SUMMARY:")
        print(f"Total trajectories processed: {len(trajectories)}")
        print(f"Shuffled set size: {len(all_shuffled)}")
        print(f"Output files saved in: {output_dir}")
        print("  - dsrl-shuffled-v0.pkl")
        print("Statistics saved in: stats/")
        print("  - dsrl-shuffled-v0-stats.yaml")
        
    else:
        # Load thresholds for classification
        cost_threshold, return_threshold = load_thresholds(env_name, args.task_info)
        print(f"Using thresholds: cost_threshold={cost_threshold}, return_threshold={return_threshold}")
        print("=" * 60)
        
        # Classify trajectories
        classified = classify_trajectories(trajectories, cost_threshold, return_threshold)
        
        if args.class_type == 'safe':
            # Include all safe trajectories (safe_only + both)
            print("Creating safe trajectory set (includes all safe trajectories)...")
            safe_indices = classified['safe_only'] + classified['both']
            
            # Shuffle with demo first
            safe_shuffled = shuffle_with_demo_first(safe_indices, trajectories, cost_threshold, return_threshold, 
                                                  is_safe_set=True, random_seed=args.random_seed)
            
            # Save safe set
            safe_output = os.path.join(output_dir, 'dsrl-safe-v0.pkl')
            save_trajectories(trajectories, safe_shuffled, safe_output)
            
            # Calculate and save statistics
            safe_trajectories = [trajectories[i] for i in safe_shuffled]
            safe_stats = calculate_trajectory_statistics(safe_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(safe_stats, output_dir, 'dsrl-safe-v0')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Safe set size: {len(safe_shuffled)}")
            print(f"  - Safe only: {len(classified['safe_only'])}")
            print(f"  - Both safe & performant: {len(classified['both'])}")
            print(f"Output files saved in: {output_dir}")
            print("  - dsrl-safe-v0.pkl")
            print("Statistics saved in: stats/")
            print("  - dsrl-safe-v0-stats.yaml")
            
        elif args.class_type == 'performant':
            # Include all performant trajectories (performant_only + both)
            print("Creating performant trajectory set (includes all performant trajectories)...")
            performant_indices = classified['performant_only'] + classified['both']
            
            # Shuffle with demo first
            performant_shuffled = shuffle_with_demo_first(performant_indices, trajectories, cost_threshold, return_threshold,
                                                        is_safe_set=False, random_seed=args.random_seed)
            
            # Save performant set
            performant_output = os.path.join(output_dir, 'dsrl-performance-v0.pkl')
            save_trajectories(trajectories, performant_shuffled, performant_output)
            
            # Calculate and save statistics
            performant_trajectories = [trajectories[i] for i in performant_shuffled]
            performant_stats = calculate_trajectory_statistics(performant_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(performant_stats, output_dir, 'dsrl-performance-v0')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Performant set size: {len(performant_shuffled)}")
            print(f"  - Performant only: {len(classified['performant_only'])}")
            print(f"  - Both safe & performant: {len(classified['both'])}")
            print(f"Output files saved in: {output_dir}")
            print("  - dsrl-performance-v0.pkl")
            print("Statistics saved in: stats/")
            print("  - dsrl-performance-v0-stats.yaml")
            
        elif args.class_type == 'unsafe':
            print("Creating unsafe trajectory set (trajectories that violate safety constraints)...")
            unsafe_indices = classified['neither'] + classified['performant_only']
            unsafe_indices = shuffle_trajectories(indices = unsafe_indices, random_seed = args.random_seed)
            unsafe_output = os.path.join(output_dir, 'dsrl-nonpreferred-v0.pkl')
            save_trajectories(trajectories, unsafe_indices, unsafe_output)
        
            unsafe_trajectories = [trajectories[i] for i in unsafe_indices]
            unsafe_stats = calculate_trajectory_statistics(unsafe_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(unsafe_stats, output_dir, 'dsrl-nonpreferred-v0')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Unsafe set size: {len(unsafe_indices)}")
            print(f"  - Violating trajectories: {len(classified['neither'])+len(classified['performant_only'])}")
            print(f"Output files saved in: {output_dir}")
            print("  - dsrl-nonpreferred-v0.pkl")
            print("Statistics saved in: stats/")
            print("  - dsrl-nonpreferred-v0-stats.yaml")
        
        elif args.class_type == 'safe-performant':
            print("Creating safe-performant trajectory set (includes all safe and performant trajectories)...")
            safe_performant_indices = classified['both']
            safe_performant_indices = shuffle_trajectories(indices = safe_performant_indices, random_seed = args.random_seed)
            safe_performant_output = os.path.join(output_dir, 'dsrl-safe-performant-v0.pkl')
            save_trajectories(trajectories, safe_performant_indices, safe_performant_output)
            
            safe_performant_trajectories = [trajectories[i] for i in safe_performant_indices]
            safe_performant_stats = calculate_trajectory_statistics(safe_performant_trajectories, cost_threshold, return_threshold, env_name)
            save_statistics(safe_performant_stats, output_dir, 'dsrl-safe-performant-v0')
            
            print("=" * 60)
            print("SUMMARY:")
            print(f"Total trajectories processed: {len(trajectories)}")
            print(f"Safe-performant set size: {len(safe_performant_indices)}")
            print(f"Output files saved in: {output_dir}")
            print("  - dsrl-safe-performant-v0.pkl")
            print("Statistics saved in: stats/")
            print("  - dsrl-safe-performant-v0-stats.yaml")
            
        else:
            raise ValueError(f"Unsupported class_type: {args.class_type}")


if __name__ == "__main__":
    main()