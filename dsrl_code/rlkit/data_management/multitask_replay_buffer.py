import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer as RLKitSimpleReplayBuffer
)
from gym.spaces import Box, Discrete, Tuple


class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            task_indices,
            use_next_obs_in_context,
            sparse_rewards,
            use_ground_truth_context=False,
            ground_truth_tasks=None,
            env_info_sizes=None,
    ):

        if env_info_sizes is None:
            env_info_sizes = {}
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = sparse_rewards
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.use_ground_truth_context = use_ground_truth_context
        self.task_indices = task_indices
        self.ground_truth_tasks = ground_truth_tasks
        if use_ground_truth_context:
            assert ground_truth_tasks is not None
        if sparse_rewards:
            env_info_sizes['sparse_reward'] = 1
        self.task_buffers = dict([(idx, RLKitSimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )) for idx in task_indices])
        self._max_replay_buffer_size = max_replay_buffer_size
        self._env_info_sizes = env_info_sizes

    def create_new_task_buffer(self, task_idx):
        if task_idx in self.task_buffers:
            raise IndexError("task_idx already exists: {}".format(task_idx))
        new_task_buffer = RLKitSimpleReplayBuffer(
            max_replay_buffer_size=self._max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=self._env_info_sizes,
        )
        self.task_buffers[task_idx] = new_task_buffer

    def add_sample(self, task, observation, action, reward, terminal,
                   next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
            observation, action, reward, terminal,
            next_observation, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            try:
                batch = self.task_buffers[task].random_batch(batch_size)
            except KeyError:
                import ipdb; ipdb.set_trace()
                print(task)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def clear_all_buffers(self):
        for buffer in self.task_buffers.values():
            buffer.clear()

    def sample_batch(self, indices, batch_size):
        batches = [self.random_batch(idx, batch_size=batch_size) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [np.concatenate(x, axis=0) for x in unpacked]

        obs, actions, rewards, next_obs, terms = unpacked
        return {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_obs,
            'terminals': terms,
        }

    def sample_context(self, indices, batch_size):
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [
            self.random_batch(
                idx,
                batch_size=batch_size,
                sequence=False)
            for idx in indices
        ]
        if any(b is None for b in batches):
            import ipdb; ipdb.set_trace()
            return None
        if self.use_ground_truth_context:
            return np.array([self.ground_truth_tasks[i] for i in indices])
        context = [self.unpack_batch(batch) for batch in batches]
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [np.concatenate(x, axis=0) for x in context]

        if self.use_next_obs_in_context:
            context = np.concatenate(context[:-1], axis=2)
        else:
            context = np.concatenate(context[:-2], axis=2)
        return context

    def unpack_batch(self, batch):
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if self.sparse_rewards:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size
        else:
            raise TypeError("Unknown space: {}".format(space))


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        print("WARNING: will deprecate this SimpleReplayBuffer.")
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', 0)
        self._advance()

    def terminate_episode(self):
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            sparse_rewards=self._sparse_rewards[indices],
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        i = 0
        indices = []
        while len(indices) < batch_size:
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size

    def copy_data(self, other_buffer: 'SimpleReplayBuffer'):
        start_i = self._top
        end_i = self._top + other_buffer._top
        if end_i > self._max_replay_buffer_size:
            raise NotImplementedError()
        self._observations[start_i:end_i] = (
            other_buffer._observations[:other_buffer._top].copy()
        )
        self._actions[start_i:end_i] = (
            other_buffer._actions[:other_buffer._top].copy()
        )
        self._rewards[start_i:end_i] = (
            other_buffer._rewards[:other_buffer._top].copy()
        )
        self._terminals[start_i:end_i] = (
            other_buffer._terminals[:other_buffer._top].copy()
        )
        self._next_obs[start_i:end_i] = (
            other_buffer._next_obs[:other_buffer._top].copy()
        )
        self._sparse_rewards[start_i:end_i] = (
            other_buffer._sparse_rewards[:other_buffer._top].copy()
        )
