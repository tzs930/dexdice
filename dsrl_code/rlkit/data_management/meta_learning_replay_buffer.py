import numpy as np
import random

from rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer as RLKitSimpleReplayBuffer
)

from rlkit.envs.env_utils import get_dim


class MetaLearningReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            task_indices,
            use_next_obs_in_context,
            sparse_rewards,
            mini_buffer_max_size,
            use_ground_truth_context=False,
            ground_truth_tasks=None,
            sample_buffer_in_proportion_to_size=False,
    ):
        self.max_replay_buffer_size = max_replay_buffer_size
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
        self.env_info_sizes = dict()
        if sparse_rewards:
            self.env_info_sizes['sparse_reward'] = 1
        self.task_buffers = []
        self.mini_buffer_max_size = mini_buffer_max_size
        self._num_steps_can_sample = 0

        self.sample_buffer_in_proportion_to_size = (
            sample_buffer_in_proportion_to_size
        )

    def create_buffer(self, size=None):
        if size is None:
            size = self.mini_buffer_max_size
        return RLKitSimpleReplayBuffer(
            max_replay_buffer_size=size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=self.env_info_sizes,
        )

    @property
    def num_steps_can_sample(self):
        return self._num_steps_can_sample

    def add_paths(self, paths):
        new_buffer = self.create_buffer()
        for path in paths:
            new_buffer.add_path(path)
        self._num_steps_can_sample += new_buffer.num_steps_can_sample()
        self.append_buffer(new_buffer)

    def append_buffer(self, new_buffer):
        self.task_buffers.append(new_buffer)
        while self.num_steps_can_sample > self.max_replay_buffer_size:
            self._remove_task_buffer()

    def _remove_task_buffer(self):
        buffer_to_remove = random.choice(self.task_buffers)
        self.task_buffers.remove(buffer_to_remove)
        self._num_steps_can_sample -= buffer_to_remove.num_steps_can_sample()

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

    def _sample_contexts(self, indices, batch_size):
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

    def sample_meta_batch(self, meta_batch_size, rl_batch_size, embedding_batch_size):
        possible_indices = np.arange(len(self.task_buffers))
        if self.sample_buffer_in_proportion_to_size:
            sizes = np.array([buffer.num_steps_can_sample() for buffer in self.task_buffers])
            sample_probs = sizes / np.sum(sizes)
            indices = np.random.choice(
                possible_indices,
                meta_batch_size,
                p=sample_probs,
            )
        else:
            indices = np.random.choice(possible_indices, meta_batch_size)
        batch = self.sample_batch(indices, rl_batch_size)
        context = self._sample_contexts(indices, embedding_batch_size)
        batch['context'] = context
        return batch

    def sample_context(self, batch_size):
        possible_indices = np.arange(len(self.task_buffers))
        index = np.random.choice(possible_indices)
        return self._sample_contexts([index], batch_size)

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
