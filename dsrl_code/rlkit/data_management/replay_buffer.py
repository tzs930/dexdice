import abc
import numpy as np

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_sample(self, observation, action, reward, cost, next_observation,
                   terminal, **kwargs):
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        pass

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                cost,
                next_obs,
                terminal,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["costs"],
            path["next_observations"],
            path["terminals"]
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                cost=cost,
                next_observation=next_obs,
                terminal=terminal,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return

