from gym.envs.mujoco import AntEnv


class AntNormal(AntEnv):
    def __init__(
            self,
            *args,
            n_tasks=2,
            randomize_tasks=True,
            **kwargs
    ):
        self.tasks = [0 for _ in range(n_tasks)]
        self._goal = 0
        super().__init__(*args, **kwargs)

    def get_all_task_idx(self):
        return self.tasks

    def reset_task(self, idx):
        pass

    def sample_tasks(self, num_tasks):
        return [0 for _ in range(num_tasks)]
