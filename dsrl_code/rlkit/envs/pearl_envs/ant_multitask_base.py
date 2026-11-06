from rlkit.envs.pearl_envs.ant import AntEnv


class MultitaskAntEnv(AntEnv):
    def __init__(self, task=None, n_tasks=2,
                 randomize_tasks=True,
                 **kwargs):
        if task is None:
            task = {}
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super(MultitaskAntEnv, self).__init__(**kwargs)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        try:
            self._task = self.tasks[idx]
        except IndexError as e:
            import ipdb; ipdb.set_trace()
        self._goal = self._task['goal']
        self.reset()
