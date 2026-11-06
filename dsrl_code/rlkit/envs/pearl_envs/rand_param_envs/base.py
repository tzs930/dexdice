
import numpy as np
from gym import Env
from gym.envs.mujoco import MujocoEnv


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        raise NotImplementedError

    def set_task(self, task):
        raise NotImplementedError

    def get_task(self):
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        pass

class RandomEnv(MetaEnv, MujocoEnv):
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs):
        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        self.save_parameters()

    def sample_tasks(self, n_tasks):
        param_sets = []

        for _ in range(n_tasks):
            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            param_variable[:] = param_val
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params