import os

from gymnasium.spaces import Box, Discrete, Tuple
from gym.spaces import Box as OldBox, Discrete as OldDiscrete, Tuple as OldTuple

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def get_dim(space):
    if isinstance(space, Box) or isinstance(space, OldBox):
        return space.low.size
    elif isinstance(space, Discrete) or isinstance(space, OldDiscrete):
        return space.n
    elif isinstance(space, Tuple) or isinstance(space, OldTuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


def mode(env, mode_type):
    try:
        getattr(env, mode_type)()
    except AttributeError:
        pass
