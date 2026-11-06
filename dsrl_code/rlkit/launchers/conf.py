import os
from os.path import join
import rlkit

rlkit_project_dir = join(os.path.dirname(rlkit.__file__), os.pardir)
LOCAL_LOG_DIR = join(rlkit_project_dir, 'data')

CODE_DIRS_TO_MOUNT = [
    rlkit_project_dir,
]

HOME = os.getenv('HOME') if os.getenv('HOME') is not None else os.getenv("USERPROFILE")

DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir=join(HOME, '.mujoco/'),
        mount_point='/root/.mujoco',
    ),
]
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    join(rlkit_project_dir, 'scripts', 'run_experiment_from_doodad.py')
)

DOODAD_DOCKER_IMAGE = "TODO"
INSTANCE_TYPE = 'c4.large'
SPOT_PRICE = 0.03

GPU_DOODAD_DOCKER_IMAGE = 'TODO'
GPU_INSTANCE_TYPE = 'g2.2xlarge'
GPU_SPOT_PRICE = 0.5

REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-west-1': "TODO",
    'us-east-1': "TODO",
}

REGION_TO_GPU_AWS_AVAIL_ZONE = {
    'us-east-1': "us-east-1b",
}

OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'



SINGULARITY_IMAGE = '/home/PATH/TO/IMAGE.img'
SINGULARITY_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin'
]
SLURM_CPU_CONFIG = dict(
    account_name='TODO',
    partition='savio',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)
SLURM_GPU_CONFIG = dict(
    account_name='TODO',
    partition='savio2_1080ti',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)


SSS_CODE_DIRS_TO_MOUNT = [
]
SSS_DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/global/home/users/USERNAME/.mujoco',
        mount_point='/root/.mujoco',
    ),
]
SSS_LOG_DIR = '/global/scratch/USERNAME/doodad-log'

SSS_IMAGE = '/global/scratch/USERNAME/TODO.img'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/global/home/users/USERNAME/path/to/rlkit/scripts'
    '/run_experiment_from_doodad.py'
)
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/USERNAME'
    '/.mujoco/mjpro150/bin'
]

GCP_IMAGE_NAME = 'TODO'
GCP_GPU_IMAGE_NAME = 'TODO'
GCP_BUCKET_NAME = 'TODO'

GCP_DEFAULT_KWARGS = dict(
    zone='us-west2-c',
    instance_type='n1-standard-4',
    image_project='TODO',
    terminate=True,
    preemptible=True,
    gpu_kwargs=dict(
        gpu_model='nvidia-tesla-p4',
        num_gpu=1,
    )
)

try:
    from rlkit.launchers.conf_private import *
except ImportError:
    print("No personal conf_private.py found.")
