import os
import shutil
import time
import argparse
import datetime
import numpy as np
import time

import sys
import pickle

class Expert:
    def __init__(self, action_dim=3, **kwargs):
        self.action_dim = action_dim

    def get_action(self, obs):
        a = np.zeros((self.action_dim))
        valid = True
        reset = False
        accept = False
        return (a, valid, reset, accept)

class RandomAgent(Expert):
    def __init__(self, action_scale=0.1, action_space_dim=2):
        self.action_scale = action_scale
        self.action_space_dim = action_space_dim
        self.counter = 0

    def get_action(self, obs):
        action = np.random.uniform(-1, 1, self.action_space_dim)*self.action_scale
        self.counter += 1
        reset = self.counter % 25 == 0
        accept = reset
        valid = True
        return (action, valid, reset, accept)

class SpaceMouseExpert(Expert):
    def __init__(self, xyz_dims=3, xyz_remap=[0, 1, 2], xyz_scale=[1, 1, 1]):
        from robosuite.devices import SpaceMouse

        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.device = SpaceMouse()

    def get_action(self, obs):
        state = self.device.get_controller_state()
        dpos, rotation, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["left_click"],
            state["right_click"],
        )

        xyz = dpos[self.xyz_remap] * self.xyz_scale

        a = xyz[:self.xyz_dims]

        valid = not np.all(np.isclose(a, 0))

        return (a, valid, reset, accept)

def collect_one_rollout(env, expert, horizon=200, threshold=-1, add_action_noise=False, render=False, noise_sigma=0.3, pause=0,):
    o = env.reset()

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    for i in range(horizon):
        a, _ = expert.get_action(o)
        traj["observations"].append(o)
        if add_action_noise:
            exec_a = a + np.random.normal(0, noise_sigma, a.shape)
        else:
            exec_a = a
        o, r, done, info = env.step(exec_a)

    for i in range(horizon):
        a, valid, reset, accept = expert.get_action(o)

        if valid:
            traj["observations"].append(o)

            o, r, done, info = env.step(a)

            traj["actions"].append(a)
            traj["rewards"].append(r)
            traj["next_observations"].append(o)
            traj["terminals"].append(done)
            traj["agent_infos"].append(info)
            traj["env_infos"].append(info)
            print(r)

            if render:
                env.render()

            if pause:
                time.sleep(pause)

        if reset or accept:
            if len(traj["rewards"]) == 0:
                accept = False
            return accept, traj

        time.sleep(0.01)
    return accept, traj

def collect_one_rollout_mdp(env, expert, horizon=200, render=False, pause=0,  threshold=-1,):
    o = env.reset()

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    ret = 0
    for i in range(horizon):
        a, valid, _, _ = expert.get_action(o)

        traj["observations"].append(o)

        o, r, done, info = env.step(a)

        traj["actions"].append(a)
        traj["rewards"].append(r)
        traj["next_observations"].append(o)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)
        ret += r
        if render:
            env.render()

        if pause:
            time.sleep(pause)

        if done:
            break
    if threshold == -1:
        accept = True
    elif ret > threshold:
        accept = True
    else:
        accept = False
    return accept, traj

def draw_grid(img, line_color=(0, 0, 0), thickness=1, type_=None, pxstep=20):
    if type_ is None:
        type_ = cv2.LINE_AA
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

def collect_one_rollout_goal_conditioned(env, expert, horizon=200, threshold=-1, add_action_noise=False, key='', render=False, noise_sigma=0.3):
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    for i in range(horizon):
        a, _ = expert.get_action(np.concatenate((o['state_observation'], o['state_desired_goal'])))
        traj["observations"].append(o)
        if add_action_noise:
            exec_a = a + np.random.normal(0, noise_sigma, a.shape)
        else:
            exec_a = a
        o, r, done, info = env.step(exec_a)

        traj["actions"].append(a)
        traj["rewards"].append(r)
        traj["next_observations"].append(o)
        traj["terminals"].append(done)
        traj["agent_infos"].append(info)
        traj["env_infos"].append(info)

    if render:
        img = img.reshape(
            3,
            env.imsize,
            env.imsize,
        ).transpose()[::-1]
        cv2.imshow('env', img)
        cv2.waitKey(1)

        img = img_goal.reshape(
            3,
            env.imsize,
            env.imsize,
        ).transpose()[::-1]
        cv2.imshow('goal_env', img)
        cv2.waitKey(1)
    if threshold == -1:
        accept = True
    elif np.abs(traj["env_infos"][-1][key]) < threshold:
        accept = True
    else:
        accept = False

    return accept, traj

def collect_demos(env, expert, path="demos.npy", N=10, horizon=200, threshold=-1, add_action_noise=False, key='', render=False, noise_sigma=.3):
    data = []
    accepted = 0
    running_total = 0
    while len(data) < N:
        accept, traj = collect_one_rollout_goal_conditioned(env, expert, horizon, threshold=threshold, add_action_noise=add_action_noise, key=key, render=render, noise_sigma=noise_sigma)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            print("last reward", traj["rewards"][-1])
            print("accepted", len(data), "trajectories")
            accepted+=1
        else:
            print("discarded trajectory")
        running_total +=1
        print('Percent Accepted Trajectories So Far: ', accepted/running_total)

    np.save(path, data)


def collect_demos_fixed(env, expert, path="demos.npy", N=10, horizon=200, **kwargs):
    data = []

    i = 0
    while len(data) < N:
        accept, traj = collect_one_rollout_mdp(env, expert, horizon, **kwargs)
        if accept:
            data.append(traj)
            print("accepted trajectory length", len(traj["observations"]))
            print("accepted", len(data), "trajectories")
            print("total rewards", sum(traj["rewards"]))
        else:
            print("discarded trajectory")

    np.save(path, data)

if __name__ == '__main__':
    expert = SpaceMouseExpert()

    size = 0.1
    low = np.array([-size, 0.4 - size, 0])
    high = np.array([size, 0.4 + size, 0.1])
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=1,
        object_meshes=None,
        workspace_low = low,
        workspace_high = high,
        hand_low = low,
        hand_high = high,
        fix_z=True,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius=0.03,
        maxlen=0.03,
        init_hand_xyz=(0, 0.4-size, 0.089),
    )
    env = ImageEnv(env,
        non_presampled_goal_img_is_garbage=True,
        recompute_reward=False,
        init_camera=sawyer_pusher_camera_upright_v2,
    )
    collect_demos(env, expert)
