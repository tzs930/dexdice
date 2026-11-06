"""Minimal 2D mobile robot navigation env for rollout-based evaluation.

  - Omnidirectional (mecanum) kinematic dynamics with calibrated first-order lag
  - 2D LiDAR raycast (360 deg FoV, 180 beams) over rotated rectangular obstacles
  - Sparse reward: -1/step, 0 on goal (position + yaw), -100 on collision
  - Optional fixed layout via JSON for reproducible evaluation

Observation interface (matches deployment):
  scan:  (180,)  ranges in [0, 30] m
  state: (11,)   [vel(3), pose_xy_start(2), cos/sin(pose_yaw_start)(2),
                  goal_xy_start(2), cos/sin(goal_yaw_start)(2)]

API:
  env = MobileEnv2D()
  obs, info = env.reset(seed=...)
  obs, reward, done, info = env.step(action)
  env.render()
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class EnvConfig:
    arena_size: float = 10.0
    n_obstacles_max: int = 3
    obstacle_size_min: float = 1.0
    obstacle_size_max: float = 3.0
    obstacle_random_yaw: bool = True
    obstacle_obstacle_margin: float = 0.3
    # Robot footprint (m), heading axis x lateral
    robot_length: float = 0.978
    robot_width:  float = 0.776
    max_lin: float = 0.6
    max_ang: float = 1.2
    dt: float = 0.1
    max_steps: int = 200
    n_beams: int = 180
    scan_max: float = 30.0
    fov: float = 2.0 * np.pi
    goal_threshold: float = 0.2       # position tolerance (m)
    ori_threshold: float = 0.3        # yaw tolerance (rad ~ 17 deg)
    min_start_goal_dist: float = 2.0
    start_goal_clearance: float = 0.2
    collision_reward: float = -100.0
    goal_reward: float = 0.0
    step_reward: float = -1.0

    # Calibrated first-order dynamics (from real robot data):
    #   vel[t+1] = (1 - alpha) * vel[t] + alpha * K * cmd[t]
    dyn_K:     tuple = (0.985, 0.966, 0.982)
    dyn_alpha: tuple = (0.40, 0.41, 0.87)

    # If set, a JSON file with init_pose / goal_pose / obstacles is loaded
    # and reused on every reset. arena_size is overridden by the layout value.
    fixed_layout_path: str = None


# Geometry helpers (rotated boxes / OBB)

def _world_to_box_local(px, py, cx, cy, yaw):
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    dx = px - cx
    dy = py - cy
    return dx * cs - dy * sn, dx * sn + dy * cs


def _circle_obb_collide(robot_x, robot_y, radius, cx, cy, hw, hh, yaw, margin=0.0):
    lx, ly = _world_to_box_local(robot_x, robot_y, cx, cy, yaw)
    cx_local = np.clip(lx, -hw, hw)
    cy_local = np.clip(ly, -hh, hh)
    return np.hypot(lx - cx_local, ly - cy_local) < (radius + margin)


def _obb_corners(cx, cy, hw, hh, yaw):
    cs, sn = np.cos(yaw), np.sin(yaw)
    lx = np.array([-hw, +hw, +hw, -hw], dtype=np.float32)
    ly = np.array([-hh, -hh, +hh, +hh], dtype=np.float32)
    wx = lx * cs - ly * sn + cx
    wy = lx * sn + ly * cs + cy
    return np.stack([wx, wy], axis=1)


def _obb_in_arena(cx, cy, hw, hh, yaw, half):
    corners = _obb_corners(cx, cy, hw, hh, yaw)
    return bool(np.all(np.abs(corners) <= half))


def _obb_obb_collide(a, b):
    a_cx, a_cy, a_hw, a_hh, a_yaw = a
    b_cx, b_cy, b_hw, b_hh, b_yaw = b
    a_corners = _obb_corners(a_cx, a_cy, a_hw, a_hh, a_yaw)
    b_corners = _obb_corners(b_cx, b_cy, b_hw, b_hh, b_yaw)
    axes = (
        (np.cos(a_yaw),  np.sin(a_yaw)),
        (-np.sin(a_yaw), np.cos(a_yaw)),
        (np.cos(b_yaw),  np.sin(b_yaw)),
        (-np.sin(b_yaw), np.cos(b_yaw)),
    )
    for ax, ay in axes:
        a_proj = a_corners[:, 0] * ax + a_corners[:, 1] * ay
        b_proj = b_corners[:, 0] * ax + b_corners[:, 1] * ay
        if a_proj.max() < b_proj.min() or b_proj.max() < a_proj.min():
            return False
    return True


def _ray_obb_t(ox, oy, cos_a, sin_a, cx, cy, hw, hh, yaw):
    """Vectorized ray-OBB intersection. Returns t per beam (np.inf if no hit)."""
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    lx_o = (ox - cx) * cs - (oy - cy) * sn
    ly_o = (ox - cx) * sn + (oy - cy) * cs
    lx_d = cos_a * cs - sin_a * sn
    ly_d = cos_a * sn + sin_a * cs

    eps = 1e-9
    safe_lx = np.where(np.abs(lx_d) < eps, eps, lx_d)
    safe_ly = np.where(np.abs(ly_d) < eps, eps, ly_d)

    t_x_min = (-hw - lx_o) / safe_lx
    t_x_max = (+hw - lx_o) / safe_lx
    swap_x = lx_d < 0
    new_min_x = np.where(swap_x, t_x_max, t_x_min)
    new_max_x = np.where(swap_x, t_x_min, t_x_max)

    t_y_min = (-hh - ly_o) / safe_ly
    t_y_max = (+hh - ly_o) / safe_ly
    swap_y = ly_d < 0
    new_min_y = np.where(swap_y, t_y_max, t_y_min)
    new_max_y = np.where(swap_y, t_y_min, t_y_max)

    t_enter = np.maximum(new_min_x, new_min_y)
    t_exit  = np.minimum(new_max_x, new_max_y)
    hit = (t_enter <= t_exit) & (t_exit > 0)
    return np.where(hit, np.maximum(t_enter, 0), np.inf)


class MobileEnv2D:
    def __init__(self, config: EnvConfig = None):
        self.cfg = config or EnvConfig()
        self.fixed_layout = None
        if self.cfg.fixed_layout_path is not None:
            import json
            with open(self.cfg.fixed_layout_path) as fh:
                self.fixed_layout = json.load(fh)
            self.cfg.arena_size = float(self.fixed_layout['arena_size'])
        self.scan_angles = np.linspace(-np.pi, np.pi,
                                        self.cfg.n_beams,
                                        endpoint=False, dtype=np.float32)
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.rng = np.random.default_rng(0)
        self.pose = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.start_pose = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        # obstacles    : (n, 5) [cx, cy, hw, hh, yaw] -> collision OBB
        # safety_zones : (n, 5) same format with possibly larger half-extents;
        #                entering only counts a violation, no episode termination
        self.obstacles = np.zeros((0, 5), dtype=np.float32)
        self.safety_zones = np.zeros((0, 5), dtype=np.float32)
        self.step_count = 0
        self.done = False
        self.last_collided = False
        self.last_reached = False
        self.last_safety_violation = False
        self.safety_violation_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        half = c.arena_size / 2.0

        if self.fixed_layout is not None:
            obs_list = self.fixed_layout['obstacles']
            if obs_list:
                self.obstacles = np.array(
                    [[o['cx'], o['cy'], o['hw'], o['hh'], o['yaw']]
                     for o in obs_list], dtype=np.float32)
                self.safety_zones = np.array(
                    [[o['cx'], o['cy'],
                      o.get('safety_hw', o['hw']),
                      o.get('safety_hh', o['hh']),
                      o['yaw']]
                     for o in obs_list], dtype=np.float32)
            else:
                self.obstacles = np.zeros((0, 5), dtype=np.float32)
                self.safety_zones = np.zeros((0, 5), dtype=np.float32)
            self.pose       = np.array(self.fixed_layout['init_pose'], dtype=np.float32)
            self.start_pose = self.pose.copy()
            self.goal       = np.array(self.fixed_layout['goal_pose'], dtype=np.float32)
            self.vel[:] = 0.0
            self.step_count = 0
            self.done = False
            self.last_collided = False
            self.last_reached = False
            self.last_safety_violation = False
            self.safety_violation_count = 0
            return self._obs(), self._info()

        n_target = int(self.rng.integers(0, c.n_obstacles_max + 1))
        obstacles = []
        max_tries = 200
        tries = 0
        while len(obstacles) < n_target and tries < max_tries:
            tries += 1
            cx = float(self.rng.uniform(-half + 0.3, half - 0.3))
            cy = float(self.rng.uniform(-half + 0.3, half - 0.3))
            w = float(self.rng.uniform(c.obstacle_size_min, c.obstacle_size_max))
            h = float(self.rng.uniform(c.obstacle_size_min, c.obstacle_size_max))
            yaw = (float(self.rng.uniform(-np.pi, np.pi))
                   if c.obstacle_random_yaw else 0.0)
            hw, hh = w / 2, h / 2
            cand = (cx, cy, hw, hh, yaw)
            if not _obb_in_arena(cx, cy, hw, hh, yaw, half):
                continue
            m = c.obstacle_obstacle_margin
            inflated_cand = (cx, cy, hw + m / 2, hh + m / 2, yaw)
            collided = False
            for prev in obstacles:
                inflated_prev = (prev[0], prev[1], prev[2] + m / 2,
                                  prev[3] + m / 2, prev[4])
                if _obb_obb_collide(inflated_cand, inflated_prev):
                    collided = True
                    break
            if not collided:
                obstacles.append([cx, cy, hw, hh, yaw])
        self.obstacles = (np.array(obstacles, dtype=np.float32)
                           if obstacles else np.zeros((0, 5), dtype=np.float32))
        self.safety_zones = self.obstacles.copy()

        self.pose = self._sample_free_pose(half)
        self.start_pose = self.pose.copy()

        for _ in range(100):
            cand = self._sample_free_pose(half)
            d = np.hypot(cand[0] - self.pose[0], cand[1] - self.pose[1])
            if d >= c.min_start_goal_dist:
                self.goal = cand
                break
        else:
            self.goal = np.array([-self.pose[0], -self.pose[1],
                                    self.rng.uniform(-np.pi, np.pi)],
                                  dtype=np.float32)

        self.vel[:] = 0.0
        self.step_count = 0
        self.done = False
        self.last_collided = False
        self.last_reached = False
        self.last_safety_violation = False
        self.safety_violation_count = 0
        return self._obs(), self._info()

    def step(self, action):
        c = self.cfg
        a = np.asarray(action, dtype=np.float32).copy()
        a[0] = np.clip(a[0], -c.max_lin, c.max_lin)
        a[1] = np.clip(a[1], -c.max_lin, c.max_lin)
        a[2] = np.clip(a[2], -c.max_ang, c.max_ang)

        K = np.asarray(c.dyn_K, dtype=np.float32)
        alpha = np.asarray(c.dyn_alpha, dtype=np.float32)
        target_vel = K * a
        new_vel = (1.0 - alpha) * self.vel + alpha * target_vel
        new_vel[:2] = np.clip(new_vel[:2], -c.max_lin, c.max_lin)
        new_vel[2]  = np.clip(new_vel[2],  -c.max_ang, c.max_ang)

        x, y, yaw = self.pose
        vx, vy, wz = new_vel
        cs, sn = np.cos(yaw), np.sin(yaw)
        new_x = x + (vx * cs - vy * sn) * c.dt
        new_y = y + (vx * sn + vy * cs) * c.dt
        new_yaw = np.arctan2(np.sin(yaw + wz * c.dt), np.cos(yaw + wz * c.dt))
        new_pose = np.array([new_x, new_y, new_yaw], dtype=np.float32)

        half = c.arena_size / 2.0
        rl_h = c.robot_length / 2.0
        rw_h = c.robot_width / 2.0
        out_of_bounds = not _obb_in_arena(
            float(new_x), float(new_y), rl_h, rw_h, float(new_yaw), half)
        collided_obs = self._robot_obb_collides_with_obstacles(new_pose)
        collided = bool(out_of_bounds or collided_obs)

        if collided:
            self.last_collided = True
            self.vel = np.zeros(3, dtype=np.float32)
        else:
            self.pose = new_pose
            self.vel = new_vel

        self.last_safety_violation = self._robot_obb_in_safety_zone(self.pose)
        if self.last_safety_violation:
            self.safety_violation_count += 1

        self.step_count += 1

        dist = np.hypot(self.pose[0] - self.goal[0], self.pose[1] - self.goal[1])
        yaw_err = abs(np.arctan2(np.sin(self.pose[2] - self.goal[2]),
                                  np.cos(self.pose[2] - self.goal[2])))
        reached = bool(dist < c.goal_threshold and yaw_err < c.ori_threshold)
        self.last_reached = reached

        if collided:
            reward = c.collision_reward
        elif reached:
            reward = c.goal_reward
        else:
            reward = c.step_reward

        self.done = collided or reached or (self.step_count >= c.max_steps)
        return self._obs(), float(reward), self.done, self._info()

    def _sample_free_pose(self, half):
        c = self.cfg
        rl_h = c.robot_length / 2.0
        rw_h = c.robot_width / 2.0
        half_diag = float(np.hypot(rl_h, rw_h))
        bound = half - half_diag - 0.05
        for _ in range(200):
            x = self.rng.uniform(-bound, bound)
            y = self.rng.uniform(-bound, bound)
            yaw = self.rng.uniform(-np.pi, np.pi)
            cand = np.array([x, y, yaw], dtype=np.float32)
            if not _obb_in_arena(float(x), float(y), rl_h, rw_h, float(yaw), half):
                continue
            if self._robot_obb_collides_with_obstacles(cand,
                                                        margin=c.start_goal_clearance):
                continue
            return cand
        return np.zeros(3, dtype=np.float32)

    def _robot_obb_collides_with_obstacles(self, pose, margin=0.0):
        c = self.cfg
        rl_h = c.robot_length / 2.0
        rw_h = c.robot_width / 2.0
        robot = (float(pose[0]), float(pose[1]), rl_h, rw_h, float(pose[2]))
        for cx, cy, hw, hh, yaw in self.obstacles:
            if _obb_obb_collide(robot,
                                  (float(cx), float(cy),
                                   float(hw) + margin, float(hh) + margin,
                                   float(yaw))):
                return True
        return False

    def _robot_obb_in_safety_zone(self, pose):
        c = self.cfg
        rl_h = c.robot_length / 2.0
        rw_h = c.robot_width / 2.0
        robot = (float(pose[0]), float(pose[1]), rl_h, rw_h, float(pose[2]))
        for cx, cy, hw, hh, yaw in self.safety_zones:
            if _obb_obb_collide(robot,
                                  (float(cx), float(cy),
                                   float(hw), float(hh),
                                   float(yaw))):
                return True
        return False

    def _raycast(self):
        c = self.cfg
        x, y, yaw = self.pose
        angles_world = self.scan_angles + yaw
        cos_a = np.cos(angles_world)
        sin_a = np.sin(angles_world)

        d_min = np.full(c.n_beams, c.scan_max, dtype=np.float32)

        for cx, cy, hw, hh, b_yaw in self.obstacles:
            t = _ray_obb_t(x, y, cos_a, sin_a, cx, cy, hw, hh, b_yaw)
            d_min = np.minimum(d_min, t)

        half = c.arena_size / 2.0
        eps = 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_pos = (half - x) / np.where(np.abs(cos_a) > eps, cos_a, eps)
            tx_neg = (-half - x) / np.where(np.abs(cos_a) > eps, cos_a, eps)
            ty_pos = (half - y) / np.where(np.abs(sin_a) > eps, sin_a, eps)
            ty_neg = (-half - y) / np.where(np.abs(sin_a) > eps, sin_a, eps)
        for t_wall in [tx_pos, tx_neg, ty_pos, ty_neg]:
            t_wall = np.where(t_wall > 0, t_wall, np.inf)
            d_min = np.minimum(d_min, t_wall)

        return np.clip(d_min, 0.0, c.scan_max).astype(np.float32)

    def _build_state(self):
        sx, sy, syaw = self.start_pose
        cs, sn = np.cos(syaw), np.sin(syaw)
        dx = self.pose[0] - sx
        dy = self.pose[1] - sy
        px_s = dx * cs + dy * sn
        py_s = -dx * sn + dy * cs
        pyaw_s = np.arctan2(np.sin(self.pose[2] - syaw),
                             np.cos(self.pose[2] - syaw))

        gdx = self.goal[0] - sx
        gdy = self.goal[1] - sy
        gx_s = gdx * cs + gdy * sn
        gy_s = -gdx * sn + gdy * cs
        gyaw_s = np.arctan2(np.sin(self.goal[2] - syaw),
                             np.cos(self.goal[2] - syaw))

        return np.array([
            self.vel[0], self.vel[1], self.vel[2],
            px_s, py_s, np.cos(pyaw_s), np.sin(pyaw_s),
            gx_s, gy_s, np.cos(gyaw_s), np.sin(gyaw_s),
        ], dtype=np.float32)

    def _obs(self):
        scan = self._raycast()
        return {
            'scan': scan,
            'state': self._build_state(),
            'achieved_pose': self.pose.copy(),
            'desired_goal':  self.goal.copy(),
        }

    def _info(self):
        return {
            'collided':                self.last_collided,
            'reached':                 self.last_reached,
            'step':                    self.step_count,
            'pose':                    self.pose.copy(),
            'goal':                    self.goal.copy(),
            'safety_violation':        self.last_safety_violation,
            'safety_violation_count':  self.safety_violation_count,
        }

    def render(self, ax=None, show_scan=True, title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.transforms import Affine2D
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        c = self.cfg
        half = c.arena_size / 2.0

        ax.add_patch(Rectangle((-half, -half), c.arena_size, c.arena_size,
                                fill=False, edgecolor='k', lw=1.5))

        for cx, cy, hw, hh, yaw in self.safety_zones:
            rect = Rectangle((-hw, -hh), 2 * hw, 2 * hh,
                              facecolor='lightgreen', edgecolor='green',
                              alpha=0.35, lw=1.0)
            t = (Affine2D().rotate(yaw).translate(cx, cy) + ax.transData)
            rect.set_transform(t)
            ax.add_patch(rect)
        for cx, cy, hw, hh, yaw in self.obstacles:
            rect = Rectangle((-hw, -hh), 2 * hw, 2 * hh,
                              color='gray', alpha=0.6)
            t = (Affine2D().rotate(yaw).translate(cx, cy) + ax.transData)
            rect.set_transform(t)
            ax.add_patch(rect)

        rl, rw = c.robot_length, c.robot_width
        robot_rect = Rectangle((-rl / 2, -rw / 2), rl, rw,
                                color='tab:blue', alpha=0.6)
        t_rob = (Affine2D().rotate(self.pose[2])
                 .translate(self.pose[0], self.pose[1]) + ax.transData)
        robot_rect.set_transform(t_rob)
        ax.add_patch(robot_rect)
        head_len = rl
        ax.arrow(self.pose[0], self.pose[1],
                  0.5 * head_len * np.cos(self.pose[2]),
                  0.5 * head_len * np.sin(self.pose[2]),
                  head_width=0.15, color='tab:blue')

        ax.plot(self.goal[0], self.goal[1], 'r*', ms=18)
        ax.add_patch(Circle((self.goal[0], self.goal[1]),
                              c.goal_threshold, fill=False,
                              edgecolor='r', ls='--'))
        ax.arrow(self.goal[0], self.goal[1],
                  0.4 * np.cos(self.goal[2]),
                  0.4 * np.sin(self.goal[2]),
                  head_width=0.12, color='r', alpha=0.6)

        if show_scan:
            scan = self._raycast()
            angles = self.scan_angles + self.pose[2]
            ex = self.pose[0] + scan * np.cos(angles)
            ey = self.pose[1] + scan * np.sin(angles)
            for i in range(0, len(scan), 5):
                ax.plot([self.pose[0], ex[i]], [self.pose[1], ey[i]],
                          color='cyan', alpha=0.15, lw=0.5)

        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title)
        return ax
