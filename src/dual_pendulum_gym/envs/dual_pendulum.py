import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dual_pendulum_gym.physics.dynamics import PhysicsParams, rk4_step


def compute_rod_tips(state, params):
    """Compute y-coordinates of rod1 tip and rod2 tip."""
    _, th1, th2, _, _, _ = state
    # Rod 1 tip: y = -l1*cos(th1)  (positive y = up, rail at y=0)
    y_rod1 = -params.l1 * np.cos(th1)
    # Rod 2 tip: y = -l1*cos(th1) - l2*cos(th2)
    y_rod2 = y_rod1 - params.l2 * np.cos(th2)
    return y_rod1, y_rod2


def compute_center_of_mass_heights(state, params):
    """Compute center-of-mass y-coordinates for rod1 and rod2.

    Rod1 CoM: halfway along rod1 from cart pivot.
    Rod2 CoM: at rod1 tip + halfway along rod2.
    Positive y = up, pivot at y=0.
    """
    _, th1, th2, _, _, _ = state
    y_cm1 = -(params.l1 / 2.0) * np.cos(th1)
    y_cm2 = -params.l1 * np.cos(th1) - (params.l2 / 2.0) * np.cos(th2)
    return float(y_cm1), float(y_cm2)


def compute_reward(state, params, prev_state=None, steps_since_rod1_above=0,
                   consecutive_balanced=0, direction_reversed=False):
    """Reward: center-of-mass height (primary) + sustained balance time bonus.

    Convention: θ=0 is hanging down, θ=±π is upright.

    Core signal: weighted CoM heights (rod2 weighted higher).
    Sustained balance multiplier: the longer both rods stay above horizontal,
    the more each step is worth.
    """
    x, th1, th2, _, th1d, th2d = state

    # Wall crash penalty
    if abs(x) >= params.rail_limit:
        return -10.0

    # Wall proximity penalty (always active)
    wall_dist = 1.0 - abs(x) / params.rail_limit
    wall_penalty = -1.0 * max(0.0, 0.8 - wall_dist)

    # === PRIMARY: Center-of-mass heights ===
    # Rod1 CoM: range [-l1/2, +l1/2] = [-0.5, +0.5]
    # Rod2 CoM: range [-(l1+l2/2), +(l1+l2/2)] = [-1.4, +1.4]
    y_cm1, y_cm2 = compute_center_of_mass_heights(state, params)
    max_y1 = params.l1 / 2.0   # 0.5
    max_y2 = params.l1 + params.l2 / 2.0  # 1.4

    # Normalize to [0, 1] where 1 = perfectly upright
    h1_norm = (y_cm1 + max_y1) / (2.0 * max_y1)  # 0 at bottom, 1 at top
    h2_norm = (y_cm2 + max_y2) / (2.0 * max_y2)

    # Weighted height reward: rod2 matters more (weight 3x vs 1x)
    height_reward = 1.0 * h1_norm + 3.0 * h2_norm  # max = 4.0

    both_above = abs(th1) > np.pi / 2 and abs(th2) > np.pi / 2

    if not both_above:
        # === SWING-UP PHASE: only height progress matters, no angular velocity bonus ===
        progress = 0.0
        if prev_state is not None:
            prev_cm1, prev_cm2 = compute_center_of_mass_heights(prev_state, params)
            progress = ((y_cm1 - prev_cm1) + 3.0 * (y_cm2 - prev_cm2)) * 3.0

        # Idling penalty
        idle_penalty = 0.0
        if steps_since_rod1_above > 200:
            idle_penalty = -0.5 * min((steps_since_rod1_above - 200) / 200.0, 1.0)

        return height_reward + progress + idle_penalty + wall_penalty

    else:
        # === BALANCE PHASE ===

        # Sustained balance multiplier: 1.0x at start, ramps to 3.0x over 200 steps
        streak_mult = 1.0 + min(consecutive_balanced / 200.0, 1.0) * 2.0

        # Stability bonus: low angular velocity = good
        ang_vel = abs(th1d) + abs(th2d)
        stability_bonus = 2.0 * max(0.0, 1.0 - ang_vel / 6.0)

        # Spin penalty: fast spinning while above = bad
        spin_penalty = 0.0
        if ang_vel > 4.0:
            spin_penalty = -1.5 * (ang_vel - 4.0)

        return (height_reward * streak_mult + stability_bonus
                + spin_penalty + wall_penalty)


def compute_status(state, params):
    """Compute status light: 'red', 'green', or 'blue'."""
    _, th1, th2, _, _, _ = state
    # Both above horizontal: |θ| > π/2
    both_above = abs(th1) > np.pi / 2 and abs(th2) > np.pi / 2
    if not both_above:
        return "red"
    y_rod1, y_rod2 = compute_rod_tips(state, params)
    if y_rod2 > y_rod1:
        return "blue"
    return "green"


class DualPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, params=None):
        self.params = params or PhysicsParams()
        self.render_mode = render_mode

        # Action: 0=left, 1=noop, 2=right
        self.action_space = spaces.Discrete(3)

        # Observation: 14-dim enriched observation
        high = np.array([1.0] * 14, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high, high, shape=(14,), dtype=np.float32)

        # Internal state: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        self.state = None
        self.renderer = None
        self._step_count = 0
        self._prev_action = 1  # start as noop
        self._prev_xdot = 0.0  # for acceleration
        self._steps_since_rod1_above = 0  # steps since rod1 was last above horizontal
        self._consecutive_dir = 0  # consecutive frames pushing same direction
        self._consecutive_balanced = 0  # consecutive steps with both rods above horizontal

    def _get_obs(self):
        x, th1, th2, xd, th1d, th2d = self.state
        p = self.params
        x_norm = x / p.rail_limit                          # ±1 at walls
        x_accel = (xd - self._prev_xdot) / p.dt            # cart acceleration
        y_rod1, y_rod2 = compute_rod_tips(self.state, p)
        max_y = p.l1 + p.l2
        y_rod1_norm = y_rod1 / max_y                       # normalized rod tip y
        y_rod2_norm = y_rod2 / max_y
        time_norm = min(self._steps_since_rod1_above, 1000) / 1000.0  # 0~1
        force_ramp = min(self._consecutive_dir, p.force_ramp_steps) / p.force_ramp_steps  # 0~1
        balance_streak = min(self._consecutive_balanced, 500) / 500.0  # 0~1

        return np.array([
            x_norm, xd, x_accel,
            np.sin(th1), np.cos(th1),
            np.sin(th2), np.cos(th2),
            th1d, th2d,
            y_rod1_norm, y_rod2_norm,
            time_norm,
            force_ramp,
            balance_streak,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both rods hanging straight down (θ=0 = hanging, small perturbation to break symmetry)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._step_count = 0
        self._prev_action = 1
        self._prev_xdot = 0.0
        self._steps_since_rod1_above = 0
        self._consecutive_dir = 0
        self._consecutive_balanced = 0
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        # Ramping force: sustained same direction = stronger push
        p = self.params
        if action == 1:  # noop resets
            self._consecutive_dir = 0
            force = 0.0
        else:
            # Same direction as last push? (ignore noop in between)
            if action == self._prev_action:
                self._consecutive_dir += 1
            else:
                self._consecutive_dir = 1
            # Ramp from force_min to force_max over force_ramp_steps
            t = min(self._consecutive_dir, p.force_ramp_steps) / p.force_ramp_steps
            force_mag = p.force_min + (p.force_max - p.force_min) * t
            force = -force_mag if action == 0 else force_mag

        self._prev_xdot = self.state[3]  # save before physics step
        prev_state = self.state.copy()
        self.state = rk4_step(self.state, force, self.params)

        # Normalize angles to [-pi, pi]
        self.state[1] = ((self.state[1] + np.pi) % (2 * np.pi)) - np.pi
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi

        self._step_count += 1

        # Track how long rod1 has been below horizontal
        if abs(self.state[1]) > np.pi / 2:
            self._steps_since_rod1_above = 0
        else:
            self._steps_since_rod1_above += 1

        # Track consecutive balanced steps (both rods above horizontal)
        both_above = abs(self.state[1]) > np.pi / 2 and abs(self.state[2]) > np.pi / 2
        if both_above:
            self._consecutive_balanced += 1
        else:
            self._consecutive_balanced = 0

        # Detect direction reversal: left(0)->right(2) or right(2)->left(0)
        direction_reversed = (
            (self._prev_action == 0 and action == 2) or
            (self._prev_action == 2 and action == 0)
        )

        obs = self._get_obs()
        reward = compute_reward(
            self.state, self.params, prev_state,
            self._steps_since_rod1_above, self._consecutive_balanced,
            direction_reversed,
        )
        self._prev_action = action
        terminated = bool(abs(self.state[0]) > self.params.rail_limit)
        truncated = self._step_count >= 1000

        info = {
            "status": compute_status(self.state, self.params),
            "action": action,
            "force": force,
            "force_ramp": min(self._consecutive_dir, p.force_ramp_steps) / p.force_ramp_steps,
            "consecutive_balanced": self._consecutive_balanced,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            from dual_pendulum_gym.rendering.renderer import PendulumRenderer
            self.renderer = PendulumRenderer(self.params, self.render_mode)
        status = compute_status(self.state, self.params) if self.state is not None else "red"
        return self.renderer.render(self.state, status=status)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
