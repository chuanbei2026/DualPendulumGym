from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicsParams:
    m_cart: float = 20.0     # cart mass (kg)
    m1: float = 0.3          # rod 1 mass (kg)
    m2: float = 0.5          # rod 2 mass (kg)
    l1: float = 1.0          # rod 1 length (m)
    l2: float = 0.8          # rod 2 length (m)
    g: float = 9.81          # gravitational acceleration (m/s^2)
    force_min: float = 80.0  # starting force when changing direction (N)
    force_max: float = 250.0 # max force after sustained pushing (N)
    force_ramp_steps: int = 30  # frames to ramp from min to max
    friction_cart: float = 8.0   # cart-rail viscous friction (N·s/m)
    damping_j1: float = 0.1     # joint 1 (cart-rod1) damping (N·m·s/rad)
    damping_j2: float = 0.1     # joint 2 (rod1-rod2) damping (N·m·s/rad)
    dt: float = 0.02         # time step (s)
    rail_limit: float = 5.0  # half-length of rail (m)


def equations_of_motion(state: np.ndarray, force: float, params: PhysicsParams) -> np.ndarray:
    """Compute derivatives of state using Lagrangian mechanics.

    State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
    Angles measured from vertical (upright = 0).

    For uniform rods, the center of mass is at l/2 and the moment of inertia
    about one end is (1/3)*m*l^2.
    """
    _, th1, th2, xd, th1d, th2d = state

    mc = params.m_cart
    m1 = params.m1
    m2 = params.m2
    l1 = params.l1
    l2 = params.l2
    g = params.g
    lc1 = l1 / 2  # center of mass distance for rod 1
    lc2 = l2 / 2  # center of mass distance for rod 2
    I1 = m1 * l1**2 / 3  # moment of inertia of rod 1 about pivot
    I2 = m2 * l2**2 / 3  # moment of inertia of rod 2 about pivot

    s1 = np.sin(th1)
    c1 = np.cos(th1)
    s2 = np.sin(th2)
    c2 = np.cos(th2)
    s12 = np.sin(th1 - th2)
    c12 = np.cos(th1 - th2)

    # Mass matrix M * [x_dd, th1_dd, th2_dd]^T = F
    # Derived from Lagrangian of cart + two uniform rods
    M = np.zeros((3, 3))
    M[0, 0] = mc + m1 + m2
    M[0, 1] = (m1 * lc1 + m2 * l1) * c1
    M[0, 2] = m2 * lc2 * c2
    M[1, 0] = M[0, 1]
    M[1, 1] = I1 + m2 * l1**2
    M[1, 2] = m2 * l1 * lc2 * c12
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]
    M[2, 2] = I2

    # Right-hand side (forces and Coriolis/centrifugal terms)
    F = np.zeros(3)
    F[0] = (force
            - params.friction_cart * xd
            + (m1 * lc1 + m2 * l1) * th1d**2 * s1
            + m2 * lc2 * th2d**2 * s2)
    F[1] = (-(m1 * lc1 + m2 * l1) * g * s1
            - params.damping_j1 * th1d
            - m2 * l1 * lc2 * th2d**2 * s12)
    F[2] = (-m2 * lc2 * g * s2
            - params.damping_j2 * th2d
            + m2 * l1 * lc2 * th1d**2 * s12)

    # Solve M * [x_dd, th1_dd, th2_dd] = F
    acc = np.linalg.solve(M, F)

    return np.array([xd, th1d, th2d, acc[0], acc[1], acc[2]])


def _rk4_substep(state: np.ndarray, force: float, dt: float, params: PhysicsParams) -> np.ndarray:
    """Single RK4 substep."""
    k1 = equations_of_motion(state, force, params)
    k2 = equations_of_motion(state + 0.5 * dt * k1, force, params)
    k3 = equations_of_motion(state + 0.5 * dt * k2, force, params)
    k4 = equations_of_motion(state + dt * k3, force, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step(state: np.ndarray, force: float, params: PhysicsParams, n_substeps: int = 4) -> np.ndarray:
    """Advance state by params.dt using n_substeps RK4 substeps for accuracy."""
    sub_dt = params.dt / n_substeps
    for _ in range(n_substeps):
        state = _rk4_substep(state, force, sub_dt, params)
    return state
