"""Waypoint-based reference trajectory generator.

Given a sparse list of waypoints `(t, x, y, z, yaw)`, builds a smooth
12-dim reference state `x_ref(t) = [p, v, rpy, omega]` for the body-level
PMP / LQG / MPC controllers.

Position and yaw are interpolated with cubic splines (clamped boundary
conditions, zero velocity at the endpoints). Linear and angular velocities
are obtained as analytic spline derivatives, so the controllers receive a
dynamically consistent reference (p and v match, yaw and wz match).

Predefined trajectories useful for benchmarking are exposed via
`make_trajectory(name, ...)`:

    static    : hold the standing pose
    line      : forward translation along +x
    square    : closed square in the x-y plane (yaw locked to 0)
    circle    : closed circle (yaw locked to 0)
    figure8   : lemniscate (yaw locked to 0)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class WaypointTrajectory:
    """Smooth reference trajectory from sparse waypoints.

    Parameters
    ----------
    waypoints : (N, 5) array-like
        Each row is `(t, x, y, z, yaw)`. Times must be strictly increasing.
    clamped_endpoints : bool
        When True, force zero linear and angular velocity at t0 and tf
        (smooth start/stop). When False, use natural cubic spline.
    """

    def __init__(self, waypoints, clamped_endpoints: bool = True):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 5:
            raise ValueError("waypoints must have shape (N, 5): t,x,y,z,yaw")
        if wp.shape[0] < 2:
            raise ValueError("need at least two waypoints")

        order = np.argsort(wp[:, 0])
        wp = wp[order]
        if np.any(np.diff(wp[:, 0]) <= 0):
            raise ValueError("waypoint times must be strictly increasing")

        self.times = wp[:, 0]
        self.points = wp[:, 1:4]
        self.yaws = np.unwrap(wp[:, 4])
        self.t0 = float(self.times[0])
        self.tf = float(self.times[-1])

        if clamped_endpoints:
            bc_xyz = ((1, np.zeros(3)), (1, np.zeros(3)))
            bc_yaw = ((1, 0.0), (1, 0.0))
        else:
            bc_xyz = "natural"
            bc_yaw = "natural"

        self._spline_xyz = CubicSpline(self.times, self.points, bc_type=bc_xyz)
        self._spline_yaw = CubicSpline(self.times, self.yaws, bc_type=bc_yaw)

    @property
    def duration(self) -> float:
        return self.tf - self.t0

    def evaluate(self, t: float):
        """Return `(p, v, yaw, yaw_rate)` at time `t`.

        For `t` outside `[t0, tf]` the position is clamped to the boundary
        waypoint and velocities are forced to zero so the robot holds the
        endpoint instead of extrapolating off the trajectory.
        """
        if t <= self.t0:
            return self._spline_xyz(self.t0), np.zeros(3), float(self._spline_yaw(self.t0)), 0.0
        if t >= self.tf:
            return self._spline_xyz(self.tf), np.zeros(3), float(self._spline_yaw(self.tf)), 0.0
        p = self._spline_xyz(t)
        v = self._spline_xyz(t, 1)
        yaw = float(self._spline_yaw(t))
        wz = float(self._spline_yaw(t, 1))
        return p, v, yaw, wz

    def reference_state(self, t: float) -> np.ndarray:
        """Compose the 12-dim controller reference at time `t`.

        Layout: x = [p(3), v(3), rpy(3), omega_body(3)].
        Roll and pitch references are zero (upright base), only yaw is set.
        """
        p, v, yaw, wz = self.evaluate(t)
        x_ref = np.zeros(12)
        x_ref[0:3] = p
        x_ref[3:6] = v
        x_ref[8] = yaw
        x_ref[11] = wz
        return x_ref

    def sample(self, dt: float):
        """Dense sampling over `[t0, tf]` for plotting or analysis."""
        n = max(2, int(np.ceil(self.duration / dt)) + 1)
        ts = np.linspace(self.t0, self.tf, n)
        states = np.array([self.reference_state(t) for t in ts])
        return ts, states


# ---------------------------------------------------------------------------
# Predefined trajectories
# ---------------------------------------------------------------------------
def _static_waypoints(height: float, duration: float):
    return np.array([
        [0.0,      0.0, 0.0, height, 0.0],
        [duration, 0.0, 0.0, height, 0.0],
    ])


def _line_waypoints(height: float, duration: float, length: float = 1.0):
    return np.array([
        [0.0,             0.0,            0.0, height, 0.0],
        [duration * 0.20, 0.0,            0.0, height, 0.0],
        [duration * 0.50, length * 0.5,   0.0, height, 0.0],
        [duration * 0.85, length,         0.0, height, 0.0],
        [duration,        length,         0.0, height, 0.0],
    ])


def _square_waypoints(height: float, duration: float, side: float = 0.6):
    return np.array([
        [0.0,             0.0,  0.0,  height, 0.0],
        [duration * 0.10, 0.0,  0.0,  height, 0.0],
        [duration * 0.30, side, 0.0,  height, 0.0],
        [duration * 0.50, side, side, height, 0.0],
        [duration * 0.70, 0.0,  side, height, 0.0],
        [duration * 0.90, 0.0,  0.0,  height, 0.0],
        [duration,        0.0,  0.0,  height, 0.0],
    ])


def _circle_waypoints(height: float, duration: float, radius: float = 0.5,
                      n_segments: int = 16):
    ts = np.linspace(0.0, duration, n_segments + 1)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    xs = radius * np.sin(thetas)
    ys = radius * (1.0 - np.cos(thetas))
    yaws = np.zeros_like(thetas)
    return np.column_stack([ts, xs, ys, np.full_like(ts, height), yaws])


def _figure8_waypoints(height: float, duration: float, a: float = 0.5,
                       n_segments: int = 32):
    ts = np.linspace(0.0, duration, n_segments + 1)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    xs = a * np.sin(thetas)
    ys = a * np.sin(thetas) * np.cos(thetas)
    yaws = np.zeros_like(thetas)
    return np.column_stack([ts, xs, ys, np.full_like(ts, height), yaws])


def make_trajectory(name: str, height: float = 0.225,
                    duration: float = 12.0) -> WaypointTrajectory:
    """Build one of the predefined waypoint trajectories."""
    builders = {
        "static":  _static_waypoints,
        "line":    _line_waypoints,
        "square":  _square_waypoints,
        "circle":  _circle_waypoints,
        "figure8": _figure8_waypoints,
    }
    name = name.lower()
    if name not in builders:
        raise ValueError(
            f"unknown trajectory '{name}'. "
            f"Available: {sorted(builders)}"
        )
    return WaypointTrajectory(builders[name](height, duration))
