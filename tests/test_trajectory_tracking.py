"""Closed-loop waypoint tracking benchmark on the SRB linear model.

This script does not require MuJoCo or gym-quadruped: it uses the same
linearised single-rigid-body dynamics that the controllers are built on
(`src.dynamics.QuadrupedDynamics`) as a stand-in plant. It is a
deterministic, reproducible benchmark to compare PMP / LQG / MPC tracking
performance against the waypoint reference produced by
`src.trajectory.WaypointTrajectory`.

It produces, in `results/`:
    * trajectory_lin_<traj>.png            (XY path overlay + error curves)
    * trajectory_metrics_lin_<traj>.csv    (per-controller RMSE, max, effort)

Run:
    python tests/test_trajectory_tracking.py --trajectory circle --duration 12
    python tests/test_trajectory_tracking.py --trajectory all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics import QuadrupedDynamics
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController
from src.trajectory import WaypointTrajectory, make_trajectory


HIP_HEIGHT = 0.225
ROBOT_MASS = 9.0
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])


def build_dynamics() -> QuadrupedDynamics:
    dyn = QuadrupedDynamics(mass=ROBOT_MASS, inertia=ROBOT_INERTIA, dt=0.01)
    dyn.r_feet_body = np.array([
        [ 0.19,  0.111, -HIP_HEIGHT],
        [ 0.19, -0.111, -HIP_HEIGHT],
        [-0.19,  0.111, -HIP_HEIGHT],
        [-0.19, -0.111, -HIP_HEIGHT],
    ])
    return dyn


def build_costs():
    Q = np.diag([
        80, 80, 400,
        8, 8, 40,
        150, 150, 30,
        1, 1, 4,
    ])
    R = np.eye(12) * 1e-4
    Q_f = Q * 5
    return Q, R, Q_f


def build_controller(name: str, dyn: QuadrupedDynamics, Q, R, Q_f, x_lin):
    A_d, B_d, g_d = dyn.get_linear_system(x_lin)
    A_c, B_c = dyn.continuous_AB(x_lin)

    if name == "pmp":
        ctrl = PontryaginController(
            A=A_c, B=B_c,
            Q_s=Q, R_u=R, Q_f=Q_f,
            g_aff=dyn.gravity_vector() / dyn.dt,
            dt=dyn.dt, horizon=200,
        )
        # The default K_ss is the continuous-time CARE gain. With dt=0.01s
        # and the chosen Q,R it places closed-loop poles outside the
        # discrete unit disk and the loop diverges. Use the converged
        # discrete-sweep gain (≈ DARE solution) instead.
        ctrl.solve_discrete_sweep(x_lin.copy(), x_lin)
        ctrl.K_ss = ctrl._gains[0]
        return ctrl
    if name == "lqg":
        ctrl = LQGController(
            A_d=A_d, B_d=B_d, g_d=g_d,
            Q=Q * dyn.dt, R=R * dyn.dt,
            Q_proc=np.diag([1e-3] * 3 + [1e-2] * 3 + [5e-3] * 3 + [1e-2] * 3),
            R_meas=np.diag([5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3),
        )
        ctrl.set_initial_estimate(x_lin)
        return ctrl
    if name == "mpc":
        ctrl = MPCController(
            A_d=A_d, B_d=B_d, g_d=g_d,
            Q=Q * dyn.dt, R=R * dyn.dt, Q_f=Q_f * dyn.dt,
            N=10, mu=0.6, fz_max=150.0,
        )
        return ctrl
    raise ValueError(name)


def simulate(controller_name: str, traj: WaypointTrajectory, duration: float,
              dyn: QuadrupedDynamics, Q, R, Q_f, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dyn.dt) + 1
    times = np.arange(n_steps) * dyn.dt

    x = traj.reference_state(0.0).copy()
    x[2] = HIP_HEIGHT
    u_ref = dyn.standing_control()

    x_lin = dyn.standing_state(height=HIP_HEIGHT)
    ctrl = build_controller(controller_name, dyn, Q, R, Q_f, x_lin)

    x_log = np.zeros((n_steps, 12))
    xref_log = np.zeros((n_steps, 12))
    u_log = np.zeros((n_steps, 12))

    for k, t in enumerate(times):
        x_ref = traj.reference_state(t)
        x_ref[2] = HIP_HEIGHT
        x_ref[5] = 0.0

        if controller_name == "lqg":
            y = x + rng.standard_normal(12) * np.array(
                [5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3
            )
            u = ctrl.step(y, x_ref, u_ref)
        else:
            u = ctrl.compute_control(x=x, x_ref=x_ref, u_ref=u_ref)

        u = np.clip(u, -150.0, 150.0)

        x_log[k] = x
        xref_log[k] = x_ref
        u_log[k] = u
        x = dyn.step(x, u)

    return {"time": times, "state": x_log, "state_ref": xref_log, "control": u_log}


def metrics_from_log(log):
    pos_err = log["state"][:, :3] - log["state_ref"][:, :3]
    vel_err = log["state"][:, 3:6] - log["state_ref"][:, 3:6]
    yaw_err = np.arctan2(
        np.sin(log["state"][:, 8] - log["state_ref"][:, 8]),
        np.cos(log["state"][:, 8] - log["state_ref"][:, 8]),
    )
    return {
        "pos_xy_rmse": float(np.sqrt(np.mean(np.sum(pos_err[:, :2] ** 2, axis=1)))),
        "pos_xy_max":  float(np.max(np.linalg.norm(pos_err[:, :2], axis=1))),
        "pos_z_rmse":  float(np.sqrt(np.mean(pos_err[:, 2] ** 2))),
        "vel_rmse":    float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=1)))),
        "yaw_rmse":    float(np.sqrt(np.mean(yaw_err ** 2))),
        "u_mean":      float(np.mean(np.linalg.norm(log["control"], axis=1))),
        "u_max":       float(np.max(np.linalg.norm(log["control"], axis=1))),
    }


def save_plots_and_csv(results, traj_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)
    colors = {"pmp": "#e74c3c", "lqg": "#2ecc71", "mpc": "#3498db"}

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"SRB-linear benchmark — trajectory: {traj_name}",
        fontsize=14, fontweight="bold",
    )

    ax_xy = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    first = next(iter(results.values()))
    ref = first["state_ref"]
    ax_xy.plot(ref[:, 0], ref[:, 1], "k--", lw=2.0, label="reference")
    for name, data in results.items():
        ax_xy.plot(data["state"][:, 0], data["state"][:, 1],
                   color=colors[name], lw=1.5, label=name.upper())
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("XY path tracking")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=9)

    ax_xyerr = plt.subplot2grid((3, 2), (0, 1))
    for name, data in results.items():
        e = data["state"][:, :2] - data["state_ref"][:, :2]
        ax_xyerr.plot(data["time"], np.linalg.norm(e, axis=1),
                      color=colors[name], lw=1.3, label=name.upper())
    ax_xyerr.set_ylabel("XY error [m]")
    ax_xyerr.legend(fontsize=8)
    ax_xyerr.grid(True, alpha=0.3)

    ax_yawerr = plt.subplot2grid((3, 2), (1, 1))
    for name, data in results.items():
        e = np.arctan2(
            np.sin(data["state"][:, 8] - data["state_ref"][:, 8]),
            np.cos(data["state"][:, 8] - data["state_ref"][:, 8]),
        )
        ax_yawerr.plot(data["time"], np.degrees(e),
                       color=colors[name], lw=1.3, label=name.upper())
    ax_yawerr.set_ylabel("Yaw error [deg]")
    ax_yawerr.set_xlabel("Time [s]")
    ax_yawerr.legend(fontsize=8)
    ax_yawerr.grid(True, alpha=0.3)

    ax_u = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    for name, data in results.items():
        ax_u.plot(data["time"], np.linalg.norm(data["control"], axis=1),
                  color=colors[name], lw=1.2, label=name.upper())
    ax_u.set_ylabel("||GRFs|| [N]")
    ax_u.set_xlabel("Time [s]")
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"results/trajectory_lin_{traj_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    csv_path = f"results/trajectory_metrics_lin_{traj_name}.csv"
    with open(csv_path, "w") as f:
        f.write("controller,pos_xy_rmse,pos_xy_max,pos_z_rmse,"
                "vel_rmse,yaw_rmse_deg,u_mean,u_max\n")
        for name, data in results.items():
            m = metrics_from_log(data)
            f.write(
                f"{name},{m['pos_xy_rmse']:.6f},{m['pos_xy_max']:.6f},"
                f"{m['pos_z_rmse']:.6f},{m['vel_rmse']:.6f},"
                f"{np.degrees(m['yaw_rmse']):.4f},"
                f"{m['u_mean']:.4f},{m['u_max']:.4f}\n"
            )
    return plot_path, csv_path


def run_one(traj_name: str, duration: float):
    dyn = build_dynamics()
    Q, R, Q_f = build_costs()
    traj = make_trajectory(traj_name, height=HIP_HEIGHT, duration=duration)

    results = {}
    for name in ["pmp", "lqg", "mpc"]:
        print(f"  Simulating {name.upper()} on '{traj_name}' ...")
        results[name] = simulate(name, traj, duration, dyn, Q, R, Q_f)

    plot_path, csv_path = save_plots_and_csv(results, traj_name)
    print(f"  Plot: {plot_path}")
    print(f"  CSV : {csv_path}")

    print(f"\n  --- Tracking metrics ({traj_name}) ---")
    print(f"  {'Ctrl':<6}{'XY RMSE':>10}{'XY max':>10}"
          f"{'Yaw RMSE':>11}{'Vel RMSE':>11}{'Mean ||u||':>12}")
    for name, data in results.items():
        m = metrics_from_log(data)
        print(
            f"  {name.upper():<6}"
            f"{m['pos_xy_rmse']:>10.4f}{m['pos_xy_max']:>10.4f}"
            f"{np.degrees(m['yaw_rmse']):>10.2f}°"
            f"{m['vel_rmse']:>11.4f}{m['u_mean']:>12.1f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        default="all",
        choices=["all", "static", "line", "square", "circle", "figure8"],
    )
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    names = (["line", "square", "circle", "figure8"]
             if args.trajectory == "all" else [args.trajectory])
    for n in names:
        print(f"\n=== Trajectory: {n} (duration {args.duration}s) ===")
        run_one(n, args.duration)
