# Quadruped Optimal Control: PMP, LQG & MPC

Optimal control framework for quadruped robot stabilization and commanded-velocity tracking in MuJoCo, built on top of `gym-quadruped`. The project compares three control strategies under external disturbances:

- Pontryagin Maximum Principle (PMP)
- Linear Quadratic Gaussian control (LQG)
- Model Predictive Control (MPC)

The main executable is `examples/run_mujoco.py`, which supports rendering, controller comparison, configurable robot models, disturbance injection, and keyboard teleoperation.

## Mathematical Foundation

### Dynamics

The floating-base state is defined as:

```text
x = [p, v, θ, ω]
````

where:

* `p ∈ R³`: base position
* `v ∈ R³`: base linear velocity
* `θ ∈ R³`: base orientation in Euler coordinates
* `ω ∈ R³`: base angular velocity

The discrete-time dynamics are modeled as:

```text
x_{k+1} = A_k x_k + B_k u_k + g_k + w_k
```

where `A_k` and `B_k` are obtained from the linearized single rigid body dynamics, `g_k` captures gravity and affine terms, and `u` contains the 12-dimensional ground reaction force vector:

```text
u = [f_FL, f_FR, f_RL, f_RR]
```

Each leg contributes a 3D contact force.

### 1. Pontryagin Maximum Principle (PMP)

The PMP controller is derived from the Hamiltonian:

```text
H(x, u, λ, t) = L(x, u, t) + <λ, f(x, u, t)>
```

The costate evolves according to:

```text
λ̇ = -∂H/∂x = -Qx - Aᵀλ
```

and the optimal control satisfies:

```text
u* = argmin_u H = -R⁻¹Bᵀλ
```

In implementation, this is solved through a backward Riccati sweep over a finite horizon.

### 2. LQG (LQR + Kalman Filter)

The LQG controller combines:

#### LQR state feedback

```text
P = Q + Aᵀ P A - Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
K = (R + Bᵀ P B)⁻¹ Bᵀ P A
```

#### Kalman state estimation

```text
Predict:  x̂⁻ = A x̂ + B u + g
          P⁻ = A P Aᵀ + Q_proc

Update:   K_kf = P⁻ Cᵀ (C P⁻ Cᵀ + R_meas)⁻¹
          x̂ = x̂⁻ + K_kf (y - C x̂⁻)
```

This allows state-feedback control under measurement noise and model uncertainty.

### 3. Model Predictive Control (MPC)

The MPC controller solves a receding-horizon quadratic program:

```text
min  Σ (x_k - x_ref)ᵀ Q (x_k - x_ref) + (u_k - u_ref)ᵀ R (u_k - u_ref)
s.t. x_{k+1} = A x_k + B u_k + g
     |f_x| ≤ μ f_z
     |f_y| ≤ μ f_z
     0 ≤ f_z ≤ f_max
```

The optimization includes:

* linearized dynamics
* friction pyramid constraints
* normal force bounds

The implementation uses OSQP.

### Orientation Estimation

An EKF is used to estimate orientation from IMU signals. The process follows the standard structure:

```text
Predict:  q⁺ = (I + ½ Ω(ω) Δt) q
Update:   y_a ≈ R_WBᵀ g
```

This estimated orientation is used inside the linearization and control pipeline.

## Important Scope Note

This repository currently implements **body-level optimal control** using ground reaction force regulation and torque mapping through the leg Jacobians. It is well suited for:

* posture stabilization
* disturbance rejection
* commanded velocity tracking at the base level
* controller comparison in simulation

However, this alone is **not a complete locomotion stack**. True walking typically also requires:

* gait scheduling
* swing-leg trajectory generation
* foothold planning
* contact sequence management

The teleoperation mode in `run_mujoco.py` provides commanded base velocities, but successful walking behavior still depends on the capabilities exposed by the underlying environment and contact dynamics.

## Project Structure

```text
quadruped-optimal-control/
├── src/
│   ├── dynamics.py          # Linearized SRB dynamics
│   ├── estimator_ekf.py     # Orientation EKF + state estimation utilities
│   ├── controller_pmp.py    # Pontryagin-based controller
│   ├── controller_lqg.py    # LQG controller
│   ├── controller_mpc.py    # MPC controller with force constraints
│   ├── trajectory.py        # Waypoint-based reference trajectory generator
├── examples/
│   └── run_mujoco.py        # Main MuJoCo runner (rendering, teleop, trajectories)
├── tests/
│   ├── test_all.py
│   └── test_trajectory_tracking.py   # MuJoCo-free benchmark on the SRB model
├── results/                 # Generated plots and CSV metrics
└── requirements.txt
```

## Features of `run_mujoco.py`

The main script supports:

* single-controller execution
* controller comparison mode
* configurable robot model via `--robot-name`
* MuJoCo rendering or headless mode
* external disturbances
* keyboard teleoperation
* automatic plot generation

Supported controllers:

* `pmp`
* `lqg`
* `mpc`
* `all`

Supported disturbance modes:

* `impulse`
* `persistent`
* `none`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install `gym-quadruped`

```bash
cd gym-quadruped-master
pip install -e .
```

### 3. Run a controller

```bash
python examples/run_mujoco.py --controller lqg --robot-name mini_cheetah
```

### 4. Run with rendering disabled

```bash
python examples/run_mujoco.py --controller mpc --robot-name mini_cheetah --no-render
```

### 5. Run all controllers for comparison

```bash
python examples/run_mujoco.py --controller all --robot-name mini_cheetah --duration 8
```

### 6. Run with teleoperation

```bash
python examples/run_mujoco.py --controller lqg --robot-name mini_cheetah --teleop
```

## Teleoperation

When `--teleop` is enabled, keyboard commands modify the commanded planar base velocity and yaw rate online.

Keys:

- `↑` / `↓`: increase or decrease forward velocity
- `←` / `→`: increase or decrease yaw rate
- `z` / `c`: increase or decrease lateral velocity
- `space`: reset commanded velocities to zero

Example:

```bash
python examples/run_mujoco.py --controller lqg --robot-name go2 --teleop

## Command-Line Arguments

python examples/run_mujoco.py \
    --controller lqg \
    --robot-name mini_cheetah \
    --duration 8 \
    --disturbance impulse \
    --teleop
```

Arguments:

* `--controller {pmp,lqg,mpc,all}`
* `--robot-name <name>`
* `--duration <seconds>`
* `--disturbance {impulse,persistent,none}`
* `--trajectory {none,static,line,square,circle,figure8}`
* `--teleop`
* `--no-render`

## Output

For single-controller runs, the script saves a plot in `results/` including:

* base position
* base velocity
* base orientation
* control effort
* disturbance profile

For comparison mode, the script saves an overlay plot comparing:

* position error
* velocity error
* control norm

Example output files:

```text
results/mujoco_lqg_mini_cheetah_impulse.png
results/mujoco_mpc_go2_persistent.png
results/mujoco_comparison_mini_cheetah_impulse.png
```

## Example Workflows

### Stabilization under impulse disturbance

```bash
python examples/run_mujoco.py --controller pmp --robot-name mini_cheetah --disturbance impulse
```

### Persistent disturbance rejection

```bash
python examples/run_mujoco.py --controller mpc --robot-name go2 --disturbance persistent
```

### Controller comparison

```bash
python examples/run_mujoco.py --controller all --robot-name mini_cheetah --duration 10 --no-render
```

### Interactive commanded-velocity test

```bash
python examples/run_mujoco.py --controller lqg --robot-name aliengo --teleop
```

## Waypoint-based trajectory tracking

`src/trajectory.py` generates smooth reference trajectories from sparse
waypoints. Each waypoint is `(t, x, y, z, yaw)`; position and yaw are
interpolated with cubic splines (clamped boundary conditions, zero
velocity at the endpoints), and linear / angular velocities come from the
analytic spline derivatives, so the controllers receive a dynamically
consistent 12-dim reference `[p, v, rpy, ω]` at every step.

Predefined trajectories accessible through `--trajectory <name>`:

| name      | description                                |
| --------- | ------------------------------------------ |
| `static`  | hold the standing pose (sanity check)      |
| `line`    | forward translation along +x               |
| `square`  | closed square in x-y, yaw locked to 0      |
| `circle`  | closed circle, yaw locked to 0             |
| `figure8` | lemniscate, yaw locked to 0                |

### Run a single controller on a trajectory

```bash
python examples/run_mujoco.py --controller mpc --trajectory circle --duration 12 --no-render
```

### Run all three controllers and produce the comparison overlay

```bash
python examples/run_mujoco.py --controller all --trajectory figure8 --duration 12 --no-render
```

For each run the script writes:

```text
results/trajectory_<controller>_<robot>_<traj>.png        # single run
results/trajectory_comparison_<robot>_<traj>.png          # --controller all
results/trajectory_metrics_<robot>_<traj>.csv             # --controller all
```

### MuJoCo-free benchmark

For a deterministic comparison of the three controllers on the *exact*
linearised SRB plant they are built on, run:

```bash
python tests/test_trajectory_tracking.py --trajectory all --duration 12
```

This produces, in `results/`:

```text
trajectory_lin_<traj>.png
trajectory_metrics_lin_<traj>.csv
```

### Trajectory-tracking results (SRB-linear benchmark)

Run with `duration=12 s`, `dt=0.01 s` on a matched-model plant. Numbers
are best-case for absolute magnitudes, but the *relative* ordering
between controllers is what matters.

| Trajectory | Controller | XY RMSE [m] | XY max [m] | Vel RMSE [m/s] | Mean ‖u‖ [N] |
| ---------- | ---------- | ----------- | ---------- | -------------- | ------------ |
| line       | PMP        | 0.0004      | 0.0008     | 0.0004         | 44.1         |
| line       | LQG        | 0.0015      | 0.0033     | 0.0178         | 44.5         |
| line       | MPC        | 0.0082      | 0.0132     | 0.0025         | 44.1         |
| square     | PMP        | 0.0024      | 0.0037     | 0.0025         | 44.2         |
| square     | LQG        | 0.0028      | 0.0052     | 0.0180         | 44.5         |
| square     | MPC        | 0.0168      | 0.0210     | 0.0112         | 44.2         |
| circle     | PMP        | 0.0022      | 0.0068     | 0.0049         | 44.2         |
| circle     | LQG        | 0.0029      | 0.0071     | 0.0186         | 44.5         |
| circle     | MPC        | 0.0186      | 0.0222     | 0.0120         | 44.2         |
| figure8    | PMP        | 0.0036      | 0.0130     | 0.0135         | 44.4         |
| figure8    | LQG        | 0.0040      | 0.0126     | 0.0221         | 44.7         |
| figure8    | MPC        | 0.0170      | 0.0280     | 0.0203         | 44.3         |

Discussion:

- **PMP** (LQR via the converged backward-sweep gain) gives the lowest
  position RMSE on every path. With perfect state and a matched plant,
  unconstrained linear feedback dominates.
- **LQG** is within a factor of 2 of PMP on position but adds a
  velocity-tracking penalty because its Kalman filter integrates
  measurement noise (we add 5 mm / 2 cm·s⁻¹ / 0.5° σ on the
  measurement vector each step).
- **MPC** has slightly larger steady-state error because its receding
  horizon assumes the *current* `x_ref` is constant for the next N
  steps; on time-varying references this introduces a small lag. The
  upside, not visible in the matched-model bench, is robustness:
  MPC is the only controller that respects friction-cone and
  per-leg `f_z` bounds, so it dominates as soon as constraints bite
  (slip, low μ, force saturation, contact loss).
- All three controllers operate near gravity-compensation effort
  (`mass·g = 88 N` total ≈ 44 N in 4-leg L2 norm), confirming that the
  trajectories are tracked through the SRB linearisation rather than
  by force-limit saturation.

The same trajectories driven through MuJoCo with
`examples/run_mujoco.py` exercise the full contact and leg dynamics.
Per the *Important Scope Note* above, this stays at the body level
(no gait planner / swing-leg synthesis), so MuJoCo runs primarily
demonstrate stabilisation quality at the commanded reference.

## Interpretation of Results

The metrics reported at the end of each run summarize controller behavior in terms of:

* state tracking error
* maximum observed deviation
* mean control effort

These are useful for comparing disturbance rejection and tracking performance, especially in the stabilization regime.

Because the system currently focuses on body-level control rather than full gait synthesis, results should be interpreted primarily as:

* base stabilization quality
* robustness to disturbances
* responsiveness to commanded velocity references

rather than as a full benchmark of autonomous quadruped locomotion.

## References

1. Kang, Wang, Xiong. *Fast Decentralized State Estimation for Legged Robot Locomotion via EKF and MHE*. arXiv:2405.20567.
2. Murrieta-Cid. *Hamilton-Jacobi-Bellman Equation and Pontryagin Maximum Principle*.
3. Di Carlo et al. *Dynamic Locomotion in the MIT Cheetah 3 through Convex Model-Predictive Control*. IROS 2018.

```
