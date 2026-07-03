# Diffusion-Guided Proximity-Aware RMPflow

This section describes the reactive execution layer used to convert diffusion-policy inference points into safe RB10 motion. The diffusion policy provides a task-space end-effector target, while RMPflow computes the joint-space command by continuously combining target tracking, orientation regulation, proximity-based collision avoidance, joint constraints, and damping.

## A. System Overview

Let \(q,\dot{q}\in\mathbb{R}^{6}\) denote the RB10 joint position and velocity used by the controller. At each control cycle, the policy target is received as

\[
g = (p_g, R_g),
\]

where \(p_g\in\mathbb{R}^{3}\) is the diffusion-policy inference point and \(R_g\in SO(3)\) is the desired or fixed task orientation carried by the incoming pose message. The policy target is published on `/RMP_goal` as `geometry_msgs/Pose`. The `rmpflow_bridge` forwards this target to the controller as `/goal_pose` in the `base_link` frame. Execution is gated by `/RMP_flag`; the RMP solve is active only when `/RMP_flag == 1` and a valid `/goal_pose` has been received.

The complete online pipeline is:

\[
\text{diffusion target} \rightarrow \text{target/orientation RMPs}
\]
\[
\text{proximity ranges} \rightarrow \text{obstacle spheres} \rightarrow \text{collision RMP}
\]
\[
\text{RB10 state} \rightarrow \text{RMPflow root solve} \rightarrow \ddot{q}^{*}
\rightarrow \dot{q}_{cmd}, q_{cmd}.
\]

In the hardware experiments, the resulting velocity command \(\dot{q}_{cmd}\) is sent to the RB10 through the direct API `move_speed_j` interface. The integrated joint position \(q_{cmd}\) is published for logging and analysis on `/target_q`.

## B. Proximity-Based Obstacle Construction

The robot-mounted proximity sensors publish range measurements on `/proximity_distance*`. The `proximity_obstacle_bridge` transforms each valid measurement into the `base_link` frame and publishes collision primitives on `/obstacles`. Although `/obstacles` uses `visualization_msgs/MarkerArray`, the controller treats it as an algorithmic collision input and converts the markers into internal `ObstacleSphere` objects.

For sensor \(k\), let

\[
T_{BS,k}=(R_{BS,k},p_{BS,k})
\]

be the transform from the sensor frame to the base frame. The implementation uses the local positive \(x\)-axis as the sensing direction,

\[
n_k = R_{BS,k} e_x .
\]

Given the scaled range \(d_k\) and obstacle radius \(r_{o,k}\), the current experimental configuration places a spherical obstacle at

\[
p_{o,k}=p_{BS,k}+n_k(d_k+r_{o,k}).
\]

Thus, the near surface of the sphere coincides with the measured surface point:

\[
p_{o,k}-r_{o,k}n_k = p_{BS,k}+n_k d_k .
\]

The proximity bridge filters out disabled sensors, invalid range messages, and detections outside the trigger distance. In the current parameter file, the range scale is \(10^{-3}\), the minimum held range is \(0.05\,\mathrm{m}\), and the per-sensor trigger distance is \(0.15\,\mathrm{m}\). With the launch command used for the experiments, surface-patch visualization and surface-patch collision memory are disabled. Therefore, the active proximity collision model is a set of online spherical hit primitives, not a persistent reconstructed surface.

## C. RMPflow Formulation

Each RMP leaf is defined on a task map

\[
x_i=\phi_i(q), \qquad J_i(q)=\frac{\partial \phi_i}{\partial q}.
\]

The leaf returns a desired task-space acceleration \(a_i\) and metric \(M_i\). The pullback to the root joint space is

\[
A_i = J_i^T M_i J_i,
\]

\[
f_i = J_i^T M_i(a_i-\dot{J}_i\dot{q}).
\]

The root metric and force are accumulated over all enabled leaves:

\[
A=\sum_i A_i, \qquad f=\sum_i f_i.
\]

The desired joint acceleration is computed as

\[
\ddot{q}^{*}=(A+\lambda I)^{-1}f,
\]

where \(\lambda I\) is the root regularization term. The current implementation uses the Pinocchio-based direct solver with `solve_method: rmp2` and `rmp_type: canonical`.

## D. RMP Leaves

### 1. Diffusion Target RMP

The target RMP attracts the TCP control frame to the diffusion-policy target \(p_g\). Let \(p(q)\) and \(\dot{p}=J_p(q)\dot{q}\) be the current TCP position and velocity. The implemented target acceleration is

\[
a_{tar}=k_p\frac{p_g-p}{\|p_g-p\|+\epsilon}-k_d\dot{p}.
\]

The target metric is distance dependent:

\[
M_{tar}=\alpha m_{\max}I+(1-\alpha)m_{\min}\hat{d}\hat{d}^{T},
\]

where \(\hat{d}\) is the softened unit direction toward the goal and \(\alpha\) is a Gaussian function of the target distance. Near the goal, the metric becomes more isotropic and stronger; farther from the goal, it is shaped along the goal direction so that collision and constraint RMPs can reshape the motion when necessary.

### 2. Orientation Axis RMPs

The quaternion in `/goal_pose` is converted to a desired rotation matrix. Its columns define three axis targets,

\[
u_{g,x}, u_{g,y}, u_{g,z}.
\]

For each enabled TCP axis, the controller applies

\[
a_{axis}=k_p(u_g-u)-k_d\dot{u},
\]

where \(u\) is the current TCP axis. This keeps the end-effector orientation consistent with the pose target while the target RMP tracks \(p_g\). If the diffusion policy produces only a 3D point, the upstream publisher should still provide a valid pose, for example by using a fixed task orientation or the current TCP orientation.

### 3. Proximity Collision RMP

The collision RMP operates on distances between robot control points and obstacle spheres. For robot control point \(j\), obstacle \(i\), control-point radius \(r_{c,j}\), and obstacle radius \(r_{o,i}\), the signed clearance is

\[
s_{ij}=\|p_{c,j}(q)-p_{o,i}\|-r_{c,j}-r_{o,i}-m,
\]

where \(m\) is the configured collision margin. The collision task coordinate is

\[
x_{ij}=\max(s_{ij},0).
\]

The implemented collision leaf combines exponential repulsion with approach-velocity damping:

\[
a_{col}=a_{rep}(x_{ij})+a_{damp}(x_{ij},\dot{x}_{ij}).
\]

The collision metric increases as clearance decreases and is modulated to zero outside the configured influence radius. Consequently, obstacle avoidance is not implemented as a discrete mode switch. The obstacle modifies the same root-space metric and force used by the target and constraint RMPs.

The solver also includes configured link-attached body guard primitives in the same collision-distance task map. These guard primitives are treated as additional collision terms alongside the proximity-derived obstacle spheres.

### 4. Constraint and Stabilization RMPs

The active graph also includes:

- joint-limit RMPs, which push motion away from joint boundaries;
- a joint-velocity-cap RMP, which damps motion near the configured velocity limit;
- a configuration-space target RMP, which weakly biases the robot toward the default joint posture;
- a joint damping RMP, which suppresses high joint velocities.

Tangent-escape RMP and tangent-escape filter modules are available in the software stack, but both are disabled in the reported experimental configuration (`enable_tangent_escape_rmp: false`, `enable_tangent_escape_filter: false`) and are therefore not part of the active controller analyzed here.

## E. Real-Time Command Generation

The RMP controller runs at the configured control rate and forms a solver state from the RB10 joint feedback. In velocity mode, the solver state may blend measured or estimated feedback with the previous virtual command state:

\[
q_c=\beta_p q_m+(1-\beta_p)q_v,
\]

\[
\dot{q}_c=\beta_v\dot{q}_m+(1-\beta_v)\dot{q}_v.
\]

Using this state, the RMP solve returns \(\ddot{q}^{*}\). The controller clamps acceleration, integrates the command, and applies hardware guards:

\[
\dot{q}_{cmd}=\dot{q}_c+\mathrm{clip}(\ddot{q}^{*})\Delta t,
\]

\[
q_{cmd}=q_c+\dot{q}_{cmd}\Delta t.
\]

Before transmission, the command guard enforces joint-position bounds, joint-velocity saturation, predictive joint-limit protection, and optional minimum-height safety checks. If `/RMP_flag` is inactive or `/goal_pose` has not yet arrived, the velocity-mode controller sends a zero-velocity hold command instead of running the RMP solve.

## F. Implementation Used in the Experiments

The experiments reported for this method use:

```text
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  cb_simulation:=false \
  use_proximity_bridge:=true \
  proximity_surface_visualization:=false \
  surface_patch_enabled:=false \
  surface_patch_collision_memory_enabled:=false \
  use_interactive_goal:=false
```

With the launch-file defaults, this selects the direct RB10 API backend (`use_direct_hardware_backend:=true`), velocity command mode (`command_mode:=velocity`), the external goal bridge (`start_rmpflow_bridge:=true`), and proximity-derived collision obstacles (`use_proximity_bridge:=true`). The interactive goal source is disabled, so the only online task target is the diffusion-policy pose on `/RMP_goal`.

The active runtime topics are summarized as follows:

```text
/RMP_goal                    diffusion-policy pose target
/RMP_flag                    execution gate
/goal_pose                   controller target forwarded by rmpflow_bridge
/proximity_distance*         proximity sensor range inputs
/obstacles                   proximity-derived obstacle spheres
/target_q                    integrated joint target q_cmd
/rmp_position_command        integrated position-command debug output
/position_controllers/commands direct-backend command-vector debug output
```

The actual hardware command is not sent through a ROS controller in this configuration. The direct backend converts the guarded velocity command to the RB10 API format and transmits it through `move_speed_j`.

## G. Method Summary

The proposed execution layer treats the diffusion-policy output as a task-space target rather than a direct robot command. Proximity measurements generate local obstacle spheres online, and these sensor-derived collision terms are combined with target, orientation, joint-limit, velocity-cap, configuration-space, and damping RMPs in one root-space solve. The resulting acceleration is integrated and guarded before being sent to the RB10 as a velocity command. This structure allows the robot to follow learned diffusion-policy targets while continuously bending its motion away from nearby obstacles detected by onboard proximity sensors.

## H. Figure Caption Draft

Fig. X. Diffusion-guided proximity-aware RMPflow for RB10 control. A diffusion policy publishes an end-effector pose target on `/RMP_goal` and activates execution through `/RMP_flag`. Proximity ranges are transformed into local obstacle spheres in the robot base frame. The RMPflow controller combines target, orientation, proximity collision, joint-limit, velocity-cap, configuration-space, and damping RMPs to compute \(\ddot{q}^{*}\). The command is integrated into \(\dot{q}_{cmd}\) and \(q_{cmd}\); in the hardware experiments, \(\dot{q}_{cmd}\) is sent to the RB10 through the direct API `move_speed_j` interface.
