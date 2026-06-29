# Proximity-Aware RMPflow Framework

## A. Framework Overview

This work uses RMPflow as a reactive motion generation layer between a learned diffusion policy and the RB10 robot controller. The diffusion policy is not used as a direct low-level joint controller. Instead, it provides a task-space goal for the end-effector, while RMPflow transforms this goal into a dynamically feasible and proximity-aware joint command by incorporating the current robot state, robot-mounted proximity measurements, and joint-level constraints.

The overall data flow is summarized as follows:

```text
Diffusion policy
  publishes /RMP_goal, /RMP_flag
        |
        v
rmpflow_bridge
  /RMP_goal + /RMP_flag -> /goal_pose
        |
        v
RMPflow controller
  subscribes /goal_pose, /obstacles, /RMP_flag, robot state
        ^
        |
proximity_obstacle_bridge
  /proximity_distance1...20 -> /obstacles
        |
        v
RMPflow solve
  target RMP + obstacle RMP + orientation RMP
  + joint-limit RMP + velocity-limit RMP + damping RMP
        |
        v
qdd* -> qdot_cmd, q_cmd
        |
        v
RB10 controller
```

The RMPflow controller receives four classes of inputs:

1. Robot joint state: current joint position and velocity, denoted by `q` and `qdot`.
2. Diffusion-policy target: the end-effector goal pose received through `/RMP_goal` and forwarded as `/goal_pose`.
3. Proximity-based obstacle set: local obstacle primitives generated from `/proximity_distance*` topics and published as `/obstacles`.
4. Execution gate: `/RMP_flag`, which determines whether the RMP solve loop should actively command the robot.

The output of the RMPflow solver is the desired joint acceleration `qdd*`. This acceleration is integrated into a commanded joint velocity `qdot_cmd` and a commanded joint configuration `q_cmd`. In the current experimental configuration, the RB10 is operated in velocity command mode; therefore, `qdot_cmd` is sent to the robot through the RB10 velocity interface, while `q_cmd` is published as `/target_q` and retained as the integrated target joint state. In position command mode, `q_cmd` is sent directly as the robot joint-position command.

## B. ROS 2 Topic-Level Pipeline

The proposed framework is implemented as a ROS 2 pipeline. The topic-level structure is important because each topic corresponds to one conceptual component of the framework.

### 1. Diffusion Goal Interface

The diffusion policy publishes a task-space end-effector goal on:

```text
/RMP_goal : geometry_msgs/Pose
```

The message contains both the desired end-effector position and orientation. The same policy or experiment manager publishes:

```text
/RMP_flag : std_msgs/UInt8
```

When `/RMP_flag == 1`, the RMPflow stack is activated. The `rmpflow_bridge` node converts the external diffusion-policy goal into the internal controller goal:

```text
/RMP_goal -> /goal_pose : geometry_msgs/PoseStamped
```

Thus, from the RMPflow perspective, the learned policy provides a time-varying target pose `x_g = (p_g, R_g)` rather than a direct joint command.

### 2. Proximity Sensor Interface

The robot-mounted proximity sensors publish range measurements:

```text
/proximity_distance1, ..., /proximity_distance20 : sensor_msgs/Range
```

The `proximity_obstacle_bridge` node transforms each valid range measurement from the corresponding sensor frame into the robot base frame. If the sensor pose is given by `T_BS,k = (R_BS,k, p_BS,k)` and the measured range is `d_k`, the detected obstacle point is approximated as

```text
p_o,k = p_BS,k + R_BS,k e_x d_k,
```

where `e_x` is the forward sensing direction of the proximity sensor. Measurements outside the trigger distance or invalid measurements are ignored. Valid detections are converted into obstacle spheres or local surface patches and published as:

```text
/obstacles : visualization_msgs/MarkerArray
```

Although this topic uses marker messages, it is not only a visualization output. The RMPflow controller subscribes to `/obstacles` and converts the markers into collision primitives used by the obstacle RMP.

### 3. Controller Outputs

After solving the RMP problem, the controller publishes the integrated joint target:

```text
/target_q : std_msgs/Float64MultiArray
```

The controller also publishes command/debug outputs such as `/rmp_position_command` and, depending on the selected backend, `/position_controllers/commands`. In the current velocity-mode hardware execution, the robot command vector corresponds to `qdot_cmd`, and the direct RB10 backend sends it to the robot using the velocity command interface. Therefore, the conceptual output of the motion generator is `q_cmd`, but the real-time hardware interface may execute the corresponding velocity command `qdot_cmd`.

## C. RMPflow Formulation

RMPflow represents each objective as a Riemannian Motion Policy (RMP). Each RMP is defined in its own task space by a desired acceleration and an importance metric. Let the `i`-th task map be

```text
x_i = phi_i(q),
```

with Jacobian

```text
J_i(q) = d phi_i(q) / d q.
```

Each task-space RMP produces a pair:

```text
(a_i, M_i),
```

where `a_i` is the desired task-space acceleration and `M_i` is the task-space metric. The metric determines how strongly the corresponding objective influences the final joint motion. The RMP is pulled back to the joint space as

```text
A_i = J_i^T M_i J_i,
f_i = J_i^T M_i (a_i - Jdot_i qdot).
```

The root joint-space metric and force are obtained by summing all RMP contributions:

```text
A = sum_i A_i,
f = sum_i f_i.
```

The final desired joint acceleration is then computed by solving:

```text
qdd* = (A + lambda I)^(-1) f,
```

where `lambda I` is a small regularization term for numerical stability. This formulation allows target tracking, obstacle avoidance, orientation control, joint-limit avoidance, velocity limiting, and damping to be combined in a single continuous optimization problem.

## D. RMP Components

### 1. Target RMP from Diffusion Policy

The target RMP attracts the end-effector toward the diffusion-policy goal. Given the current end-effector position `p = phi_ee(q)`, velocity `pdot = J_ee(q) qdot`, and goal position `p_g`, the target acceleration is implemented in the form:

```text
a_target = k_p (p_g - p) / (||p_g - p|| + eps) - k_d pdot.
```

The corresponding target metric is distance-dependent. Near the goal, the metric can become more isotropic and stronger, improving convergence and reducing residual tracking error. Far from the goal, the metric is shaped to allow other RMPs, especially obstacle avoidance, to influence the motion. The orientation part of the diffusion-policy goal is handled by axis target RMPs, which align the end-effector orientation axes with the desired orientation.

This design separates high-level decision making from low-level safety. The diffusion policy proposes the desired task-space motion, while RMPflow determines how the robot should realize that motion under proximity and kinematic constraints.

### 2. Proximity-Based Obstacle RMP

The obstacle RMP uses the obstacle primitives generated from proximity sensors. For each robot control point `p_c,j(q)` and obstacle primitive `o_i = (p_o,i, r_o,i)`, the signed clearance is computed as

```text
s_ij = ||p_c,j(q) - p_o,i|| - r_c,j - r_o,i.
```

When this clearance becomes small, the obstacle RMP generates a repulsive acceleration and an approach-velocity damping term:

```text
a_obs = a_repulsion(s_ij) + a_damping(s_ij, sdot_ij).
```

The obstacle metric increases as the clearance decreases, which gives collision avoidance a larger influence in the root-space solve. In other words, the robot does not switch from target tracking to obstacle avoidance by a discrete rule. Instead, the obstacle RMP continuously increases its importance as the robot approaches a measured obstacle.

This is the main mechanism by which the robot produces reactive avoidance behavior. If the diffusion policy goal lies behind or near an obstacle, the target RMP continues to attract the end-effector toward the goal, while the obstacle RMP bends the joint-space acceleration away from the detected surface.

### 3. Constraint and Stabilization RMPs

In addition to the target and obstacle RMPs, the controller includes several stabilizing RMPs:

- Joint-limit RMP prevents motion toward joint boundaries.
- Joint-velocity-cap RMP damps motion when joint velocities approach the configured limit.
- Damping RMP suppresses excessive joint velocity and improves smoothness.
- Orientation RMP aligns the end-effector orientation with the diffusion-policy goal.

These terms are not implemented as post-hoc overrides. They are included in the same metric-weighted RMPflow solve. Therefore, the final acceleration `qdd*` is the result of simultaneous arbitration among task execution, obstacle avoidance, and robot constraints.

## E. Command Generation and Robot Behavior

After the RMPflow solver computes `qdd*`, the controller integrates the acceleration over the control period `dt`:

```text
qdot_cmd = qdot + qdd* dt,
q_cmd    = q    + qdot_cmd dt.
```

Before sending a command to the robot, the controller applies command-level guards, including joint-position bounds, joint-velocity saturation, predictive joint-limit checks, and minimum-height safety checks. The resulting command is then sent to the RB10 controller.

The resulting robot behavior can be interpreted as follows:

- If no obstacle is detected, the robot follows the diffusion-policy goal through the target and orientation RMPs.
- If a proximity sensor detects an obstacle, the corresponding obstacle primitive increases the obstacle RMP contribution, causing the robot to deviate from the direct path to the goal.
- If the robot approaches a joint limit or velocity limit, the corresponding constraint RMP and command guard reduce or redirect the motion.
- Once the obstacle influence decreases, the target RMP again dominates and the robot continues toward the diffusion-policy goal.

Thus, the robot behavior is not generated by a sequence of manually defined states such as "move", "avoid", and "resume". Instead, these behaviors emerge from the continuous metric-weighted combination of RMPs.

## F. Difference from a Conventional RMP Controller

A conventional RMP-based controller typically assumes a fixed goal and known obstacle positions. Its structure can be summarized as:

```text
Fixed goal + known obstacles -> RMPflow -> joint command.
```

The proposed framework extends this structure in three important ways.

### 1. Diffusion-Guided Target RMP

The target is not a manually specified static waypoint. It is generated by a diffusion policy and streamed to the controller through `/RMP_goal`. The learned policy provides task-level adaptability, while RMPflow provides model-based reactive control. This prevents the learned policy from directly commanding unsafe joint motion and allows the final command to remain constrained by obstacle and joint-limit RMPs.

### 2. Proximity-Aware Local Obstacle Field

Instead of relying only on manually placed or externally reconstructed obstacles, the controller builds obstacle primitives directly from robot-mounted proximity sensors. This allows the obstacle RMP to respond to local, close-range hazards around the robot body and end-effector. The method is particularly suitable for manipulation tasks where the relevant obstacle geometry may be close, partial, or only locally observable.

### 3. Unified RMP Arbitration

The framework does not use a rule-based switch between learned motion and safety behavior. The target RMP, obstacle RMP, orientation RMP, joint-limit RMP, velocity-limit RMP, and damping RMP are all represented in the same RMPflow graph. Their effects are combined through the root-space metric and force:

```text
A = sum_i J_i^T M_i J_i,
f = sum_i J_i^T M_i (a_i - Jdot_i qdot).
```

This yields a continuous arbitration mechanism. The diffusion-policy goal determines the nominal motion, while proximity and constraint RMPs reshape the motion whenever safety or feasibility becomes more important.

## G. Paper-Level Contribution Statement

The proposed framework contributes a proximity-aware RMPflow architecture for diffusion-guided manipulation on the RB10 robot. A diffusion policy generates task-space end-effector goals, which are converted into target RMPs rather than direct robot commands. Robot-mounted proximity sensors generate local obstacle primitives online, which are converted into obstacle RMPs. These learned goals and sensor-derived obstacle constraints are then unified with orientation, joint-limit, velocity-limit, and damping RMPs in a single RMPflow solve. The resulting desired joint acceleration is integrated, filtered by command-level safety guards, and sent to the RB10 controller as a joint position or velocity command.

Compared with a simple RMP controller with a fixed target and predefined obstacles, the proposed approach enables learned goal generation, local proximity-aware obstacle avoidance, and continuous metric-based arbitration between task execution and safety constraints. This allows the robot to follow diffusion-policy goals while reactively bending its motion away from nearby obstacles detected by onboard proximity sensors.

## H. Suggested Figure Caption

Fig. X. Proposed diffusion-guided and proximity-aware RMPflow framework. The diffusion policy publishes a task-space goal on `/RMP_goal`, which is forwarded as `/goal_pose` and used by the target RMP. Robot-mounted proximity sensors publish `/proximity_distance*` range measurements, which are transformed into obstacle primitives and published as `/obstacles`. The RMPflow controller combines target, obstacle, orientation, joint-limit, velocity-limit, and damping RMPs to compute the desired joint acceleration. The acceleration is integrated into joint commands and sent to the RB10 robot controller, producing goal-directed motion with reactive local obstacle avoidance.

