# Stage-2 Tangent Escape RMP: Minimal GDS/Lyapunov Note

This note records what can be claimed before adding Stage-3 score-based
multi-branch tangent selection.  It is intentionally limited to the Stage-2
single tangent branch that is currently implemented in the Pinocchio direct
RMP solver.

## 1. Purpose

Stage 2 changes the tangent escape behavior from a direct task-space
acceleration injection into a scalar tangent branch inside the RMP tree.

The goal of this note is not to prove that the full robot controller is
globally stable.  The goal is narrower:

- define the scalar tangent task used by the Stage-2 escape leaf;
- show that, while the escape mode is fixed, the leaf has a
  potential-damping form;
- identify which parts are not covered by this local Lyapunov argument;
- make clear what must be revisited before or during Stage 3.

## 2. Fixed-Mode Definitions

For a proximity sensor/control point \(i\), the control-point position is

\[
x_i(q) \in \mathbb{R}^3,
\qquad
\dot{x}_i = J_i(q)\dot{q}.
\]

When the Stage-2 escape mode first activates for this control point, the
implementation stores:

\[
x_{0,i}=x_i(q_{on}),
\qquad
\hat{t}_{0,i}=\hat{t}_i(q_{on}),
\]

where \(\hat{t}_{0,i}\) is the unit tangent direction selected at activation
time.  During the active fixed mode, the scalar tangent coordinate is

\[
s_i = \hat{t}_{0,i}^{T}(x_i - x_{0,i}),
\]

and its velocity is

\[
\dot{s}_i = \hat{t}_{0,i}^{T}\dot{x}_i
          = \hat{t}_{0,i}^{T}J_i(q)\dot{q}.
\]

The scalar task Jacobian used by the implementation is therefore

\[
J_{s,i} = \hat{t}_{0,i}^{T}J_i(q).
\]

The scalar curvature term is

\[
\dot{J}_{s,i}\dot{q}
=\hat{t}_{0,i}^{T}\dot{J}_i(q,\dot{q})\dot{q},
\]

because \(\hat{t}_{0,i}\) is fixed inside this active mode.

## 3. Implemented Stage-2 Leaf

The Stage-2 target tangent displacement is

\[
s_i^* = \ell_{esc}.
\]

The implemented scalar acceleration command is

\[
a_{s,i}
= k_s(s_i^* - s_i) - b_s\dot{s}_i,
\]

where:

- \(k_s\) is `tangent_escape_rmp_position_gain`;
- \(b_s\) is `tangent_escape_rmp_damping_gain`;
- \(\ell_{esc}\) is `tangent_escape_rmp_escape_length`.

The scalar metric is

\[
M_{s,i} = \gamma_i m_s,
\]

where:

- \(m_s\) is `tangent_escape_rmp_metric_scalar`;
- \(\gamma_i\in[0,1]\) is the distance/alignment activation.

The implementation then uses the normal RMP pullback for this scalar leaf:

\[
A_i = J_{s,i}^{T} M_{s,i} J_{s,i},
\]

\[
f_i = J_{s,i}^{T}M_{s,i}
\left(a_{s,i}-\dot{J}_{s,i}\dot{q}\right).
\]

This is the same scalar-leaf accumulation pattern used by other scalar RMP
terms in the solver.

## 4. Fixed-Mode Potential-Damping Interpretation

For the local argument, assume a fixed active mode:

- \(x_{0,i}\) is fixed;
- \(\hat{t}_{0,i}\) is fixed;
- \(\gamma_i\) is treated as constant over the local interval;
- the scalar acceleration clamp is inactive;
- \(m_s > 0\), \(k_s > 0\), and \(b_s > 0\).

Under these assumptions, the natural scalar force is

\[
F_{s,i}
=M_{s,i}a_{s,i}
=\gamma_i m_s
\left(k_s(s_i^*-s_i)-b_s\dot{s}_i\right).
\]

This is equivalent to a one-dimensional mechanical system with effective
potential

\[
\Phi_i(s_i)
=\frac{1}{2}\gamma_i m_s k_s(s_i-s_i^*)^2
\]

and effective damping

\[
B_i = \gamma_i m_s b_s.
\]

Indeed,

\[
-\frac{\partial \Phi_i}{\partial s_i}
=\gamma_i m_s k_s(s_i^*-s_i),
\]

so

\[
F_{s,i}
=-\frac{\partial \Phi_i}{\partial s_i} - B_i\dot{s}_i.
\]

Therefore, inside a fixed mode with constant activation, the Stage-2 tangent
escape leaf is compatible with a potential-damping scalar RMP form.

## 5. Minimal Lyapunov Argument

For the isolated scalar branch in a fixed mode, define

\[
V_i(s_i,\dot{s}_i)
=\frac{1}{2}M_{s,i}\dot{s}_i^2+\Phi_i(s_i).
\]

With constant \(M_{s,i}=\gamma_i m_s\), the closed scalar dynamics are

\[
M_{s,i}\ddot{s}_i
=-\frac{\partial \Phi_i}{\partial s_i}-B_i\dot{s}_i.
\]

Then

\[
\dot{V}_i
=M_{s,i}\dot{s}_i\ddot{s}_i
 + \frac{\partial \Phi_i}{\partial s_i}\dot{s}_i.
\]

Substituting the dynamics:

\[
\dot{V}_i
=\dot{s}_i
\left(
-\frac{\partial \Phi_i}{\partial s_i}-B_i\dot{s}_i
\right)
 + \frac{\partial \Phi_i}{\partial s_i}\dot{s}_i
=-B_i\dot{s}_i^2.
\]

Since \(B_i=\gamma_i m_s b_s\ge 0\),

\[
\dot{V}_i \le 0.
\]

Thus, for the isolated fixed-mode scalar branch, the stored energy is
non-increasing.  If \(\gamma_i>0\), \(m_s>0\), and \(b_s>0\), damping removes
energy whenever \(\dot{s}_i\neq 0\).

## 6. What This Does Not Prove

This note does not prove global stability of the full controller.

The following effects are outside the minimal fixed-mode argument:

- state-dependent activation \(\gamma_i(q,\dot{q})\);
- activation on/off switching;
- storing and resetting \(x_{0,i}\) and \(\hat{t}_{0,i}\);
- scalar acceleration clamping by `tangent_escape_rmp_max_accel`;
- interaction with target, collision, damping, joint-limit, and
  joint-velocity-limit RMPs;
- root acceleration clamping;
- command integration and hardware velocity/position guards;
- discontinuous proximity measurements;
- future score-based branch weighting;
- future previous-direction memory, blocked memory, or stuck timers.

Because \(x_{0,i}\) and \(\hat{t}_{0,i}\) are stored internal state variables,
the overall Stage-2 escape system is already hybrid at the mode level.  The
fixed-mode leaf can be described as GDS-compatible, but the full switched
closed-loop system still requires separate hybrid or CLF/CBF-style analysis.

## 7. Current Implementation Mapping

The current implementation corresponds to this note as follows:

- `src/pinocchio_direct_solver.cpp`
  - `accumulate_tangent_escape(...)`
  - `tangent_escape_rmp_leaf_mode == "gds"`
  - stores `origin` as \(x_{0,i}\)
  - stores `tangent` as \(\hat{t}_{0,i}\)
  - computes `scalar_s`
  - computes `scalar_velocity`
  - computes `desired_tangent_accel`
  - pulls back through `accumulate_scalar_leaf(...)`

- `include/rb10_rmpflow_rviz/rmp_eigen_solver.hpp`
  - `TangentEscapeRmpParams::position_gain`
  - `TangentEscapeRmpParams::escape_length`
  - `TangentEscapeRmpParams::damping_gain`
  - `TangentEscapeRmpParams::metric_scalar`
  - `TangentEscapeRmpParams::leaf_mode`

- `scripts/check_tangent_escape_closed_loop.py`
  - checks whether the Stage-2 branch entered GDS mode;
  - reports scalar progress, active duration, clearance, qdd saturation, jerk,
    and goal-error progress;
  - does not claim formal stability.

## 8. Before Stage 3

Before adding score-based multi-branch selection, the following Stage-2 claims
are acceptable:

1. The active tangent escape leaf is represented as a scalar task
   \(s_i=\hat{t}_{0,i}^{T}(x_i-x_{0,i})\).
2. Inside a fixed active mode, with fixed activation, the branch has an
   effective quadratic potential and positive damping.
3. The scalar metric is nonnegative because
   \(M_{s,i}=\gamma_i m_s\), with \(\gamma_i\in[0,1]\) and \(m_s\ge0\).
4. The isolated fixed-mode branch admits the Lyapunov candidate
   \(V_i=\frac{1}{2}M_{s,i}\dot{s}_i^2+\Phi_i\), with
   \(\dot{V}_i=-B_i\dot{s}_i^2\le0\), in the unclamped local region.
5. Switching, state-dependent activation, and full closed-loop stability are
   explicitly not claimed by this fixed-mode proof.

## 9. Stage-3 Implication

Stage 3 will introduce multiple tangent branches and score-based soft weights.
Once weights depend on the robot state or sensor state, multiplying each branch
by a weight is not automatically covered by the simple fixed-mode Lyapunov
argument above.

For Stage 3, the correct theoretical path is:

- keep each tangent branch in a potential-damping/GDS-compatible form;
- use smooth nonnegative weights where possible;
- document that ordinary weighted pullback is a heuristic unless the
  RMPfusion-style correction term or another stability wrapper is added;
- treat hard memory, stuck timers, and branch switching as hybrid supervisor
  logic, not as a single smooth GDS.

The practical development may still proceed to Stage 3 after this note, but
the theoretical claim should remain scoped exactly as described here.
