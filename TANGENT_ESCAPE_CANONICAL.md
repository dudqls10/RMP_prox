# Canonical Tangent-Escape RMP

## Status and scope

> Compatibility design only. This velocity-reference/RELEASE implementation is
> retained for explicit rollback and A/B validation, but the normal fake-proximity
> profile now selects `tangent_escape_rmp_acceleration_model: "risk_damped"`.

The final Escape design is:

> Score-supervised, risk-conditioned, tangential-velocity canonical RMP.

The implementation selects one control-point/obstacle pair and one world-frame tangent
direction, tracks a bounded tangent velocity, and adds one rank-one canonical RMP to the
root solve.

This implementation can still be selected explicitly with:

```yaml
tangent_escape_rmp_acceleration_model: "canonical_velocity"
```

## Controller decomposition

Let the robot state be \(q,\dot q\in\mathbb R^6\). All non-Escape leaves are accumulated
first:

\[
M_0=\sum_{\ell\ne\mathrm{Escape}}J_\ell^\top M_\ell J_\ell,
\qquad
f_0=\sum_{\ell\ne\mathrm{Escape}}
J_\ell^\top M_\ell(a_\ell-c_\ell).
\]

Thus \(M_0,f_0\) include Target, Collision for every control-point/obstacle pair,
orientation, joint-limit, velocity-cap, damping, and enabled external leaves.

A separate nominal solve excludes both Collision and Escape:

\[
\ddot q_{\mathrm{nom}}
=\operatorname{resolve}(M_{\mathrm{nom}},f_{\mathrm{nom}}).
\]

For control point \(i\), nominal intent is

\[
v_{i,\mathrm{nom}}
=J_i\left(\dot q+T_n\ddot q_{\mathrm{nom}}\right).
\]

This avoids treating the TCP goal as a position goal for a mid-link control point.

## Activation and state

The quintic smoothstep is

\[
H(r)=6r^5-15r^4+10r^3,\qquad r\in[0,1].
\]

Define

\[
A_\uparrow(x;x_0,x_1)
=H\left(\operatorname{clip}\frac{x-x_0}{x_1-x_0},0,1\right),
\qquad
A_\downarrow=1-A_\uparrow.
\]

Distance and nominal-blocking gates are

\[
\alpha_d
=H\left(\operatorname{clip}
\frac{d_{\mathrm{inf}}-d}
{d_{\mathrm{inf}}-d_{\mathrm{safe}}},0,1\right),
\]

\[
\beta=\hat v_{i,\mathrm{nom}}^\top\hat o,
\qquad
\alpha_b
=H\left(\operatorname{clip}
\frac{\beta-\beta_{\mathrm{on}}}
{\beta_{\mathrm{full}}-\beta_{\mathrm{on}}},0,1\right),
\]

where \(\hat o\) points from the control point toward the obstacle. If nominal speed is
below the intent threshold, \(\alpha_b=0\).

Filtered TCP goal progress is

\[
P_g=-\operatorname{LPF}(\dot e_g),
\qquad
e_g=\lVert x_{\mathrm{tcp}}-x_g\rVert.
\]

The stuck confidence is

\[
\alpha_{\mathrm{stuck}}
=A_\downarrow(P_g;P_{\mathrm{low}},P_{\mathrm{ok}})
 A_\downarrow(\lVert J_i\dot q\rVert;v_{\mathrm{still}},v_{\mathrm{move}})
 A_\uparrow(\lVert v_{i,\mathrm{nom}}\rVert;
             v_{\mathrm{intent,on}},v_{\mathrm{intent,full}}),
\]

where juxtaposition above denotes multiplication of the three gates.

The pair blockage and raw Escape activation are

\[
B_{ih}=\alpha_d\alpha_b,
\qquad
\rho_{ih}
=\alpha_d\alpha_b
\left[\gamma_{\mathrm{pre}}
+(1-\gamma_{\mathrm{pre}})\alpha_{\mathrm{stuck}}\right].
\]

Only one pair is owned at a time. A new pair requests a switch only when

\[
B_{\mathrm{new}}>B_{\mathrm{current}}+\Delta B.
\]

The selected pair's filtered activation satisfies

\[
\dot z
=\operatorname{clip}
\left(\frac{\rho-z}{\tau_z},
      -\dot z_{\downarrow},
      \dot z_{\uparrow}\right),
\qquad 0\le z\le1.
\]

The persistent state is:

- mode and handoff phase;
- active and pending pair;
- fixed world-frame tangent \(t^\ast\);
- \(z,\lambda,b_{\mathrm{drv}},b_{\mathrm{rel}},v_d,\dot v_d\);
- filtered goal progress and previous goal error;
- previous final joint acceleration;
- commanded/actual tangent-distance integrals;
- episode-scoped failed-direction memory.

The high-level modes are `OFF`, `PREVENT`, `RECOVERY`, `RELEASE`, and `RESELECT`.

- `OFF`: no branch, drive, release gate, metric, or force;
- `PREVENT`: blocked nominal intent with the low \(v_{\mathrm{pre}}\) drive;
- `RECOVERY`: high stuck confidence, continuously increasing the drive toward
  \(v_{\mathrm{rec}}\);
- `RELEASE`: drive goes to zero while the release metric brakes residual tangent speed;
- `RESELECT`: the current branch is brought to zero effect before pair/direction swap.

## One direction, latched

For the active obstacle direction \(\hat o\),

\[
P_T=I-\hat o\hat o^\top.
\]

The first tangent basis vector is the normalized \(P_Tv_{i,\mathrm{nom}}\), when valid.
Fallbacks are the projected previous tangent, projected sensor bias, and finally a
deterministic orthogonal world axis. The second basis vector is

\[
v=\hat o\times u.
\]

Every tangent direction is

\[
t(\theta)=u\cos\theta+v\sin\theta.
\]

Sixteen coarse seeds are evaluated at

\[
\theta_k=\frac{2\pi k}{16},\qquad k=0,\ldots,15.
\]

Projected previous direction, exact nominal tangent, and positive/negative sensor-bias
directions are added as non-duplicate seeds. After the best seed is found, a bounded
one-dimensional search refines its angle within the neighboring coarse interval.

The final \(t^\ast\) is latched in the world frame. It is not reprojected or rotated
while its metric is nonzero. Excessive \(\lvert {t^\ast}^\top\hat o\rvert\), a new
critical sensor, pair change, or branch failure requests `RESELECT`.

A new direction is accepted only when

\[
S_{\mathrm{new}}>S_{\mathrm{current}}+\Delta S.
\]

Scoring runs on Escape entry, pair change, branch failure, a new critical sensor, or
tangent invalidation. There is no per-cycle softmax and no simultaneous candidate
pullback.

## Candidate trial solve and score

For candidate \(t_k\),

\[
j_k=t_k^\top J_i,\qquad
v_{t,k}=j_k\dot q,\qquad
c_k=t_k^\top c_i.
\]

Entry and reselection trials use the planned drive reference, not the current
\(b_{\mathrm{drv}}\), which is zero during handoff:

\[
v_{\mathrm{trial}}
=v_{\mathrm{pre}}
+(v_{\mathrm{rec}}-v_{\mathrm{pre}})\alpha_{\mathrm{stuck}},
\]

\[
z_{\mathrm{trial}}=\max(z,\rho),\qquad
m_k=m_ez_{\mathrm{trial}},
\]

\[
a_k=\operatorname{sat}
\left(k_v(v_{\mathrm{trial}}-v_{t,k})\right).
\]

The candidate adds only a rank-one term:

\[
\Delta M_k=m_kj_k^\top j_k,
\qquad
\Delta f_k=m_kj_k^\top(a_k-c_k).
\]

It is evaluated using the same resolver and candidate-dependent metric scaling as the
real root solve:

\[
\ddot q_k
=\operatorname{resolve}(M_0+\Delta M_k,\ f_0+\Delta f_k).
\]

For both trial and final solves, the current resolver is

\[
s(M)=\max\left(0.01\max_{a,b}|M_{ab}|,\ 1\right),
\]

\[
\operatorname{resolve}(M,f)
=\left(\frac{M}{s(M)}+\epsilon I\right)^{-1}
\frac{f}{s(M)}.
\]

FK and all non-Escape leaves are not recomputed per candidate.

For preview horizon \(T_p\),

\[
\Delta p_{j,k}
=T_pJ_j\dot q
+\frac12T_p^2(J_j\ddot q_k+c_j),
\]

\[
\hat x_{\mathrm{tcp},k}
=x_{\mathrm{tcp}}+T_pJ_{\mathrm{tcp}}\dot q
+\frac12T_p^2(J_{\mathrm{tcp}}\ddot q_k+c_{\mathrm{tcp}}).
\]

After normalization, the score is

\[
S_k
=2.0G_k-1.5R_{\mathrm{sector},k}
+0.3C_k-2.0B_k-0.2Q_k.
\]

With \(\ddot q_0=\operatorname{resolve}(M_0,f_0)\), define the corresponding TCP
preview error \(e_0^+\), and let \(e_k^+\) be the candidate preview error. The normalized
terms are

\[
G_k=\operatorname{clip}
\left(\frac{e_0^+-e_k^+}{g_{\mathrm{scale}}},-1,1\right),
\]

\[
R_{\mathrm{sector},k}
=1-\exp\left(
-\frac{1}{r_{\mathrm{scale}}}
\sum_{(j,h)\in\mathcal N}
\alpha_d(d_{jh})
\left[
\frac{\hat o_{jh}^\top\Delta p_{j,k}}
{\lVert\Delta p_{j,k}\rVert+\epsilon}
\right]_+
\right),
\]

\[
C_k=\frac{1+t_k^\top t_{\mathrm{prev}}}{2},
\qquad
Q_k=\operatorname{clip}
\left(
\frac{\lVert\ddot q_k-\ddot q_{\mathrm{prev}}\rVert}
{a_{\mathrm{jump}}},
0,1\right).
\]

\(B_k\) is the episode-memory penalty defined below. All scales must be positive and
are configuration parameters.

Candidates are rejected before scoring when they violate tangent tolerance, predicted
joint position/velocity/acceleration limits, minimum realized tangent displacement, or
a hard observed-sector risk threshold.

## Canonical velocity leaf

During an active branch,

\[
y={t^\ast}^\top p_i(q),
\qquad
J_y={t^\ast}^\top J_i,
\]

\[
v_t=J_y\dot q,
\qquad
c_y={t^\ast}^\top c_i.
\]

There is no virtual origin, target distance, or Escape spring.

The drive reference and its filter are

\[
v_r=b_{\mathrm{drv}}
\left[v_{\mathrm{pre}}
+(v_{\mathrm{rec}}-v_{\mathrm{pre}})
\alpha_{\mathrm{stuck}}\right],
\]

\[
\dot v_d=\frac{v_r-v_d}{\tau_v}.
\]

The configured speeds satisfy

\[
0<v_{\mathrm{pre}}\le v_{\mathrm{rec}}\le v_{\max}.
\]

The bounded canonical acceleration is

\[
u=\dot v_d+k_v(v_d-v_t),
\qquad
a_y=a_{\max}\tanh\left(\frac{u}{a_{\max}}\right).
\]

For \(0\le v_{\mathrm{stop}}<v_{\mathrm{hold}}\), the residual-speed gate is

\[
r_v
=H\left(\operatorname{clip}
\frac{\lvert v_t\rvert-v_{\mathrm{stop}}}
{v_{\mathrm{hold}}-v_{\mathrm{stop}}},0,1\right).
\]

The scalar metric is

\[
M_y
=m_e\lambda
\left[z+(1-z)b_{\mathrm{rel}}r_v\right]\ge0.
\]

The root contribution is

\[
M_{\mathrm{esc}}^{(q)}
=J_y^\top M_yJ_y,
\]

\[
f_{\mathrm{esc}}^{(q)}
=J_y^\top M_y(a_y-c_y).
\]

The final solve is

\[
\ddot q
=\operatorname{resolve}
\left(
M_0+J_y^\top M_yJ_y,\
f_0+J_y^\top M_y(a_y-c_y)
\right).
\]

The Escape metric is PSD and rank one. Collision RMP continues to own normal-direction
safety for every control-point/obstacle pair.

## Zero-effect handoff

The quintic ramp

\[
h(\xi)=10\xi^3-15\xi^4+6\xi^5
\]

is used for \(\lambda\), \(b_{\mathrm{drv}}\), and \(b_{\mathrm{rel}}\).

Direction or control-point changes follow:

1. ramp \(b_{\mathrm{drv}}\) down and \(v_d\) toward zero;
2. ramp \(\lambda:1\rightarrow0\);
3. at exactly \(\lambda=0\), record failure if warranted and replace the pair/tangent;
4. ramp \(\lambda:0\rightarrow1\);
5. ramp \(b_{\mathrm{drv}}\) up.

At the swap,

\[
M_y=0,\qquad f_y=0,
\]

so a discontinuous task-axis change has zero direct effect on the root RMP.

`RELEASE` first raises \(b_{\mathrm{rel}}\), lowers the drive, and uses \(r_v\) to retain
braking authority. Once \(\lvert v_t\rvert<v_{\mathrm{stop}}\), \(\lambda\) ramps down
only after the filtered desired speed is also below \(v_{\mathrm{stop}}\), and the state
then returns to `OFF`. A timeout does not remove the branch while residual tangent speed
remains.

`RESELECT` uses the same braking sequence, but it cannot wait forever for a base leaf
that keeps driving the old scalar axis. It therefore begins the quintic
\(\lambda:1\rightarrow0\) ramp when either both speeds satisfy the stop threshold or

\[
t_{\mathrm{brake}}\ge
\max(T_{\mathrm{handoff}},3\tau_v).
\]

This bounded wait adds no public parameter. It does not weaken the zero-effect swap:
the old pair/tangent is still replaced only after the ramp reaches exactly
\(\lambda=0\). `RELEASE` retains the stricter residual-speed condition.

## Episode memory

While driving,

\[
\chi=\lambda z,
\]

\[
s_{\mathrm{cmd}}=\int\chi[v_d]_+\,dt,
\qquad
s_{\mathrm{act}}=\int\chi[v_t]_+\,dt,
\]

\[
\eta_{\mathrm{move}}
=\frac{s_{\mathrm{act}}}{s_{\mathrm{cmd}}+\epsilon}.
\]

A branch is recorded as failed only when:

- \(s_{\mathrm{cmd}}\ge s_{\mathrm{test}}\);
- \(\eta_{\mathrm{move}}<\eta_{\min}\); and
- observed sector risk increased above its noise threshold, or predicted
  joint/actuator saturation blocked motion.

Failed directions are stored as continuous world-frame vectors, not coarse candidate
slots. Their score penalty is an angular Gaussian maximum:

\[
B(t)
=\max_r b_r
\exp\left(
-\frac{1-t^\top t^{\mathrm{fail}}_r}{2\sigma_B^2}
\right).
\]

Memory persists for the current Escape episode and is reset when the active sensor
control point or obstacle episode changes, or the episode ends. It has no unconditional
wall-clock decay.

## Parameters

Canonical parameters are grouped as follows:

- geometry: safe/influence distances, blocking beta thresholds, and
  `tangent_escape_rmp_normal_tolerance`;
- activation: \(\gamma_{\mathrm{pre}}\), progress LPF and thresholds, motion/intent
  thresholds, \(z\) time constant and rise/fall limits;
- selection: 16 coarse candidates, refinement iterations, pair/score switch margins,
  preview horizon, hard feasibility thresholds;
- score: goal, sector-risk (including `tangent_escape_rmp_sector_risk_weight`),
  continuity, memory, and acceleration-jump weights and normalization scales;
- drive: \(m_e,v_{\mathrm{pre}},v_{\mathrm{rec}},v_{\max},\tau_v,k_v,a_{\max}\);
- release/handoff: \(v_{\mathrm{stop}},v_{\mathrm{hold}}\), drive, release, and
  \(\lambda\) ramp durations;
- memory: \(s_{\mathrm{test}},\eta_{\min},\sigma_B\), and risk-noise threshold;
- feasibility: joint position buffers and velocity/acceleration limits;
- runtime: internal `control_dt`, derived from the controller `control_rate`.

The implementation always uses exactly 16 coarse seeds before continuous refinement.

## Diagnostics identifiers

The canonical debug record uses a fixed 61-value wire prefix and these IDs:

| Field | ID | Meaning |
|---|---:|---|
| `schema_id` | 6 | Fixed canonical wire-schema tag |
| `state_mode_id` | 0 | `OFF` |
|  | 1 | `PREVENT` |
|  | 2 | `RECOVERY` |
|  | 3 | `RELEASE` |
|  | 4 | `RESELECT` |
| `handoff_phase_id` | 0 | `OFF` |
|  | 1 | `ENGAGE` |
|  | 2 | `DRIVE` |
|  | 3 | `RELEASE_DRIVE_DOWN` |
|  | 4 | `RELEASE_BRAKE` |
|  | 5 | `RELEASE_LAMBDA_DOWN` |
|  | 6 | `RESELECT_DRIVE_DOWN` |
|  | 7 | `RESELECT_BRAKE` |
|  | 8 | `RESELECT_LAMBDA_DOWN` |

Diagnostics expose the active CP, episode key, tangent, clearance,
\(\alpha_d,\alpha_b,\alpha_{\mathrm{stuck}},\rho,z,\lambda,b_{\mathrm{drv}},
b_{\mathrm{rel}},v_r,v_d,v_t,a_y,M_y\), transition reason, memory statistics, and
per-candidate feasibility and score terms.

The trace logger accepts `schema_id=6` and decodes only the canonical slot names and
units.

## Initial implementation assumptions

- Obstacle episodes use a stable marker/detection source ID. Multiple spheres generated
  from one marker share that ID, and the nearest sphere is retained per
  `(control point, source ID)` pair.
- Tagged sensor obstacles may activate only their matching sensor CP. Untagged global
  obstacles are ignored by Escape. Collision still evaluates all CP/obstacle pairs.
- The Escape timestep is fixed and derived internally as
  `control_dt = 1 / control_rate`; it is not a public ROS parameter and wall-clock time
  is not read inside the solver. Direct library users set
  `TangentEscapeRmpParams::control_dt`.
- Actuator saturation is approximated using configured joint-acceleration bounds because
  the controller's post-solve clamp is not fed back into the leaf.
- The candidate acceleration bound is additionally capped by the controller's global
  joint-acceleration limit.
- Actual CP motion uses the solver input \(\dot q\), which may be a configured blend of
  measured and virtual velocity.
- Candidate trials are stateless. Batch/offline solves must freeze or disable the hybrid
  Escape state rather than advancing it once per batch sample.

## Recommended validation sequence

1. **Disabled regression:** build and run with Escape disabled; verify identical
   non-Escape \(M,f,\ddot q\).
2. **Math/unit checks:** quintic endpoints, PSD/rank-one metric, bounded acceleration,
   release gate, and exactly zero metric/force at \(\lambda=0\).
3. **Candidate checks:** compare each trial against the production root resolver;
   validate hard gates, score normalization, 16-seed coverage, and refinement.
4. **Single-sensor fake obstacle:** verify PREVENT then RECOVERY, one pair and one tangent
   only, fixed world tangent, and Collision normal response on all CPs.
5. **Handoff stress:** force direction, CP, and obstacle-normal changes; verify every swap
   occurs at \(\lambda=0\) without a joint-acceleration impulse.
6. **Release test:** remove blockage and confirm drive decay, residual tangent braking,
   and clean return to `OFF`.
7. **Memory test:** block one branch, satisfy the commanded-distance failure gate, confirm
   reselection and episode-local memory reset.
8. **Closed-loop simulation:** exercise joint limits, adjacent sensor sectors, noisy
   detections, goal changes, and repeated local-minimum scenarios.
9. **Low-speed hardware validation:** start with conservative metric, speed, and
   acceleration limits. Enable by default only after bounded commands, reliable release,
   and repeatable collision clearance are demonstrated.

This canonical leaf improves local-minimum recovery but does not provide a formal hard
collision guarantee. A CBF/QP safety layer remains the appropriate future addition for
strict safety constraints.
