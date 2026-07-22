# Risk-Damped Tangent Escape RMP

## Runtime selection

The normal fake-proximity profile selects:

```yaml
enable_tangent_escape_rmp: true
tangent_escape_rmp_acceleration_model: "risk_damped"
```

The previous `canonical_velocity` implementation remains compiled for an explicit
rollback/A-B run. It is not evaluated and its state is not updated while
`risk_damped` is selected.

## Active leaf

For every sensed control-point/obstacle pair, clearance uses the same geometry and
margin as Collision RMP. Distance and nominal-blocking gates produce

\[
B=\alpha_d(d)\alpha_b(\beta).
\]

There is no activation filter: the current sample's `B` is used directly. If the
active obstacle disappears or its `B` becomes zero, Escape contributes zero metric
and zero force in that same solve.

Only one pair and one world-frame tangent direction are latched at a time. All pairs
remain available to Collision RMP and are still evaluated for Escape scoring. The
existing 16-direction coarse search, local angular refinement, continuity score,
sector risk, and failed-direction memory choose the tangent direction; the score does
not scale the acceleration.

For the selected pair, only clearance rate is filtered. Its history is keyed by
control-point index and sensor `source_id`, and is discarded when that pair is absent.
Thus a first or reappearing sample starts with \(\dot d_f=0\).

Define

\[
d_+=\max(d,0),\qquad v_c=\max(0,-\dot d_f),
\]

\[
g_v(\dot d_f)=\frac{1}{1+\exp(\dot d_f/\ell_v)},
\]

\[
a_{\rm risk,raw}
=k_{t,p}\exp(-d_+/\sigma_t)
+g_v(\dot d_f)
\frac{k_{t,v}v_c}{d_+/\sigma_v+\epsilon}.
\]

To keep the zero-input equilibrium speed within the configured Cartesian tangent
speed, the implementation also applies

\[
a_{\rm risk}=\min(a_{\rm risk,raw},b_t v_{t,\max}).
\]

With \(v_t=t^{*\top}J_i\dot q\), the scalar acceleration and metric are

\[
a_y=a_{\max}\tanh\!\left(\frac{a_{\rm risk}-b_t v_t}{a_{\max}}\right),
\qquad
m_y=m_t\lambda B.
\]

The root contribution is

\[
M_{\rm esc}=J_y^\top m_yJ_y,
\qquad
f_{\rm esc}=J_y^\top m_y(a_y-c_y),
\]

where \(J_y=t^{*\top}J_i\) and \(c_y=t^{*\top}\dot J_i\dot q\).

## Switching contract

- First `OFF -> active` entry may start with \(\lambda=1\), because no previous
  Escape axis exists.
- A pair or direction replacement ramps the old \(\lambda\) to zero, changes the
  axis only at zero effect, then ramps the new \(\lambda\) from zero to one.
- If the active pair vanishes, `B=0` overrides the handoff state immediately. There
  is no RELEASE phase and no residual-velocity braking metric.
- Stuck/progress signals are used only to detect a failed branch and request a
  reselect. They do not activate Escape or scale `a_risk`.

## Active parameter mapping

- `tangent_escape_rmp_metric_scalar`: sets \(m_t\) directly. The current value is
  `50000.0`; it is independent from Collision's metric scalar.
- `tangent_escape_rmp_clearance_margin`: independent clearance margin. The current
  value is `0.0 m`, numerically equal to Collision but not read from it.
- `...safe_distance`, `...influence_distance`: distance gate \(\alpha_d\).
- `...goal_block_beta_on`, `...goal_block_beta_full`: blocking gate \(\alpha_b\).
- `...risk_distance_gain`, `...risk_distance_scale`: \(k_{t,p},\sigma_t\). The
  current distance gain is `0.30`, reduced to soften the zero-closing-speed
  tangent drive.
- `...risk_approach_gain`, `...risk_approach_distance_scale`,
  `...risk_approach_epsilon`: \(k_{t,v},\sigma_v,\epsilon\).
- `...risk_velocity_gate_scale`: \(\ell_v\).
- `...risk_clearance_rate_filter_time_constant`: clearance-rate LPF time constant.
- `...risk_tangent_damping_gain`: \(b_t\).
- `...max_speed`, `...max_accel`: \(v_{t,\max},a_{\max}\).
- `...handoff_duration`: the \(\lambda\) ramp duration.

The leaf metric is PSD and rank one, and the acceleration is bounded. Because tangent
selection and failure-memory handoffs are hybrid/non-GDS operations, those facts do
not by themselves constitute a global Lyapunov collision-safety proof.
