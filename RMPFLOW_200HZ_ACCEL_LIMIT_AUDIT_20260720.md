# Tangent Escape: 200 Hz and task-acceleration-limit audit

Date: 2026-07-20

## Configuration correction

`config/params.yaml` specified `control_rate: 200.0`, but
`rb10_rmpflow_fake_proximity.launch.py` declared a `100.0` Hz launch default
and passed it after the parameter file. The launch value therefore overrode
the YAML value.

The fake-proximity launch default is now `200.0` Hz. A controller startup log
and fresh topic intervals both confirmed the new run at approximately 200 Hz.

The Escape task-acceleration setting is now:

```yaml
tangent_escape_rmp_max_accel: 0.0
```

The implementation interprets a non-positive value as no task-space
saturation:

\[
a_y=\dot v_d+k_v(v_d-v_t).
\]

The production controller still applies the global per-joint limit
\(\lvert\ddot q_i\rvert\le10\ {\rm rad/s^2}\).

## Why the old 100/200 comparison was invalid

The earlier 100 Hz run and the first fixed-obstacle 200 Hz run did not have
identical obstacle inputs. During the old 100 Hz reverse path, the obstacle
was moved about 2.44 seconds after the reverse goal was issued:

- initial: `(-0.1392, -0.5426)`
- intermediate: `(-0.3733, -0.7117)`
- final: `(+0.2591, -0.7117)`

The 200 Hz run kept it at `(-0.1392, -0.5426)`. Before the obstacle moved, the
100 and 200 Hz reverse trajectories were nearly identical. Therefore the old
logs do not show that 200 Hz itself caused the failure.

## Fixed obstacle, 200 Hz, 0.45 m/s² task limit

Trace:

`/home/song/ros2_ws/log/escape_tuning/gain20_200hz/rmpflow_trace_20260720_110620.csv`

- Forward: reached in 16.85 s
- Reverse: timeout at 30.03 s
- Reverse final goal error: 0.7437 m
- Escape acceleration: `[-0.4446, +0.4493] m/s²`
- Final Escape metric: about 99,130
- Final state: RECOVERY, \(z\approx0.993,\lambda=1\)
- Commanded positive tangent distance: 0.611 m
- Actual positive tangent distance: 0.0281 m
- Move ratio: 0.046
- Active maximum joint-acceleration norm: 6.38 rad/s²
- Active global acceleration saturation: 0 rows
- Zero-effect handoff violations: 0

The Escape leaf was fully active but spent most of its authority cancelling
the base acceleration instead of producing useful tangent motion.

## Fixed obstacle, 200 Hz, no task limit

Trace:

`/home/song/ros2_ws/log/escape_tuning/gain20_200hz_unbounded_task/rmpflow_trace_20260720_111459.csv`

- Forward: reached in 19.10 s, error 0.0012 m
- Reverse: reached in 18.08 s, error 0.0025 m
- Escape task acceleration: `[-1.284, +3.016] m/s²`
- Active maximum joint-acceleration component: 9.44 rad/s²
- Active global acceleration saturation: 0 rows
- Zero-effect handoff violations: 0
- Maximum estimated joint-acceleration jerk: about 7,215 rad/s³

Removing the 0.45 m/s² saturation recovered this particular fixed-obstacle
round trip. It also increased task acceleration, velocity-tracking error, and
jerk. This is a performance result, not a Lyapunov proof.

## Fixed obstacle, 100 Hz, no task limit

Trace:

`/home/song/ros2_ws/log/escape_tuning/gain20_100hz_unbounded_task_fixed/rmpflow_trace_20260720_112037.csv`

- Forward: reached in 13.67 s
- Reverse: timeout at 30.02 s
- Reverse final error: 0.7302 m
- Active task acceleration: `[-0.481, +1.006] m/s²`
- Active acceleration saturation: 464 of 856 logged active rows

Thus the hybrid controller is not rate-invariant in performance even though
its explicit filters, ramps, and distance integrals use
\(dt=1/\text{control_rate}\). Small sampled-state differences can select a
different CP/tangent/handoff path, and the resulting branch can enter a
different acceleration-saturation regime. Tuning and validation must
therefore use the intended 200 Hz rate.

## 200 Hz conditional proof profile with no task limit

Trace:

`/home/song/ros2_ws/log/escape_tuning/gain20_proof_ideal_200hz_unbounded_task/rmpflow_trace_20260720_111718.csv`

Plot:

`/home/song/ros2_ws/log/escape_tuning/gain20_proof_ideal_200hz_unbounded_task/gain20_stability_proof_200hz_unbounded_task.png`

For complete certificate samples:

- structured-GDS base: 100%
- external RMP empty: 100%
- energy guard enabled: 100%
- conditional nonincrease: 99.98%
- the excluded rows are goal-change episode boundaries
- maximum sampled energy-bound violation: 0
- maximum tank-identity residual: \(6.54\times10^{-13}\)
- maximum absolute solve power: \(2.36\times10^{-10}\)
- clamp power: 0
- tank: \(1000\rightarrow6.95\times10^{-20}\)
- minimum applied Escape scale: 0
- raw joint-acceleration norm: up to 75.37 rad/s²

This validates the conditional solver-level energy budget for the special
proof profile. It does not prove the production sampled controller:

- the production base is not the structured-GDS proof profile;
- the production energy guard is disabled;
- downstream command guards and servo tracking are outside the certificate;
- no finite task-acceleration bound remains;
- the proof-profile route itself did not reach the forward goal.

## Current conclusion

The intended runtime is now 200 Hz, and the current no-task-limit production
profile completed the selected fixed-obstacle round trip once. The
zero-effect handoff rule was observed without violations.

The current production configuration still has no full Lyapunov/passivity
proof. The next structural robustness fix should make a persistently low move
ratio sufficient to trigger a zero-effect branch reselection; currently it
also requires increased sector risk or joint/actuator saturation.
