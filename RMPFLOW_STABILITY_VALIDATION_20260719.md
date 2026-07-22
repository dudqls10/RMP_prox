# RB10 RMPflow stability validation — 2026-07-19

## 결론

이번 검증은 다음 세 결과를 분리한다.

1. `paper_gds` Collision과 Joint-limit leaf의 structured-GDS 항등식은
   단위 테스트로 검증됐다.
2. 고정 goal/obstacle, exact root solve, proof-profile base leaf라는 가정
   아래 continuous-time base RMPflow와 ideal Escape energy tank의
   조건부 Lyapunov 정리가 성립한다.
3. 현재 디지털 구현은 더 좁은 정리,

   \[
   E_{k+1}=E_k-h[P_{e,k}]_+,\qquad
   \sum_k h[P_{e,k}]_+\le E_0,
   \]

   즉 sampled Escape 양의 model-energy 예산을 보장한다.

실제 semi-implicit state에 대한 \(W_{k+1}\le W_k\), 실제 하드웨어
plant 안정성, hard collision safety는 이번 결과로 증명되지 않았다.

## 자동 검증

`colcon test-result --verbose` 결과:

- 20 tests
- 0 errors
- 0 failures
- 0 skipped

검증 항목에는 Collision/Joint-limit GDS 미분과 dissipation identity,
Escape rank-one power identity, tank balance, canonical rank-one leaf,
same-state dual solve, zero-effect handoff, bounded `RESELECT_BRAKE`, exact
GDS profile과 structural sub-check가 포함된다.

## Run A — unclamped continuous-model proof profile

Artifacts:

- CSV:
  `/home/song/ros2_ws/log/rmpflow_trace/rmpflow_trace_20260719_155357.csv`
- Plot:
  `/home/song/ros2_ws/log/rmpflow_trace/rmpflow_trace_20260719_155357_proof.png`

설정은 `max_joint_accel=1000000`으로 acceleration clamp 영향을 사실상
제거했다. 이것은 물리 안전 설정이 아니라 continuous-model 식을
분리하는 proof 설정이다.

Certificate/energy 결과:

- complete certificate/dual samples: 126/126
- base/config/domain/SPD/solve flags: 모두 126/126
- external RMP empty, guard enabled: 모두 126/126
- environment static 및 conditional certificate: 115/126 (91.27%)
- 나머지 11행은 goal/obstacle snapshot 변경 때문에 조건 적용 불가
- energy-bound violation: 0
- tank identity residual 최대 절댓값: \(2.19\times10^{-14}\)
- nonincrease upper-bound 최대: \(8.37\times10^{-10}\)
  (설정 tolerance \(10^{-9}\) 이하)
- final cumulative positive/negative/net Escape energy:
  \(0.37147,\ 7.65994,\ -7.28847\)
- final tank: \(99.62853/100\)
- Escape guard scale: 항상 1

Same-state Escape ON/OFF 결과:

- Escape-active snapshots: 21
- \(\|\Delta\ddot q\|\): mean \(6.95\times10^{-4}\),
  max \(1.78\times10^{-3}\ \mathrm{rad/s^2}\)
- \(\|\Delta a_{\rm TCP}\|\): mean \(2.86\times10^{-4}\),
  max \(7.34\times10^{-4}\ \mathrm{m/s^2}\)
- \(\|\Delta a_{\rm CP}\|\): mean \(2.44\times10^{-4}\),
  max \(6.28\times10^{-4}\ \mathrm{m/s^2}\)

Tangent/handoff:

- Engage/Drive \(\max|n^\top t|=0.0447\)
- bounded RESELECT \(\max|n^\top t|=0.0545\)
- RELEASE braking \(\max|n^\top t|=0.3203\)
- 이전 구현의 RESELECT 최대 0.6757 및 수 초 체류는 bounded handoff로
  제거됐다.

관측된 triggered-sensor sphere clearance 최소는 \(9.758\ \mathrm{mm}\)로
음수 overlap은 없었다. 이는 기록된 model snapshots의 관측 결과이지
continuous collision avoidance 증명이 아니다.

이 run의 raw \(\|\ddot q\|\) 최대는 \(111.93\ \mathrm{rad/s^2}\)였다.
따라서 Run A를 production actuator safety 근거로 사용할 수 없다.

## Run B — per-joint acceleration limit 10 rad/s²

Artifacts:

- CSV:
  `/home/song/ros2_ws/log/rmpflow_trace/rmpflow_trace_20260719_155732.csv`
- Plot:
  `/home/song/ros2_ws/log/rmpflow_trace/rmpflow_trace_20260719_155732_bounded_proof.png`

결과:

- complete certificate/dual samples: 306/306
- base/config/domain/SPD/solve flags: 모두 306/306
- environment static: 221/306 (72.22%)
- conditional certificate: 218/306 (71.24%)
- componentwise acceleration clamp active: 14/306
- \(P_{\rm clamp}\) 범위:
  \([-2.739\times10^6,\ 4.420\times10^6]\)
- command \(\|\ddot q\|\) 최대: \(21.746\ \mathrm{rad/s^2}\);
  각 joint component는 \(10\ \mathrm{rad/s^2}\) 이내
- Escape energy-bound violation: 0
- triggered-sensor sphere clearance 최소: \(9.878\ \mathrm{mm}\)

즉 Escape tank 예산은 지켜졌지만, 단순 componentwise acceleration
clamp가 양의 model power를 만들 수 있어 세 개의 static-environment
snapshot에서 model-rate certificate까지 깨졌다. 전역 acceleration
limit를 추가하는 것만으로 논문형 안정성이 자동 보존되지는 않는다.

## 정확한 증명 범위

증명/검증된 것:

- `paper_gds` Collision과 Joint-limit leaf의 structured-GDS identity
- proof-profile base graph의 runtime config/domain/SPD/solve 조건
- Escape의 정확한 interconnection power
  \(P_e=\dot q^\top(f_e-M_e\ddot q)\)
- enabled tank의 sampled positive Escape energy bound
- 방향/CP 교체가 \(\lambda=0\), 즉 \(M_e=f_e=0\)에서 일어나는 구조
- 동일 state, 동일 \(M_0,f_0\)에서 Escape ON/OFF 순간 차이

증명되지 않은 것:

- 실제 sampled transition의 \(W_{k+1}\le W_k\)
- moving-goal 전체 16-pose route의 단일 공통 storage
- command pipeline의 모든 clamp/hold를 포함한 전역 안정성
- 실제 servo tracking error와 delay를 포함한 hardware stability
- 미관측 obstacle과 inter-sample motion을 포함한 hard collision safety
- 목표 수렴과 Zeno switching 부재

이번 1.5초-per-pose run에서는 16개 pose 모두 5 mm tolerance 도달 전에
다음 pose로 넘어갔다. 따라서 경로 실행과 Escape 활성은 확인했지만
route convergence 성공으로 판정하지 않는다.

## hard safety와 sampled stability에 추가로 필요한 것

1. 모든 base leaf storage \(V_0\)와 실제 command transition을 평가해
   \(\Delta V_0\)를 제한하는 discrete passivity controller 또는
   energy-preserving/discrete-gradient integrator.
2. acceleration/velocity/position constraint를 단순 clamp가 아니라
   passivity 조건과 함께 푸는 constrained solve. 불가능한 state에서는
   명시적 safe fallback이 필요하다.
3. signed clearance의 forward invariance를 직접 강제하는
   braking-distance-aware CBF/QP safety layer.
4. sensor/model/servo error와 delay bound, 더 높은 주기의 rosbag 또는
   controller-cycle logging, 실제 hardware tracking 검증.

