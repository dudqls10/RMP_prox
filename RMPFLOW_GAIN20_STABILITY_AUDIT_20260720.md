# Gain-20 Tangent Escape stability audit — 2026-07-20

## 결론

이번 결과는 두 가지를 분리한다.

1. Gain 20 production profile은 고정-cylinder 왕복 시뮬레이션에서 양방향
   목표 도달에 성공했다.
2. structured-GDS proof profile에서는 Escape energy-tank를 포함한
   continuous-time model의 조건부 Lyapunov 부등식과 sampled positive
   Escape-energy bound가 검증됐다.

그러나 production profile 자체는 non-GDS leaf, root regularization,
acceleration/command clamp와 downstream guard를 포함하므로 전체
Lyapunov 안정성이 증명된 상태는 아니다. Proof profile은 안정성 조건을
만족했지만 현재 왕복 목표 성능에는 실패했다. 따라서 “동일한 현재
controller가 성능과 전역 안정성을 동시에 증명했다”고 주장해서는 안
된다.

## 조건부 정리

Non-Escape base가 structured GDS이고

\[
V_0(q,\dot q)
=\frac12\dot q^\top G_0(q,\dot q)\dot q+\Phi_0(q)
\]

에 대해

\[
\dot V_0=-D_0+\dot q^\top(M_0\ddot q-f_0),\qquad D_0\ge0
\]

라고 하자. Rank-one Escape contribution은

\[
M_e=mj^\top j,\qquad f_e=mj^\top(a_y-c_y)
\]

이고

\[
v_t=j\dot q,\qquad \ddot y=j\ddot q+c_y
\]

이다. 합성 equation

\[
(M_0+M_e)\ddot q=f_0+f_e
\]

으로부터 base storage에 들어가는 정확한 Escape power는

\[
\boxed{
P_e=\dot q^\top(f_e-M_e\ddot q)
=m v_t(a_y-\ddot y)
}
\]

이다. 단순한 \(\dot q^\top f_e\)는 Escape metric contribution을 누락하므로
사용할 수 없다.

Tank energy \(E\in[0,E_{\max}]\)와

\[
\dot E=-[P_e]_+
\]

를 사용하고, tank가 비었을 때 Escape metric과 force를 같은 scale로
줄여 \(P_e\le0\)를 강제한다고 하자. 그러면

\[
W=V_0+E
\]

에 대해

\[
\begin{aligned}
\dot W
&=-D_0+P_e-[P_e]_+\\
&=-D_0+\min(P_e,0)\\
&\le0.
\end{aligned}
\]

따라서 고정 goal/obstacle, exact solve, clamp 없음, proper \(V_0\), SPD
\(G_0\)라는 조건에서 continuous-time state boundedness를 얻는다.

현재 sampled tank 구현은 더 좁은 결과

\[
E_{k+1}=E_k-h[P_{e,k}]_+\ge0
\]

와

\[
\boxed{
\sum_{k=0}^{N-1}h[P_{e,k}]_+\le E_0
}
\]

를 보장한다. 이것은 실제 semi-implicit state transition에 대한
\(W_{k+1}\le W_k\) 증명은 아니다.

## Gain 20 production run

Artifact:

- `/home/song/ros2_ws/log/escape_tuning/gain20/rmpflow_trace_20260720_095113.csv`

관측 결과:

- Forward 1 cm 도달: 13.922 s
- Reverse 1 cm 도달: 19.596 s
- 최종 TCP 위치오차: \(4.858\times10^{-7}\) m
- Escape-active 최소 clearance: 0.092354 m
- Escape scalar acceleration:
  \([-0.450000,\ 0.439639]\ {\rm m/s^2}\)
- Effective Escape metric: \([0,\ 100000]\)
- Active joint-acceleration saturation:
  \(116/1379=8.41\%\)
- Same-state Escape 추가 \(\|\Delta\ddot q\|\):
  mean 1.011, max 5.438 \({\rm rad/s^2}\)
- Same-state CP acceleration:
  mean tangent 0.4216, mean normal 0.0615 \({\rm m/s^2}\)

이 run의 certificate 전제는 불합격이다.

- `base_gds_structural=0`: 전 sample
- `base_config_profile_valid=0`: 전 sample
- `escape_stability_guard_enabled=0`: 전 sample
- `conditional_nonincrease=0`: 전 sample
- 누적 positive Escape model energy: 726.288
- 누적 negative Escape model energy: 4523.779
- 최대 positive acceleration-clamp power: \(1.95419\times10^5\)

따라서 이 run은 수렴성과 bounded-command의 실험 증거이지 Lyapunov
증명은 아니다.

## Gain 20 proof-profile run

Artifacts:

- CSV:
  `/home/song/ros2_ws/log/escape_tuning/gain20_proof_ideal/rmpflow_trace_20260720_095707.csv`
- Plot:
  `/home/song/ros2_ws/log/escape_tuning/gain20_proof_ideal/gain20_stability_proof.png`

Proof profile은 다음을 사용했다.

- Collision/Joint-limit: `paper_gds`
- Constant Target/Axis metrics
- Heuristic velocity-cap leaf disabled
- `root_solve_offset=0`
- Escape guard enabled
- Tank initial/capacity: 1000 generalized model-energy units
- Acceleration clamp를 사실상 제거한 continuous-model run

9,376개의 complete certificate sample 결과:

- `base_gds_structural`: 100%
- `external_rmp_empty`: 100%
- `guard_enabled`: 100%
- `environment_static`: 99.97%
- `conditional_nonincrease`: 99.97%
- \(\max|\)tank identity residual\(|=4.55\times10^{-13}\)
- 최대 energy-bound violation:
  \(1.14\times10^{-13}\)
- \(\max|P_{\rm solve}|=1.67\times10^{-10}\)
- \(P_{\rm clamp}=0\)
- Tank: \(1000\rightarrow2.62\times10^{-18}\)
- Applied positive Escape energy: 1000
- Applied negative Escape energy: 3606.336
- Minimum Escape scale: 0

`environment_static`와 `conditional_nonincrease`가 100%가 아닌 행은 goal
snapshot이 바뀐 episode boundary이다. Tank는 음수가 되지 않았고
양의 Escape energy가 정확히 초기 budget 이내로 제한됐다.

다만 이 profile에서는:

- 첫 초기점 오차가 0.0125 m에서 남음
- Forward goal 최종 오차가 0.9066 m
- Return goal 최종 오차가 0.0125 m
- Raw \(\|\ddot q\|\) 최대가 75.39 \({\rm rad/s^2}\)

였다. 즉 proof 조건은 검증했지만 production 성능과 물리 command
안전성을 검증한 run은 아니다.

## Zero-effect handoff

State machine은 한 controller solve 안에서 다음 순서를 사용한다.

1. old branch의 \(\lambda\)를 0으로 설정
2. CP/tangent 교체
3. new branch의 metric/force 계산

따라서 교체 cycle에는

\[
M_e=f_e=0
\]

이다. 기존 로그 검사기는 교체 직전 행까지 0이어야 한다고 요구해
false positive를 만들었다. 검사기를 교체 cycle의 new branch만
판정하도록 수정한 뒤:

- Production gain-20: 4 changes, 0 observed violations
- Proof profile: 5 changes, 0 observed violations

을 얻었다. Canonical integration test도 교체 cycle의 root delta metric과
delta force가 모두 0인지 직접 검사한다.

## 정확히 증명된 범위

- `paper_gds` Collision/Joint-limit의 leaf dissipation identity
- RMPflow proof profile의 base GDS structural 조건
- 정확한 Escape interconnection power
- Escape metric/force 동시 scaling의 PSD 보존
- Sampled positive Escape-energy upper bound
- Ideal continuous-time proof profile의 조건부 \(W=V_0+E\) nonincrease
- CP/tangent zero-effect handoff

## 아직 증명되지 않은 범위

- Production `lula_canonical` base의 공통 Lyapunov 함수
- 실제 sampled state의 \(W_{k+1}\le W_k\)
- Componentwise acceleration clamp와 이후 position/velocity/min-Z guard를
  모두 포함한 passivity
- 실제 servo tracking error와 delay를 포함한 hardware stability
- 목표로의 asymptotic convergence와 Zeno switching 부재
- Hard collision avoidance

Production 전체의 sampled Lyapunov 증명에는 모든 base storage를
runtime에서 평가하고, 최종 command transition의 \(\Delta V_0\)까지
제약하는 discrete passivity projection 또는 constrained QP가 추가로
필요하다.
