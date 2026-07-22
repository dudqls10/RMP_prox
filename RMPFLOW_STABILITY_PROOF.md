# RB10 RMPflow stability certificate

## 1. 무엇을 증명하고 무엇을 증명하지 않는가

이 문서는 세 결과를 분리한다.

1. `paper_gds` Collision leaf와 Joint-limit leaf는 각각 structured GDS이다.
2. 모든 non-Escape leaf와 task map이 아래 가정을 만족하면, RMPflow
   pullback과 합은 공통 Lyapunov 함수 \(V_0\)를 갖는다.
3. canonical Escape는 GDS라고 가정하지 않는다. 대신 base energy에
   주입하는 정확한 interconnection power를 유한 에너지 탱크로 제한한다.

이 결과는 **Lyapunov/에너지 안정성**에 관한 것이다. 유한한 smooth
Collision potential만으로

\[
d_{ih}(t)\ge 0
\]

의 forward invariance, 즉 절대 충돌하지 않는 hard safety는 증명되지
않는다. 실제 hard safety에는 모델 오차와 sampling margin을 포함한
acceleration-level CBF/QP가 별도로 필요하다.

## 2. Collision structured GDS

Signed clearance를 \(s\), 그 속도를 \(\dot s\)라 한다. 구현은

\[
G(s,\dot s)=w(s)u(\dot s)
\]

를 사용한다. 거리 함수 \(w\)는 nonnegative이고 \(w'(s)\le0\)이며,
velocity profile은

\[
\dot s\,u'(\dot s)\ge0
\]

을 만족한다. 따라서 1차원 GDS curvature terms는

\[
\Xi_G=\frac12w\dot s\,u',\qquad
\xi_G=\frac12u w'\dot s^2
\]

이고 resolved inertia는

\[
M=G+\Xi_G
=w\left(u+\frac12\dot s\,u'\right)\ge0
\]

이다.

Repulsion은 position-only potential로 정의한다.

\[
F_\Phi(s)=-\Phi'(s)\ge0
\]

Damping coefficient는 \(B(s,\dot s)\ge0\)이고 natural force는

\[
f_s=F_\Phi-\xi_G-B\dot s
\]

이다. Leaf equation \(M\ddot s=f_s\)를 대입하면

\[
\begin{aligned}
V_s
&=\frac12G(s,\dot s)\dot s^2+\Phi(s),\\
\dot V_s
&=\dot s\,[M\ddot s+\xi_G+\Phi'(s)]\\
&=-B\dot s^2\le0.
\end{aligned}
\]

Root pullback에서는 task-map curvature
\(c_s=\dot J_s\dot q\)와 GDS intrinsic curvature \(\xi_G\)를 구분한다.

\[
M_q=J_s^\top M J_s,\qquad
f_q=J_s^\top(f_s-Mc_s).
\]

구현:

- `include/rb10_rmpflow_rviz/paper_gds_collision.hpp`
- `src/paper_gds_collision.cpp`
- `test/test_paper_gds_collision.cpp`

`lula_canonical`은 기존 사이트/production 식을 그대로 보존한다.
`paper_gds`만 위 증명 대상이다.

## 3. Joint-limit structured GDS

각 joint의 buffered open interval을

\[
q\in(l,u)
\]

라 하고 logit task를 사용한다.

\[
z(q)=\log\frac{q-l}{u-q}.
\]

그 Jacobian과 curvature는

\[
J_z=\frac1{q-l}+\frac1{u-q},
\]

\[
c_z=\dot J_z\dot q
=\left[-\frac1{(q-l)^2}+\frac1{(u-q)^2}\right]\dot q^2.
\]

Task space에서

\[
G_z=M_z=m_z>0,
\]

\[
\Phi_z=\frac12k_z(z-z_0)^2,\qquad B_z=d_z>0
\]

로 둔다. Natural force는

\[
f_z=-k_z(z-z_0)-d_z\dot z.
\]

따라서

\[
V_z=\frac12m_z\dot z^2+\Phi_z,\qquad
\dot V_z=-d_z\dot z^2\le0.
\]

\(q\to l^+\) 또는 \(q\to u^-\)이면
\(|z|\to\infty\)이고 \(\Phi_z\to\infty\)이다. 이는 exact continuous-time
model과 open-domain initial state라는 가정 아래 energy level set이
joint boundary에 도달하지 못하게 한다. 실제 sampled controller의
clamp, delay, actuator saturation까지 포함한 hard joint-limit 보장은
별도 command safety analysis가 필요하다.

구현:

- `include/rb10_rmpflow_rviz/paper_gds_joint_limit.hpp`
- `src/paper_gds_joint_limit.cpp`
- `test/test_paper_gds_joint_limit.cpp`

## 4. Base RMPflow composition

각 non-Escape leaf \(\ell\)이 structured GDS이고 task map이 smooth하고
time invariant라고 하자. Natural pullback과 sum으로

\[
M_0=\sum_\ell J_\ell^\top M_\ell J_\ell,
\]

\[
f_0=\sum_\ell J_\ell^\top(f_\ell-M_\ell c_\ell)
\]

를 만든다. RMPflow closure theorem에 의해 base system은

\[
M_0\ddot q=f_0
\]

이고, 공통 storage

\[
V_0(q,\dot q)
=\frac12\dot q^\top G_0(q,\dot q)\dot q+\Phi_0(q)
\]

에 대해

\[
\dot V_0=-D_0,\qquad
D_0=\dot q^\top B_0\dot q\ge0
\]

이다.

이 repository에서 runtime flag `base_gds_structural`이 1이 되려면:

- Collision과 Joint-limit가 모두 `paper_gds`
- Target metric이 constant가 되도록
  `target_rmp_min_metric_alpha=1`,
  `target_rmp_proximity_metric_boost_scalar=1`
- Axis-target boost가 1
- Damping metric이 constant가 되도록
  `damping_rmp_metric_scalar=0`
- heuristic joint velocity cap leaf 비활성
- external canonical leaf 비활성
- `root_solve_offset=0`
- graph의 task map이 현재 certificate allow-list의 smooth map
- 해당 sample의 \(M_0\)가 finite, symmetric positive definite

이어야 한다. 전용
`rb10_rmpflow_gds_stability_validation.launch.py`가 이 profile을 만든다.
이 flag는 구성과 수치조건을 검사하지만, 구현된 모든 Jacobian/curvature가
수학적 미분과 일치한다는 사실 자체는 unit test와 모델 검증의 가정이다.

Launch의 기본 `max_joint_accel=1000000`은 clamp가 없는 이상적 GDS
모델률을 분리해 검증하기 위한 simulation 설정이다. 실제와 가까운
bounded-command 검증은 `max_joint_accel:=10.0`처럼 별도로 실행하고,
그때 `clamp_active`, \(P_{\rm clamp}\), `conditional_nonincrease`를 함께
판정해야 한다. 큰 기본값 자체는 물리 안전 설정이 아니다.

## 5. Escape가 base energy에 주입하는 정확한 power

Escape scalar leaf는

\[
M_e=mj^\top j,\qquad
f_e=mj^\top(a_y-c_y)
\]

이고

\[
v_t=j\dot q,\qquad
\ddot y=j\ddot q+c_y.
\]

합성 equation은

\[
(M_0+M_e)\ddot q=f_0+f_e.
\]

Structured-GDS identity를 base storage에 적용하면

\[
\dot V_0
=-D_0+\dot q^\top(M_0\ddot q-f_0).
\]

합성 equation으로부터

\[
M_0\ddot q-f_0=f_e-M_e\ddot q
\]

이므로 Escape의 정확한 interconnection power는

\[
\boxed{
P_e
=\dot q^\top(f_e-M_e\ddot q)
=m v_t(a_y-\ddot y)
}.
\]

따라서 \(\dot q^\top f_e\)만 측정하는 것은 metric contribution을
누락하므로 올바른 증명이 아니다.

Regularization과 command acceleration clamp를 포함하면

\[
r_{\rm solve}=M\ddot q_{\rm raw}-f,
\]

\[
P_{\rm solve}=\dot q^\top r_{\rm solve},
\]

\[
P_{\rm clamp}
=\dot q^\top M(\ddot q_{\rm cmd}-\ddot q_{\rm raw})
\]

이고

\[
\dot V_0
=-D_0+P_e+P_{\rm solve}+P_{\rm clamp}+P_{\rm time}.
\]

## 6. 유한 에너지 탱크 정리

### 6.1 이상적인 연속시간 폐루프

Tank energy를 \(E\in[0,E_{\max}]\)라 한다. 먼저 base damping을
credit으로 사용하지 않는 이상적인 연속시간 guard를 정의한다.

\[
\dot E=-[P_e]_+,\qquad E(0)=E_{\rm init}.
\]

이상적인 연속시간 guard가 Escape metric과 force를 같은
scale \(\alpha\in[0,1]\)로 줄여 \(E=0\)에서 \(P_e\le0\)를 강제한다고
하자. 그러면

\[
\dot E=-[P_e]_+
\]

이고 모든 \(T\)에 대해

\[
\boxed{
\int_0^T[P_e(t)]_+dt
\le E_{\rm init}
}
\]

이다. Composite storage를 \(W=V_0+E\)라 하면

\[
\dot W
\le-D_0+P_{\rm solve}+P_{\rm clamp}+P_{\rm time}.
\]

따라서 time-invariant base GDS, 정확한 solve, 실제 acceleration이
command acceleration과 같고 \(P_{\rm time}=0\)이라는 이상적 조건에서는
\(\dot W\le0\)이다.

### 6.2 현재 디지털 구현이 실제로 보장하는 것

`escape_stability_guard_enabled=true`일 때 현재 controller는 sample
\(k\)에서 metric과 force를 같은 scale
\(\alpha_k\in[0,1]\)로 줄이고 combined solve와 acceleration clamp를
다시 계산하여

\[
h[P_{e,k}]_+\le E_k
\]

를 만족하는 **feasible scale**을 선택한다. Acceleration clamp가 있으면
\(P_e(\alpha)\)의 전역 단조성이 일반적으로 보장되지 않으므로, finite
bisection의 결과를 수학적으로 “가장 큰 scale”이라고 주장하지 않는다.
마지막 feasibility 검사와 \(\alpha=0\) fallback이 위 부등식을 보장한다.

여기서 \(h\)는 wall-clock으로 측정한 실제 loop jitter가 아니라
`control_rate`에서 설정한 고정 `control_dt`이다. Tank update는

\[
\boxed{
E_{k+1}=E_k-h[P_{e,k}]_+\ge0
}
\]

이므로 정확히 증명되는 runtime 누적 bound는

\[
\boxed{
\sum_{k=0}^{N-1}h[P_{e,k}]_+
\le E_{\rm init}
}
\]

이다. `energy_bound_violation=0`과 `tank_identity_residual`이 이 식을
감사한다.

각 sample에서 다음 조건이 동시에 성립하면

1. `base_gds_structural=1`
2. goal과 obstacle snapshot이 해당 step에서 time invariant
3. external RMP가 없음
4. Escape energy guard가 켜졌거나, guard가 꺼졌다면 uncovered
   \([P_e]_+\)를 upper bound에 포함
5. \([P_e]_+\mathbf 1_{\rm guard\ off}
   +P_{\rm solve}+P_{\rm clamp}\le\varepsilon_P\)

수치 tolerance \(\varepsilon_P\)까지 허용한 측정 연속시간 모델의 순간
upper bound

\[
-D_0+P_{\rm solve}+P_{\rm clamp}+P_{\rm time}
\le -D_0+\varepsilon_P
\]

를 얻는다. 따라서 strict한 0 이하가 아니라 tolerance-relaxed
certificate이다. 이 판정이 runtime field
`conditional_nonincrease=1`이다.

중요하게도 이것은 semi-implicit Euler로 적분된 실제
\((q_{k+1},\dot q_{k+1})\)에 대해

\[
W_{k+1}\le W_k
\]

를 직접 계산해 증명한 discrete-time certificate는 아니다. 그 보장까지
필요하면 모든 base leaf의 \(V_0(q,\dot q)\)를 runtime에 평가하고,
예측된 다음 상태에서의 \(\Delta V_0\)를 포함하는 discrete passivity
controller 또는 energy-preserving integrator가 추가로 필요하다.

또한 solver certificate가 만들어진 뒤 controller가 적용하는 joint
position clamp, joint-limit 인접 속도 감속/외향 속도 제거, min-Z hold와
command step/velocity guard는 위 \(P_{\rm clamp}\)에 포함되지 않는다.
전용 launch는 configurable limit를 매우 크게 완화하지만, 무조건 남는
domain guard까지 제거하지는 않는다. 따라서 로그가 이 guard들을 밟지
않았다는 별도 확인 없이 certificate를 실제 command transition의
discrete-time 증명으로 해석하면 안 된다.

Goal 또는 obstacle snapshot이 바뀌면 `environment_static=0`이 된다.
특히 goal 변경은 \(V_0\)의 potential 자체를 바꿀 수 있고 그 storage
jump를 현재 tank에서 차감하지 않는다. 따라서 연속 구간 전체가 아니라
고정 goal·고정 obstacle episode별 조건부 정리이다.

이것은 convergence까지 자동으로 뜻하지 않는다. Asymptotic convergence는
LaSalle 조건, damping의 detectability, 목표 근처 Escape release,
그리고 Zeno switching 부재가 추가로 필요하다. 방향/CP handoff는
\(m=0\)에서 수행되므로 \(P_e=0\)이고 \(q,\dot q\)가 연속이면 \(W\)에
jump가 없다.

## 7. Runtime certificate schema v1

Topic: `/rmp_stability_certificate`

| index | field |
|---:|---|
| 0 | schema version |
| 1 | base GDS structural flag |
| 2 | exogenous goal/obstacle unchanged from previous sample |
| 3 | external RMP set empty |
| 4 | Escape energy guard enabled |
| 5 | conditional nonincrease certificate |
| 6 | applied Escape metric/force scale |
| 7–8 | tank energy / capacity |
| 9–10 | requested / applied Escape power |
| 11 | solve residual power |
| 12 | acceleration-clamp power |
| 13 | solve + clamp power |
| 14–16 | cumulative positive / negative / net Escape energy |
| 17 | sample count |
| 18–21 | requested/applied metric trace and force norm |
| 22–24 | raw/command acceleration norm and clamp flag |
| 25 | root solve offset |
| 26 | nonzero Escape contribution flag |
| 27 | tank identity residual |
| 28 | finite positive-energy bound violation |
| 29 | nonincrease upper-bound term excluding \(-D_0\) |
| 30 | initial tank energy |
| 31 | base configuration profile valid |
| 32 | current base task/domain data valid |
| 33 | current base root metric symmetric positive definite |
| 34 | current base root solve residual valid |

RViz의 `StabilityCertificate` marker와 trace CSV가 같은 값을 표시한다.

## 8. 검증 범위

Automated tests는 다음 algebraic identity를 검사한다.

- Collision \(w'\), \(\Xi_G\), \(\xi_G\) finite difference
- Collision \(M\ge0\), \(B\ge0\), \(\dot s u'\ge0\)
- Collision \(\dot V=-B\dot s^2\)
- Joint logit \(J,\dot J\dot q\) finite difference
- Joint-limit \(\dot V=-d_z\dot z^2\)
- Direct Escape power와 rank-one closed form의 일치
- Rank-one metric cap
- Tank nonnegativity와 energy balance sign

실험에서 `conditional_nonincrease=0`인 구간은 “불안정이 증명됨”이 아니라
해당 순간에 위 정리의 가정 중 하나가 만족되지 않았다는 뜻이다.
반대로 1은 simulation command-state model, 측정된 snapshot, 그리고 수치
residual 범위 안에서의 조건부 certificate이다. Hardware servo의 tracking
error와 delay, 미관측 장애물과 actuator/model error까지 포함한 hard safety
인증도, sampled state에 대한 \(W_{k+1}\le W_k\) 인증도 아니다.
