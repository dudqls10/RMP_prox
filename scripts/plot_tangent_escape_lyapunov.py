#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LOG_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()
REQUIRED_COLUMNS = [
    "time_ros_s",
    "tangent_escape_rmp_active",
    "tangent_escape_rmp_leaf_mode_id",
    "tangent_escape_rmp_control_point_index",
    "tangent_escape_rmp_clearance_m",
    "tangent_escape_rmp_activation",
    "tangent_escape_rmp_effective_metric_scalar",
    "tangent_escape_rmp_scalar_s_m",
    "tangent_escape_rmp_scalar_target_m",
    "tangent_escape_rmp_scalar_velocity_m_s",
    "tangent_escape_rmp_scalar_error_m",
    "tangent_escape_rmp_desired_tangent_accel_m_s2",
    "rmp_joint_accel_norm",
]

SENSOR_NAMES = [
    "tof6_1_L",
    "tof6_1_F",
    "tof6_1_R",
    "tof6_1_U",
    "tof_S",
    "tof_E",
    "tof_N",
    "tof_W",
    "tof3_1_S",
    "tof3_1_W",
    "tof3_1_N",
    "tof3_1_E",
    "tof2_1_E",
    "tof2_1_S",
    "tof2_1_W",
    "tof2_1_N",
    "tof2_E",
    "tof2_S",
    "tof2_W",
    "tof2_N",
]


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def latest_compatible_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = set(reader.fieldnames or [])
        except OSError:
            continue
        if all(column in fieldnames for column in REQUIRED_COLUMNS):
            return path
    raise FileNotFoundError(
        f"No compatible rmpflow_trace CSV with Stage-2 tangent escape RMP columns found in {log_dir}"
    )


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError("Missing Lyapunov plot columns: " + ", ".join(missing))
    return rows, fieldnames


def sensor_name(index: int) -> str:
    if 0 <= index < len(SENSOR_NAMES):
        return SENSOR_NAMES[index]
    return "unknown"


def active_intervals(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, row in enumerate(rows):
        active = parse_float(row, "tangent_escape_rmp_active") >= 0.5
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(rows) - 1))
    return intervals


def qdd_norm(row: Dict[str, str]) -> float:
    value = parse_float(row, "rmp_joint_accel_norm")
    if math.isfinite(value):
        return value
    values = [parse_float(row, f"rmp_joint_accel_{index}_rad_s2") for index in range(1, 7)]
    if all(math.isfinite(value) for value in values):
        return math.sqrt(sum(value * value for value in values))
    return float("nan")


def goal_error(row: Dict[str, str]) -> float:
    value = parse_float(row, "goal_error_m")
    if math.isfinite(value):
        return value
    return float("nan")


def parse_candidate_count(row: Dict[str, str]) -> float:
    value = parse_float(row, "tangent_escape_rmp_candidate_count")
    if math.isfinite(value):
        return value
    payload = row.get("tangent_escape_rmp_candidates_json", "")
    if not payload:
        return float("nan")
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return float("nan")
    if isinstance(decoded, dict):
        count = decoded.get("count")
        if isinstance(count, (int, float)):
            return float(count)
        candidates = decoded.get("candidates")
        if isinstance(candidates, list):
            return float(len(candidates))
    return float("nan")


def vector_dot(row: Dict[str, str], lhs_prefix: str, rhs_prefix: str) -> float:
    lhs = [parse_float(row, f"{lhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    rhs = [parse_float(row, f"{rhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in lhs + rhs):
        return float("nan")
    return sum(lhs[index] * rhs[index] for index in range(3))


def make_samples(
    rows: Sequence[Dict[str, str]],
    position_gain: float,
    damping_gain: float,
    max_accel: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Tuple[int, int]]]:
    if not rows:
        return [], [], []
    first_time = parse_float(rows[0], "time_ros_s")
    all_samples: List[Dict[str, float]] = []
    active_samples: List[Dict[str, float]] = []

    for index, row in enumerate(rows):
        stamp = parse_float(row, "time_ros_s")
        time_rel = stamp - first_time if math.isfinite(stamp) and math.isfinite(first_time) else float("nan")
        qdd = qdd_norm(row)
        goal = goal_error(row)
        all_samples.append({
            "row": index,
            "time": time_rel,
            "qddNorm": qdd,
            "goalError": goal,
            "active": parse_float(row, "tangent_escape_rmp_active"),
            "supervisorMode": parse_float(row, "tangent_escape_rmp_supervisor_mode_id"),
            "holdActive": parse_float(row, "tangent_escape_rmp_hold_active"),
            "holdBonus": parse_float(row, "tangent_escape_rmp_selected_hold_bonus"),
            "stuckActive": parse_float(row, "tangent_escape_rmp_stuck_active"),
            "stuckTimer": parse_float(row, "tangent_escape_rmp_stuck_timer_s"),
            "metricBoost": parse_float(row, "tangent_escape_rmp_metric_boost"),
            "accelBoost": parse_float(row, "tangent_escape_rmp_accel_boost"),
            "blockedPenalty": parse_float(
                row,
                "tangent_escape_rmp_selected_blocked_penalty",
            ),
            "maxBlockedMemory": parse_float(row, "tangent_escape_rmp_max_blocked_memory"),
            "branchAge": parse_float(row, "tangent_escape_rmp_branch_age_s"),
            "branchProgress": parse_float(row, "tangent_escape_rmp_branch_progress_m"),
            "clearanceImprovement": parse_float(
                row,
                "tangent_escape_rmp_clearance_improvement_m",
            ),
        })

        active_value = parse_float(row, "tangent_escape_rmp_active")
        if not math.isfinite(active_value) or active_value < 0.5:
            continue
        effective_metric = parse_float(row, "tangent_escape_rmp_effective_metric_scalar")
        scalar_s = parse_float(row, "tangent_escape_rmp_scalar_s_m")
        target = parse_float(row, "tangent_escape_rmp_scalar_target_m")
        velocity = parse_float(row, "tangent_escape_rmp_scalar_velocity_m_s")
        error = parse_float(row, "tangent_escape_rmp_scalar_error_m")
        desired_accel = parse_float(row, "tangent_escape_rmp_desired_tangent_accel_m_s2")
        mode_id = parse_float(row, "tangent_escape_rmp_leaf_mode_id")
        accel_boost = parse_float(row, "tangent_escape_rmp_accel_boost")
        if not math.isfinite(accel_boost) or accel_boost <= 0.0:
            accel_boost = 1.0
        unclamped_accel = position_gain * error - damping_gain * velocity
        if math.isfinite(mode_id) and int(round(mode_id)) == 4:
            effective_position_gain = position_gain * accel_boost
            effective_damping_gain = damping_gain * accel_boost
            effective_max_accel = max_accel * accel_boost
            spring = effective_position_gain * error
            if effective_max_accel > 0.0:
                spring = effective_max_accel * math.tanh(
                    effective_position_gain * error / effective_max_accel
                )
            unclamped_accel = spring - effective_damping_gain * velocity
        kinetic = 0.5 * effective_metric * velocity * velocity
        potential = 0.5 * effective_metric * position_gain * error * error
        lyapunov = kinetic + potential
        damping_vdot = -effective_metric * damping_gain * velocity * velocity
        logged_kinetic = parse_float(row, "tangent_escape_rmp_escape_kinetic")
        logged_potential = parse_float(row, "tangent_escape_rmp_escape_potential")
        logged_lyapunov = parse_float(row, "tangent_escape_rmp_escape_lyapunov")
        logged_damping_vdot = parse_float(row, "tangent_escape_rmp_escape_damping_vdot")
        if math.isfinite(logged_kinetic):
            kinetic = logged_kinetic
        if math.isfinite(logged_potential):
            potential = logged_potential
        if math.isfinite(logged_lyapunov):
            lyapunov = logged_lyapunov
        if math.isfinite(logged_damping_vdot):
            damping_vdot = logged_damping_vdot
        clamp_residual = desired_accel - unclamped_accel
        cp_index = parse_float(row, "tangent_escape_rmp_control_point_index")
        active_samples.append({
            "row": index,
            "time": time_rel,
            "cpIndex": cp_index,
            "modeId": mode_id,
            "modeGeneration": parse_float(row, "tangent_escape_rmp_mode_generation"),
            "weightsLatched": parse_float(row, "tangent_escape_rmp_weights_latched"),
            "boundedPotential": parse_float(row, "tangent_escape_rmp_bounded_potential"),
            "activation": parse_float(row, "tangent_escape_rmp_activation"),
            "clearance": parse_float(row, "tangent_escape_rmp_clearance_m"),
            "effectiveMetric": effective_metric,
            "scalarS": scalar_s,
            "target": target,
            "error": error,
            "velocity": velocity,
            "desiredAccel": desired_accel,
            "unclampedAccel": unclamped_accel,
            "clampResidual": clamp_residual,
            "kinetic": kinetic,
            "potential": potential,
            "lyapunov": lyapunov,
            "dampingVdot": damping_vdot,
            "qddNorm": qdd,
            "goalError": goal,
            "candidateCount": parse_candidate_count(row),
            "selectedCandidateIndex": parse_float(row, "tangent_escape_rmp_selected_candidate_index"),
            "selectedWeight": parse_float(row, "tangent_escape_rmp_selected_candidate_weight"),
            "selectedScore": parse_float(row, "tangent_escape_rmp_selected_candidate_score"),
            "selectedGoalScore": parse_float(row, "tangent_escape_rmp_selected_goal_score"),
            "selectedContinuityScore": parse_float(
                row,
                "tangent_escape_rmp_selected_continuity_score",
            ),
            "selectedDuplicateRisk": parse_float(
                row,
                "tangent_escape_rmp_selected_duplicate_risk",
            ),
            "selectedAdjacentRisk": parse_float(row, "tangent_escape_rmp_selected_adjacent_risk"),
            "softmaxBeta": parse_float(row, "tangent_escape_rmp_softmax_beta"),
            "branchWeightSum": parse_float(row, "tangent_escape_rmp_branch_weight_sum"),
            "weightEntropy": parse_float(row, "tangent_escape_rmp_weight_entropy"),
            "normalDotTangent": (
                parse_float(row, "tangent_escape_rmp_normal_dot_tangent")
                if math.isfinite(parse_float(row, "tangent_escape_rmp_normal_dot_tangent"))
                else vector_dot(row, "tangent_escape_rmp_normal", "tangent_escape_rmp_tangent")
            ),
            "desiredAccelDotNormal": parse_float(
                row,
                "tangent_escape_rmp_desired_accel_dot_normal_m_s2",
            ),
            "desiredAccelDotTangent": parse_float(
                row,
                "tangent_escape_rmp_desired_accel_dot_tangent_m_s2",
            ),
            "selectedScoreGap": parse_float(row, "tangent_escape_rmp_selected_score_gap"),
            "selectedWeightGap": parse_float(row, "tangent_escape_rmp_selected_weight_gap"),
            "maxCandidateAdjacentRisk": parse_float(
                row,
                "tangent_escape_rmp_max_candidate_adjacent_risk",
            ),
            "maxCandidateDuplicateRisk": parse_float(
                row,
                "tangent_escape_rmp_max_candidate_duplicate_risk",
            ),
            "maxCandidateNormalDotTangent": parse_float(
                row,
                "tangent_escape_rmp_max_abs_candidate_normal_dot_tangent",
            ),
            "supervisorMode": parse_float(row, "tangent_escape_rmp_supervisor_mode_id"),
            "holdActive": parse_float(row, "tangent_escape_rmp_hold_active"),
            "holdBonus": parse_float(row, "tangent_escape_rmp_selected_hold_bonus"),
            "stuckActive": parse_float(row, "tangent_escape_rmp_stuck_active"),
            "stuckTimer": parse_float(row, "tangent_escape_rmp_stuck_timer_s"),
            "metricBoost": parse_float(row, "tangent_escape_rmp_metric_boost"),
            "accelBoost": accel_boost,
            "blockedPenalty": parse_float(row, "tangent_escape_rmp_selected_blocked_penalty"),
            "maxBlockedMemory": parse_float(row, "tangent_escape_rmp_max_blocked_memory"),
            "branchAge": parse_float(row, "tangent_escape_rmp_branch_age_s"),
            "branchProgress": parse_float(row, "tangent_escape_rmp_branch_progress_m"),
            "clearanceImprovement": parse_float(
                row,
                "tangent_escape_rmp_clearance_improvement_m",
            ),
        })

    previous: Optional[Dict[str, float]] = None
    for sample in active_samples:
        vdot = float("nan")
        residual = float("nan")
        if previous is not None:
            dt = sample["time"] - previous["time"]
            same_fixed_mode = (
                sample["cpIndex"] == previous["cpIndex"] and
                (
                    not math.isfinite(sample["modeGeneration"]) or
                    not math.isfinite(previous["modeGeneration"]) or
                    round(sample["modeGeneration"]) == round(previous["modeGeneration"])
                )
            )
            if math.isfinite(dt) and dt > 1e-9 and same_fixed_mode:
                vdot = (sample["lyapunov"] - previous["lyapunov"]) / dt
                residual = vdot - sample["dampingVdot"]
        sample["finiteDiffVdot"] = vdot
        sample["vdotResidual"] = residual
        previous = sample

    return all_samples, active_samples, active_intervals(rows)


def finite_range(samples: Sequence[Dict[str, float]], key: str) -> str:
    values = finite_values(sample.get(key, float("nan")) for sample in samples)
    if not values:
        return "n/a"
    return f"{fmt(min(values))} to {fmt(max(values))}"


def max_abs(samples: Sequence[Dict[str, float]], key: str) -> float:
    values = finite_values(abs(sample.get(key, float("nan"))) for sample in samples)
    return max(values) if values else float("nan")


def write_report(
    csv_path: Path,
    output_path: Path,
    all_samples: Sequence[Dict[str, float]],
    active_samples: Sequence[Dict[str, float]],
    intervals: Sequence[Tuple[int, int]],
    position_gain: float,
    damping_gain: float,
) -> None:
    if not active_samples:
        raise ValueError("No tangent_escape_rmp_active rows found; cannot build Lyapunov plot")

    cp_indices = sorted({
        int(round(sample["cpIndex"]))
        for sample in active_samples
        if math.isfinite(sample["cpIndex"])
    })
    cp_text = ", ".join(f"{index} ({sensor_name(index)})" for index in cp_indices)
    mode_ids = sorted({
        int(round(sample["modeId"]))
        for sample in active_samples
        if math.isfinite(sample["modeId"])
    })
    mode_text = ", ".join(str(mode) for mode in mode_ids)
    mode_generations = sorted({
        int(round(sample["modeGeneration"]))
        for sample in active_samples
        if math.isfinite(sample.get("modeGeneration", float("nan")))
    })
    clamp_count = sum(1 for sample in active_samples if abs(sample["clampResidual"]) > 1e-3)
    positive_vdot_count = sum(
        1 for sample in active_samples
        if math.isfinite(sample.get("finiteDiffVdot", float("nan"))) and sample["finiteDiffVdot"] > 1e-9
    )
    stage3_available = any(
        math.isfinite(sample.get("candidateCount", float("nan"))) and sample["candidateCount"] > 0.0
        for sample in active_samples
    )
    stage3_display = "block" if stage3_available else "none"
    max_abs_normal_dot = max_abs(active_samples, "normalDotTangent")
    max_abs_desired_normal = max_abs(active_samples, "desiredAccelDotNormal")
    stage4_available = any(
        math.isfinite(sample.get("supervisorMode", float("nan"))) and
        sample.get("supervisorMode", 0.0) > 0.0
        for sample in all_samples
    )
    stage4_display = "block" if stage4_available else "none"

    payload = {
        "allSamples": all_samples,
        "activeSamples": active_samples,
        "positionGain": position_gain,
        "dampingGain": damping_gain,
        "stage3Available": stage3_available,
        "stage4Available": stage4_available,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tangent Escape RMP Diagnostic</title>
<style>
body {{
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f6f7f9;
  color: #18202a;
}}
header {{
  padding: 18px 24px;
  background: #17202a;
  color: white;
}}
main {{
  max-width: 1220px;
  margin: 0 auto;
  padding: 20px 24px 36px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 22px;
}}
h2 {{
  margin: 22px 0 10px;
  font-size: 18px;
}}
.meta {{
  opacity: 0.82;
  font-size: 13px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}}
.card {{
  background: white;
  border: 1px solid #d8dde5;
  border-radius: 8px;
  padding: 12px;
}}
.label {{
  color: #657184;
  font-size: 12px;
  margin-bottom: 4px;
}}
.value {{
  font-size: 18px;
  font-weight: 700;
}}
.note {{
  background: #fff8df;
  border: 1px solid #e9d891;
  border-radius: 8px;
  padding: 12px;
  line-height: 1.45;
  margin: 14px 0;
}}
.chart {{
  background: white;
  border: 1px solid #d8dde5;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 14px;
}}
.chart-title {{
  font-weight: 700;
  margin-bottom: 8px;
}}
svg {{
  width: 100%;
  height: 310px;
  display: block;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 12px;
  color: #4c5969;
  margin-top: 8px;
}}
.swatch {{
  display: inline-block;
  width: 12px;
  height: 3px;
  margin-right: 5px;
  vertical-align: middle;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: white;
  border: 1px solid #d8dde5;
}}
th, td {{
  padding: 8px;
  border-bottom: 1px solid #edf0f4;
  text-align: right;
  font-size: 12px;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<header>
  <h1>Tangent Escape RMP Diagnostic</h1>
  <div class="meta">Input CSV: {html.escape(str(csv_path))}</div>
</header>
<main>
  <div class="note">
    This is a diagnostic visualization, not a global stability proof. The plotted
    In mode 4 (<code>stable_hybrid_gds</code>), the energy chart is the sum over all
    latched candidate branches using the bounded log-cosh potential. A generation
    change marks a hybrid reset of direction, score weight, activation, or gain.
    Positive finite-difference <code>dV/dt</code> can still occur because the other
    RMP leaves exchange energy with this leaf; the structural GDS condition is the
    non-positive damping curve within a fixed generation. A global hybrid stability
    claim additionally requires checking energy at every generation transition.
  </div>
  <section class="grid">
    <div class="card"><div class="label">Active Rows</div><div class="value">{len(active_samples)}</div></div>
    <div class="card"><div class="label">Active Intervals</div><div class="value">{len(intervals)}</div></div>
    <div class="card"><div class="label">Control Point</div><div class="value">{html.escape(cp_text)}</div></div>
    <div class="card"><div class="label">Mode ID</div><div class="value">{html.escape(mode_text)}</div></div>
    <div class="card"><div class="label">Mode Generations</div><div class="value">{len(mode_generations)}</div></div>
    <div class="card"><div class="label">V Range</div><div class="value">{finite_range(active_samples, "lyapunov")}</div></div>
    <div class="card"><div class="label">Positive dV/dt Samples</div><div class="value">{positive_vdot_count}</div></div>
    <div class="card"><div class="label">Force Model Mismatches</div><div class="value">{clamp_count}</div></div>
    <div class="card"><div class="label">Max |Force Residual|</div><div class="value">{fmt(max_abs(active_samples, "clampResidual"))}</div></div>
    <div class="card"><div class="label">Candidate Count</div><div class="value">{finite_range(active_samples, "candidateCount")}</div></div>
    <div class="card"><div class="label">Selected Weight</div><div class="value">{finite_range(active_samples, "selectedWeight")}</div></div>
    <div class="card"><div class="label">Weight Entropy</div><div class="value">{finite_range(active_samples, "weightEntropy")}</div></div>
    <div class="card"><div class="label">Softmax Beta</div><div class="value">{finite_range(active_samples, "softmaxBeta")}</div></div>
    <div class="card"><div class="label">Max |n dot t|</div><div class="value">{fmt(max_abs_normal_dot)}</div></div>
    <div class="card"><div class="label">Max |a dot n|</div><div class="value">{fmt(max_abs_desired_normal)}</div></div>
    <div class="card"><div class="label">Score Gap</div><div class="value">{finite_range(active_samples, "selectedScoreGap")}</div></div>
    <div class="card"><div class="label">Max Duplicate Risk</div><div class="value">{finite_range(active_samples, "maxCandidateDuplicateRisk")}</div></div>
    <div class="card"><div class="label">Max Adjacent Risk</div><div class="value">{finite_range(active_samples, "maxCandidateAdjacentRisk")}</div></div>
    <div class="card"><div class="label">Supervisor Mode</div><div class="value">{finite_range(all_samples, "supervisorMode")}</div></div>
    <div class="card"><div class="label">Hold Bonus</div><div class="value">{finite_range(active_samples, "holdBonus")}</div></div>
    <div class="card"><div class="label">Stuck Active</div><div class="value">{finite_range(active_samples, "stuckActive")}</div></div>
    <div class="card"><div class="label">Blocked Penalty</div><div class="value">{finite_range(active_samples, "blockedPenalty")}</div></div>
  </section>

  <h2>Energy And Dissipation</h2>
  <div class="chart">
    <div class="chart-title">Lyapunov Candidate Components</div>
    <svg id="energyChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#1f77b4"></span>V</span>
      <span><span class="swatch" style="background:#2ca02c"></span>potential</span>
      <span><span class="swatch" style="background:#ff7f0e"></span>kinetic</span>
    </div>
  </div>
  <div class="chart">
    <div class="chart-title">Finite-Difference dV/dt vs Damping Prediction</div>
    <svg id="vdotChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#d62728"></span>finite diff dV/dt</span>
      <span><span class="swatch" style="background:#9467bd"></span>-M*b*sdot^2</span>
      <span><span class="swatch" style="background:#7f7f7f"></span>zero</span>
    </div>
  </div>

  <h2>Stage-2 Scalar State</h2>
  <div class="chart">
    <div class="chart-title">Scalar Progress</div>
    <svg id="scalarChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#1f77b4"></span>s</span>
      <span><span class="swatch" style="background:#111"></span>target</span>
      <span><span class="swatch" style="background:#d62728"></span>error</span>
    </div>
  </div>
  <div class="chart">
    <div class="chart-title">Acceleration Command And Force Model Residual</div>
    <svg id="accelChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#1f77b4"></span>desired accel</span>
      <span><span class="swatch" style="background:#ff7f0e"></span>potential+damping prediction</span>
      <span><span class="swatch" style="background:#d62728"></span>model residual</span>
    </div>
  </div>

  <section style="display:{stage3_display}">
    <h2>Stage-3 Score And Softmax</h2>
    <div class="chart">
      <div class="chart-title">Selected Candidate Score Terms</div>
      <svg id="stage3ScoreChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#1f77b4"></span>selected score</span>
        <span><span class="swatch" style="background:#2ca02c"></span>goal score</span>
        <span><span class="swatch" style="background:#ff7f0e"></span>continuity score</span>
        <span><span class="swatch" style="background:#8c564b"></span>duplicate risk</span>
        <span><span class="swatch" style="background:#d62728"></span>adjacent risk</span>
      </div>
    </div>
    <div class="chart">
      <div class="chart-title">Softmax Selection</div>
      <svg id="stage3WeightChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#1f77b4"></span>selected weight</span>
        <span><span class="swatch" style="background:#2ca02c"></span>branch weight sum</span>
        <span><span class="swatch" style="background:#9467bd"></span>entropy</span>
      </div>
    </div>
    <div class="chart">
      <div class="chart-title">Tangent Quality</div>
      <svg id="stage3TangentChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#1f77b4"></span>normal dot tangent</span>
        <span><span class="swatch" style="background:#d62728"></span>desired accel dot normal</span>
        <span><span class="swatch" style="background:#9467bd"></span>max candidate |normal dot tangent|</span>
      </div>
    </div>
  </section>

  <section style="display:{stage4_display}">
    <h2>Stage-4 Supervisor</h2>
    <div class="note">
      Supervisor mode: 0=off, 1=preventive/hold, 2=stuck boost, 3=recovery.
      The forced_stuck launch profile is only for checking these transitions and memory wiring.
    </div>
    <div class="chart">
      <div class="chart-title">Mode, Hold, Stuck, And Memory</div>
      <svg id="stage4SupervisorChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#111111"></span>supervisor mode</span>
        <span><span class="swatch" style="background:#1f77b4"></span>hold active</span>
        <span><span class="swatch" style="background:#2ca02c"></span>hold bonus</span>
        <span><span class="swatch" style="background:#d62728"></span>stuck active</span>
        <span><span class="swatch" style="background:#9467bd"></span>blocked memory</span>
      </div>
    </div>
    <div class="chart">
      <div class="chart-title">Branch Progress And Boost</div>
      <svg id="stage4ProgressChart"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#1f77b4"></span>branch progress m</span>
        <span><span class="swatch" style="background:#ff7f0e"></span>clearance improvement m</span>
        <span><span class="swatch" style="background:#d62728"></span>metric boost</span>
        <span><span class="swatch" style="background:#9467bd"></span>accel boost</span>
      </div>
    </div>
  </section>

  <h2>Closed-Loop Context</h2>
  <div class="chart">
    <div class="chart-title">Activation And Clearance</div>
    <svg id="activationChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#1f77b4"></span>activation</span>
      <span><span class="swatch" style="background:#2ca02c"></span>clearance m</span>
    </div>
  </div>
  <div class="chart">
    <div class="chart-title">Joint Acceleration Norm And Goal Error</div>
    <svg id="closedLoopChart"></svg>
    <div class="legend">
      <span><span class="swatch" style="background:#d62728"></span>qdd norm</span>
      <span><span class="swatch" style="background:#1f77b4"></span>goal error</span>
    </div>
  </div>

  <h2>Active Samples</h2>
  <table id="sampleTable"></table>
</main>
<script>
const data = {payload_json};

function finite(v) {{ return Number.isFinite(v); }}
function extent(seriesList, key) {{
  const values = [];
  for (const series of seriesList) {{
    for (const p of series.points) {{
      const v = p[key || series.y];
      if (finite(v)) values.push(v);
    }}
  }}
  if (!values.length) return [0, 1];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {{ lo -= 1; hi += 1; }}
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}}
function fmt(v) {{
  if (!finite(v)) return "";
  if (Math.abs(v) >= 1000 || Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  return v.toPrecision(4);
}}
function drawChart(id, points, series, opts={{}}) {{
  const svg = document.getElementById(id);
  const width = svg.clientWidth || 1000;
  const height = svg.clientHeight || 310;
  const margin = {{left: 58, right: 18, top: 18, bottom: 38}};
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = "";
  const xVals = points.map(p => p.time).filter(finite);
  if (!xVals.length) return;
  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  const yExt = extent(series.map(s => ({{...s, points}})));
  const yMin = opts.yMin ?? yExt[0], yMax = opts.yMax ?? yExt[1];
  const px = x => margin.left + (x - xMin) / Math.max(xMax - xMin, 1e-9) * (width - margin.left - margin.right);
  const py = y => height - margin.bottom - (y - yMin) / Math.max(yMax - yMin, 1e-9) * (height - margin.top - margin.bottom);
  function line(x1,y1,x2,y2,color,widthValue=1,dash="") {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
    el.setAttribute("x1", x1); el.setAttribute("y1", y1);
    el.setAttribute("x2", x2); el.setAttribute("y2", y2);
    el.setAttribute("stroke", color); el.setAttribute("stroke-width", widthValue);
    if (dash) el.setAttribute("stroke-dasharray", dash);
    svg.appendChild(el);
  }}
  function text(x,y,value,anchor="middle") {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.setAttribute("x", x); el.setAttribute("y", y);
    el.setAttribute("text-anchor", anchor);
    el.setAttribute("font-size", "11");
    el.setAttribute("fill", "#586577");
    el.textContent = value;
    svg.appendChild(el);
  }}
  line(margin.left, margin.top, margin.left, height-margin.bottom, "#cbd2dc");
  line(margin.left, height-margin.bottom, width-margin.right, height-margin.bottom, "#cbd2dc");
  for (let i=0; i<=4; i++) {{
    const y = yMin + (yMax-yMin)*i/4;
    const yy = py(y);
    line(margin.left, yy, width-margin.right, yy, "#edf0f4");
    text(margin.left-8, yy+4, fmt(y), "end");
  }}
  for (let i=0; i<=5; i++) {{
    const x = xMin + (xMax-xMin)*i/5;
    const xx = px(x);
    line(xx, height-margin.bottom, xx, height-margin.bottom+4, "#cbd2dc");
    text(xx, height-margin.bottom+20, fmt(x));
  }}
  if (opts.zeroLine && yMin < 0 && yMax > 0) {{
    line(margin.left, py(0), width-margin.right, py(0), "#7f7f7f", 1, "5 4");
  }}
  for (const s of series) {{
    let d = "";
    for (const p of points) {{
      const x = p.time, y = p[s.y];
      if (!finite(x) || !finite(y)) continue;
      d += (d ? "L" : "M") + px(x) + "," + py(y);
    }}
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", s.width || 2);
    if (s.dash) path.setAttribute("stroke-dasharray", s.dash);
    svg.appendChild(path);
  }}
}}

drawChart("energyChart", data.activeSamples, [
  {{y:"lyapunov", color:"#1f77b4", width:2.5}},
  {{y:"potential", color:"#2ca02c", width:2}},
  {{y:"kinetic", color:"#ff7f0e", width:2}}
]);
drawChart("vdotChart", data.activeSamples, [
  {{y:"finiteDiffVdot", color:"#d62728", width:2.5}},
  {{y:"dampingVdot", color:"#9467bd", width:2}},
], {{zeroLine:true}});
drawChart("scalarChart", data.activeSamples, [
  {{y:"scalarS", color:"#1f77b4", width:2.5}},
  {{y:"target", color:"#111", width:2, dash:"6 4"}},
  {{y:"error", color:"#d62728", width:2}}
], {{zeroLine:true}});
drawChart("accelChart", data.activeSamples, [
  {{y:"desiredAccel", color:"#1f77b4", width:2.5}},
  {{y:"unclampedAccel", color:"#ff7f0e", width:2, dash:"6 4"}},
  {{y:"clampResidual", color:"#d62728", width:2}}
], {{zeroLine:true}});
if (data.stage3Available) {{
  drawChart("stage3ScoreChart", data.activeSamples, [
    {{y:"selectedScore", color:"#1f77b4", width:2.5}},
    {{y:"selectedGoalScore", color:"#2ca02c", width:2}},
    {{y:"selectedContinuityScore", color:"#ff7f0e", width:2}},
    {{y:"selectedDuplicateRisk", color:"#8c564b", width:2}},
    {{y:"selectedAdjacentRisk", color:"#d62728", width:2}}
  ], {{zeroLine:true}});
  drawChart("stage3WeightChart", data.activeSamples, [
    {{y:"selectedWeight", color:"#1f77b4", width:2.5}},
    {{y:"branchWeightSum", color:"#2ca02c", width:2}},
    {{y:"weightEntropy", color:"#9467bd", width:2}}
  ], {{zeroLine:true}});
  drawChart("stage3TangentChart", data.activeSamples, [
    {{y:"normalDotTangent", color:"#1f77b4", width:2.5}},
    {{y:"desiredAccelDotNormal", color:"#d62728", width:2}},
    {{y:"maxCandidateNormalDotTangent", color:"#9467bd", width:2}}
  ], {{zeroLine:true}});
}}
if (data.stage4Available) {{
  drawChart("stage4SupervisorChart", data.allSamples, [
    {{y:"supervisorMode", color:"#111111", width:2.5}},
    {{y:"holdActive", color:"#1f77b4", width:2.5}},
    {{y:"holdBonus", color:"#2ca02c", width:2}},
    {{y:"stuckActive", color:"#d62728", width:2}},
    {{y:"maxBlockedMemory", color:"#9467bd", width:2}}
  ], {{zeroLine:true}});
  drawChart("stage4ProgressChart", data.allSamples, [
    {{y:"branchProgress", color:"#1f77b4", width:2.5}},
    {{y:"clearanceImprovement", color:"#ff7f0e", width:2}},
    {{y:"metricBoost", color:"#d62728", width:2}},
    {{y:"accelBoost", color:"#9467bd", width:2}}
  ], {{zeroLine:true}});
}}
drawChart("activationChart", data.activeSamples, [
  {{y:"activation", color:"#1f77b4", width:2.5}},
  {{y:"clearance", color:"#2ca02c", width:2}}
]);
drawChart("closedLoopChart", data.allSamples, [
  {{y:"qddNorm", color:"#d62728", width:2}},
  {{y:"goalError", color:"#1f77b4", width:2}}
]);

const table = document.getElementById("sampleTable");
const columns = [
  ["row","row"], ["time","t"], ["lyapunov","V"], ["finiteDiffVdot","dV/dt"],
  ["dampingVdot","damping"], ["scalarS","s"], ["error","error"],
  ["velocity","sdot"], ["desiredAccel","accel"], ["qddNorm","qdd"],
  ["modeGeneration","generation"], ["weightsLatched","latched"]
];
if (data.stage3Available) {{
  columns.push(
    ["candidateCount","cand"],
    ["selectedCandidateIndex","sel"],
    ["selectedWeight","w"],
    ["selectedScore","score"],
    ["selectedGoalScore","goal"],
    ["selectedContinuityScore","cont"],
    ["selectedDuplicateRisk","dup"],
    ["selectedAdjacentRisk","adj"],
    ["selectedScoreGap","gap"],
    ["normalDotTangent","n.t"],
    ["holdBonus","hold"],
    ["stuckActive","stuck"],
    ["blockedPenalty","blocked"]
  );
}}
let htmlTable = "<thead><tr>" + columns.map(c => `<th>${{c[1]}}</th>`).join("") + "</tr></thead><tbody>";
for (const p of data.activeSamples) {{
  htmlTable += "<tr>" + columns.map(c => `<td>${{fmt(p[c[0]])}}</td>`).join("") + "</tr>";
}}
htmlTable += "</tbody>";
table.innerHTML = htmlTable;
</script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Stage-2 tangent escape Lyapunov diagnostic HTML report."
    )
    parser.add_argument("csv", nargs="?", type=Path, help="Trace CSV. Defaults to latest compatible CSV.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--save", type=Path, help="Output HTML path.")
    parser.add_argument(
        "--position-gain",
        type=float,
        default=16.0,
        help="Stage-2 scalar position gain k_s used for V and unclamped accel.",
    )
    parser.add_argument(
        "--damping-gain",
        type=float,
        default=4.0,
        help="Stage-2 scalar damping gain b_s used for damping dissipation.",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        default=0.6,
        help="Bounded spring acceleration limit used by stable_hybrid_gds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.expanduser() if args.csv else latest_compatible_csv(args.log_dir.expanduser())
    rows, _ = load_rows(csv_path)
    all_samples, active_samples, intervals = make_samples(
        rows,
        position_gain=args.position_gain,
        damping_gain=args.damping_gain,
        max_accel=args.max_accel,
    )
    if not active_samples:
        raise ValueError(f"No active tangent escape RMP rows found in {csv_path}")
    output = args.save.expanduser() if args.save else csv_path.with_name(
        f"{csv_path.stem}_tangent_escape_lyapunov.html"
    )
    write_report(
        csv_path,
        output,
        all_samples,
        active_samples,
        intervals,
        position_gain=args.position_gain,
        damping_gain=args.damping_gain,
    )
    positive_vdot = sum(
        1 for sample in active_samples
        if math.isfinite(sample.get("finiteDiffVdot", float("nan"))) and sample["finiteDiffVdot"] > 1e-9
    )
    clamp_count = sum(1 for sample in active_samples if abs(sample["clampResidual"]) > 1e-3)
    print(f"Input CSV: {csv_path}")
    print(f"Active rows: {len(active_samples)}")
    print(f"Positive dV/dt samples: {positive_vdot}")
    print(f"Clamp samples: {clamp_count}")
    print(f"Saved report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
