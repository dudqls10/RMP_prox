#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LOG_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()
REQUIRED_COLUMNS = [
    "time_ros_s",
    "tangent_escape_rmp_active",
    "tangent_escape_rmp_leaf_mode_id",
    "tangent_escape_rmp_activation",
    "tangent_escape_rmp_clearance_m",
    "tangent_escape_rmp_effective_metric_scalar",
    "tangent_escape_rmp_scalar_s_m",
    "tangent_escape_rmp_scalar_target_m",
    "tangent_escape_rmp_scalar_velocity_m_s",
    "tangent_escape_rmp_scalar_error_m",
    "tangent_escape_rmp_desired_tangent_accel_m_s2",
    "rmp_joint_accel_1_rad_s2",
    "rmp_joint_accel_2_rad_s2",
    "rmp_joint_accel_3_rad_s2",
    "rmp_joint_accel_4_rad_s2",
    "rmp_joint_accel_5_rad_s2",
    "rmp_joint_accel_6_rad_s2",
]
STAGE4_COLUMNS = [
    "tangent_escape_rmp_supervisor_mode_id",
    "tangent_escape_rmp_hold_active",
    "tangent_escape_rmp_selected_hold_bonus",
    "tangent_escape_rmp_stuck_timer_s",
    "tangent_escape_rmp_stuck_active",
    "tangent_escape_rmp_metric_boost",
    "tangent_escape_rmp_accel_boost",
    "tangent_escape_rmp_selected_blocked_penalty",
    "tangent_escape_rmp_max_blocked_memory",
    "tangent_escape_rmp_branch_age_s",
    "tangent_escape_rmp_branch_progress_m",
    "tangent_escape_rmp_clearance_improvement_m",
]


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str


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


def fmt_range(values: Sequence[float], digits: int = 4) -> str:
    finite = finite_values(values)
    if not finite:
        return "n/a"
    return (
        f"min={fmt(min(finite), digits)} max={fmt(max(finite), digits)} "
        f"first={fmt(finite[0], digits)} last={fmt(finite[-1], digits)}"
    )


def norm(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


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
        f"No compatible tangent escape RMP trace CSV found in {log_dir}"
    )


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError("Missing validation columns: " + ", ".join(missing))
    return rows, fieldnames


def row_time(row: Dict[str, str]) -> float:
    return parse_float(row, "time_ros_s")


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


def interval_duration(rows: Sequence[Dict[str, str]], start: int, end: int) -> float:
    start_t = row_time(rows[start])
    end_t = row_time(rows[end])
    if math.isfinite(start_t) and math.isfinite(end_t):
        return max(end_t - start_t, 0.0)
    return float("nan")


def qdd_vector(row: Dict[str, str]) -> List[float]:
    return [parse_float(row, f"rmp_joint_accel_{index}_rad_s2") for index in range(1, 7)]


def qdd_norm(row: Dict[str, str]) -> float:
    value = parse_float(row, "rmp_joint_accel_norm")
    if math.isfinite(value):
        return value
    return norm(qdd_vector(row))


def goal_error(row: Dict[str, str]) -> float:
    value = parse_float(row, "goal_error_m")
    if math.isfinite(value):
        return value
    goal = [parse_float(row, f"controller_goal_{axis}") for axis in ("x", "y", "z")]
    ee = [parse_float(row, f"rmp_ee_{axis}") for axis in ("x", "y", "z")]
    if all(math.isfinite(value) for value in goal + ee):
        return norm([ee[index] - goal[index] for index in range(3)])
    return float("nan")


def vector_norm(row: Dict[str, str], prefix: str) -> float:
    values = [parse_float(row, f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    return norm(values)


def vector_dot(row: Dict[str, str], lhs_prefix: str, rhs_prefix: str) -> float:
    lhs = [parse_float(row, f"{lhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    rhs = [parse_float(row, f"{rhs_prefix}_{axis}") for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in lhs + rhs):
        return float("nan")
    return sum(lhs[index] * rhs[index] for index in range(3))


def max_jerk(rows: Sequence[Dict[str, str]]) -> Tuple[float, Optional[int]]:
    best = float("nan")
    best_index: Optional[int] = None
    previous_qdd: Optional[List[float]] = None
    previous_time = float("nan")
    previous_index: Optional[int] = None
    for index, row in enumerate(rows):
        current_qdd = qdd_vector(row)
        current_time = row_time(row)
        if not all(math.isfinite(value) for value in current_qdd) or not math.isfinite(current_time):
            continue
        if previous_qdd is not None and math.isfinite(previous_time):
            dt = current_time - previous_time
            if dt > 1e-6:
                value = norm([
                    (current_qdd[joint] - previous_qdd[joint]) / dt
                    for joint in range(6)
                ])
                if math.isfinite(value) and (not math.isfinite(best) or value > best):
                    best = value
                    best_index = previous_index if previous_index is not None else index
        previous_qdd = current_qdd
        previous_time = current_time
        previous_index = index
    return best, best_index


def parse_candidates(row: Dict[str, str]) -> List[Dict[str, float]]:
    payload = row.get("tangent_escape_rmp_candidates_json", "")
    if not payload:
        return []
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return []
    candidates = decoded.get("candidates") if isinstance(decoded, dict) else None
    if not isinstance(candidates, list):
        return []
    result: List[Dict[str, float]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parsed: Dict[str, float] = {}
        for key in (
            "slot",
            "weight",
            "score",
            "duplicateRisk",
            "adjacentRisk",
            "holdBonus",
            "blockedPenalty",
            "metricScalar",
            "metricBoost",
            "accelBoost",
            "normalDotTangent",
            "scalarS",
            "scalarVelocity",
            "scalarError",
            "potentialEnergy",
            "kineticEnergy",
            "lyapunovEnergy",
            "dampingVdot",
            "weightsLatched",
            "modeGeneration",
            "boundedPotential",
            "modeNormalDotTangent",
            "clearanceRate",
            "collisionAccel",
            "scaledCollisionAccel",
        ):
            try:
                parsed[key] = float(candidate.get(key, float("nan")))
            except (TypeError, ValueError):
                parsed[key] = float("nan")
        direction = candidate.get("direction")
        if isinstance(direction, list) and len(direction) >= 3:
            for axis, value in zip(("X", "Y", "Z"), direction[:3]):
                try:
                    parsed[f"direction{axis}"] = float(value)
                except (TypeError, ValueError):
                    parsed[f"direction{axis}"] = float("nan")
        result.append(parsed)
    return result


def lyapunov(row: Dict[str, str], position_gain: float) -> float:
    aggregate = parse_float(row, "tangent_escape_rmp_escape_lyapunov")
    if math.isfinite(aggregate):
        return aggregate
    metric = parse_float(row, "tangent_escape_rmp_effective_metric_scalar")
    velocity = parse_float(row, "tangent_escape_rmp_scalar_velocity_m_s")
    error = parse_float(row, "tangent_escape_rmp_scalar_error_m")
    if not all(math.isfinite(value) for value in (metric, velocity, error)):
        return float("nan")
    return 0.5 * metric * velocity * velocity + 0.5 * metric * position_gain * error * error


def fixed_mode_vdot_stats(
    active_rows: Sequence[Tuple[int, Dict[str, str]]],
    position_gain: float,
) -> Tuple[int, int, float, float]:
    pair_count = 0
    positive_count = 0
    max_vdot = float("nan")
    min_vdot = float("nan")
    for previous, current in zip(active_rows, active_rows[1:]):
        previous_index, previous_row = previous
        current_index, current_row = current
        if current_index != previous_index + 1:
            continue
        same_cp = (
            round(parse_float(previous_row, "tangent_escape_rmp_control_point_index")) ==
            round(parse_float(current_row, "tangent_escape_rmp_control_point_index"))
        )
        same_candidate = (
            round(parse_float(previous_row, "tangent_escape_rmp_selected_candidate_index")) ==
            round(parse_float(current_row, "tangent_escape_rmp_selected_candidate_index"))
        )
        same_supervisor = (
            round(parse_float(previous_row, "tangent_escape_rmp_supervisor_mode_id")) ==
            round(parse_float(current_row, "tangent_escape_rmp_supervisor_mode_id"))
        )
        previous_generation = parse_float(
            previous_row,
            "tangent_escape_rmp_mode_generation",
        )
        current_generation = parse_float(
            current_row,
            "tangent_escape_rmp_mode_generation",
        )
        same_generation = (
            not math.isfinite(previous_generation) or
            not math.isfinite(current_generation) or
            round(previous_generation) == round(current_generation)
        )
        if not (same_cp and same_candidate and same_supervisor and same_generation):
            continue
        t0 = row_time(previous_row)
        t1 = row_time(current_row)
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            continue
        v0 = lyapunov(previous_row, position_gain)
        v1 = lyapunov(current_row, position_gain)
        if not (math.isfinite(v0) and math.isfinite(v1)):
            continue
        vdot = (v1 - v0) / (t1 - t0)
        pair_count += 1
        if vdot > 1e-9:
            positive_count += 1
        max_vdot = vdot if not math.isfinite(max_vdot) else max(max_vdot, vdot)
        min_vdot = vdot if not math.isfinite(min_vdot) else min(min_vdot, vdot)
    return pair_count, positive_count, min_vdot, max_vdot


def add_check(checks: List[Check], section: str, name: str, status: str, detail: str) -> None:
    checks.append(Check(section=section, name=name, status=status, detail=detail))


def summarize(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Check], Dict[str, str]]:
    intervals = active_intervals(rows)
    active_indices = [index for start, end in intervals for index in range(start, end + 1)]
    active = [(index, rows[index]) for index in active_indices]
    active_rows = [row for _, row in active]
    checks: List[Check] = []
    summary: Dict[str, str] = {}

    duration = float("nan")
    if rows:
        first_time = row_time(rows[0])
        last_time = row_time(rows[-1])
        if math.isfinite(first_time) and math.isfinite(last_time):
            duration = max(last_time - first_time, 0.0)
    active_duration = sum(interval_duration(rows, start, end) for start, end in intervals)
    summary["rows"] = str(len(rows))
    summary["duration_s"] = fmt(duration)
    summary["active_rows"] = str(len(active_rows))
    summary["active_intervals"] = str(len(intervals))
    summary["active_duration_s"] = fmt(active_duration)
    summary["supervisor_modes"] = "n/a"
    summary["hold_rows"] = "0"
    summary["stuck_rows"] = "0"
    summary["blocked_memory"] = "n/a"

    if not active_rows:
        add_check(checks, "Performance", "Escape activation", "FAIL", "No active RMP escape rows.")
        add_check(checks, "Closure", "Leaf checks", "FAIL", "No active leaf rows to validate.")
        add_check(checks, "Fixed-mode stability", "Lyapunov check", "FAIL", "No active rows.")
        add_check(checks, "Safety / invariance", "Empirical clearance", "FAIL", "No active rows.")
        add_check(
            checks,
            "Stage-4 supervisor",
            "Supervisor activation",
            "FAIL" if args.stage4_profile != "auto" else "INFO",
            "No active escape rows were recorded.",
        )
        return checks, summary

    released = intervals[-1][1] < len(rows) - 1
    add_check(
        checks,
        "Performance",
        "Escape activation",
        "PASS",
        f"active_rows={len(active_rows)}, intervals={len(intervals)}, active_duration={fmt(active_duration)} s",
    )
    add_check(
        checks,
        "Performance",
        "Release before log end",
        "PASS" if released else "WARN",
        "Escape became inactive before the log ended." if released else "Escape was still active at log end.",
    )

    collision_scaled_active = any(
        math.isfinite(parse_float(row, "tangent_escape_rmp_leaf_mode_id")) and
        int(round(parse_float(row, "tangent_escape_rmp_leaf_mode_id"))) == 5
        for row in active_rows
    )
    scalar_s_values = finite_values(parse_float(row, "tangent_escape_rmp_scalar_s_m") for row in active_rows)
    scalar_target_values = finite_values(
        parse_float(row, "tangent_escape_rmp_scalar_target_m") for row in active_rows
    )
    if scalar_s_values and scalar_target_values and abs(scalar_target_values[0]) > 1e-9:
        progress_ratio = max(scalar_s_values) / scalar_target_values[0]
        add_check(
            checks,
            "Performance",
            "Tangent scalar progress",
            "PASS" if progress_ratio >= args.scalar_progress_ratio else "WARN",
            (
                f"max(s)/target={fmt(progress_ratio)}, "
                f"s_range={fmt_range(scalar_s_values)}, target={fmt(scalar_target_values[0])}"
            ),
        )
    else:
        add_check(
            checks,
            "Performance",
            "Tangent scalar progress",
            "INFO" if collision_scaled_active else "WARN",
            (
                "Not applicable to collision_scaled mode; it has no displacement target."
                if collision_scaled_active else
                "No finite scalar target/progress."
            ),
        )

    goal_values = finite_values(goal_error(row) for row in rows)
    active_goal_values = finite_values(goal_error(row) for row in active_rows)
    if len(active_goal_values) >= 2:
        goal_progress = active_goal_values[0] - active_goal_values[-1]
        add_check(
            checks,
            "Performance",
            "Goal progress during escape",
            "PASS" if goal_progress > 0.0 else "WARN",
            (
                f"start={fmt(active_goal_values[0])} m, end={fmt(active_goal_values[-1])} m, "
                f"progress={fmt(goal_progress)} m"
            ),
        )
    elif goal_values:
        add_check(checks, "Performance", "Goal progress during escape", "INFO", "Goal error exists but not enough active samples.")
    else:
        add_check(checks, "Performance", "Goal progress during escape", "INFO", "Goal error was not logged.")

    selected_keys: List[Tuple[int, int]] = []
    for row in active_rows:
        cp = parse_float(row, "tangent_escape_rmp_control_point_index")
        slot = parse_float(row, "tangent_escape_rmp_selected_candidate_index")
        if math.isfinite(cp) and math.isfinite(slot):
            selected_keys.append((int(round(cp)), int(round(slot))))
    branch_switches = sum(1 for lhs, rhs in zip(selected_keys, selected_keys[1:]) if lhs != rhs)
    summary["branch_switches"] = str(branch_switches)
    add_check(
        checks,
        "Performance",
        "Branch switching",
        "INFO",
        f"selected CP/slot switches={branch_switches} across {len(selected_keys)} active samples",
    )

    mode_ids = finite_values(parse_float(row, "tangent_escape_rmp_leaf_mode_id") for row in active_rows)
    invalid_modes = [value for value in mode_ids if int(round(value)) not in {2, 3, 4, 5}]
    add_check(
        checks,
        "Closure",
        "In-tree tangent leaf mode",
        "PASS" if mode_ids and not invalid_modes else "FAIL",
        f"leaf mode ids={sorted(set(int(round(value)) for value in mode_ids)) if mode_ids else 'n/a'}",
    )
    collision_scaled_rows = [
        row for row in active_rows
        if math.isfinite(parse_float(row, "tangent_escape_rmp_leaf_mode_id")) and
        int(round(parse_float(row, "tangent_escape_rmp_leaf_mode_id"))) == 5
    ]
    if collision_scaled_rows:
        scale_values = finite_values(
            parse_float(row, "tangent_escape_rmp_collision_accel_scale")
            for row in collision_scaled_rows
        )
        scale_spread = max(scale_values) - min(scale_values) if scale_values else float("inf")
        scale_matches_argument = (
            not scale_values or args.collision_accel_scale is None or
            max(abs(value - args.collision_accel_scale) for value in scale_values) <=
            args.collision_formula_tol
        )
        add_check(
            checks,
            "Collision-scaled policy",
            "Scalar acceleration scale",
            "PASS" if scale_values and scale_spread <= args.collision_formula_tol and scale_matches_argument else "FAIL",
            (
                f"scale={fmt_range(scale_values)}, spread={fmt(scale_spread)}, "
                f"expected={fmt(args.collision_accel_scale) if args.collision_accel_scale is not None else 'logged value'}"
            ),
        )

        formula_residuals = []
        target_values = []
        error_values = []
        for row in collision_scaled_rows:
            scaled_accel = parse_float(
                row,
                "tangent_escape_rmp_scaled_collision_accel_m_s2",
            )
            tangent_velocity = parse_float(
                row,
                "tangent_escape_rmp_scalar_velocity_m_s",
            )
            desired_accel = parse_float(
                row,
                "tangent_escape_rmp_desired_tangent_accel_m_s2",
            )
            accel_boost = parse_float(row, "tangent_escape_rmp_accel_boost")
            if not math.isfinite(accel_boost):
                accel_boost = 1.0
            if all(math.isfinite(value) for value in (scaled_accel, tangent_velocity, desired_accel)):
                limit = args.max_accel * accel_boost
                expected = accel_boost * scaled_accel - args.damping_gain * tangent_velocity
                if limit > 0.0:
                    expected = max(-limit, min(limit, expected))
                formula_residuals.append(desired_accel - expected)
            target_values.append(parse_float(row, "tangent_escape_rmp_scalar_target_m"))
            error_values.append(parse_float(row, "tangent_escape_rmp_scalar_error_m"))

        max_formula_residual = (
            max(abs(value) for value in formula_residuals)
            if formula_residuals else float("inf")
        )
        add_check(
            checks,
            "Collision-scaled policy",
            "Acceleration equation",
            "PASS" if formula_residuals and max_formula_residual <= args.collision_formula_tol else "FAIL",
            (
                "desired=clip(accel_boost*scaled_collision-damping*tangent_velocity); "
                f"residual={fmt_range(formula_residuals)}"
            ),
        )
        finite_targets = finite_values(target_values)
        finite_errors = finite_values(error_values)
        no_virtual_target = (
            finite_targets and finite_errors and
            max(abs(value) for value in finite_targets + finite_errors) <=
            args.collision_formula_tol
        )
        add_check(
            checks,
            "Collision-scaled policy",
            "No virtual displacement target",
            "PASS" if no_virtual_target else "FAIL",
            f"target={fmt_range(finite_targets)}, error={fmt_range(finite_errors)}",
        )
        add_check(
            checks,
            "Fixed-mode stability",
            "Collision-scaled GDS claim",
            "INFO",
            (
                "leaf mode 5 is a bounded direct-acceleration RMP. It is inside the RMP tree, "
                "but it is not the bounded-potential GDS used by modes 2-4."
            ),
        )

    filter_active_values = finite_values(parse_float(row, "tangent_escape_active") for row in rows)
    filter_active = any(value >= 0.5 for value in filter_active_values)
    add_check(
        checks,
        "Closure",
        "No post-solve filter contamination",
        "PASS" if not filter_active else "WARN",
        "post-solve tangent filter inactive" if not filter_active else "post-solve tangent filter was active in this CSV",
    )

    metrics = finite_values(parse_float(row, "tangent_escape_rmp_effective_metric_scalar") for row in active_rows)
    negative_metrics = [value for value in metrics if value < -args.metric_eps]
    zero_metric_with_activation = [
        row for row in active_rows
        if parse_float(row, "tangent_escape_rmp_activation") > 1e-6 and
        parse_float(row, "tangent_escape_rmp_effective_metric_scalar") <= args.metric_eps
    ]
    add_check(
        checks,
        "Closure",
        "PSD metric scalar",
        "PASS" if metrics and not negative_metrics and not zero_metric_with_activation else "FAIL",
        (
            f"effective_metric={fmt_range(metrics)}, negative_count={len(negative_metrics)}, "
            f"zero_metric_active_count={len(zero_metric_with_activation)}"
        ),
    )

    candidate_metric_values: List[float] = []
    candidate_normal_dots: List[float] = []
    candidate_mode_normal_dots: List[float] = []
    candidate_duplicate_risks: List[float] = []
    for row in active_rows:
        for candidate in parse_candidates(row):
            value = candidate.get("metricScalar", float("nan"))
            if math.isfinite(value):
                candidate_metric_values.append(value)
            dot = candidate.get("normalDotTangent", float("nan"))
            if math.isfinite(dot):
                candidate_normal_dots.append(abs(dot))
            mode_dot = candidate.get("modeNormalDotTangent", float("nan"))
            if math.isfinite(mode_dot):
                candidate_mode_normal_dots.append(abs(mode_dot))
            duplicate_risk = candidate.get("duplicateRisk", float("nan"))
            if math.isfinite(duplicate_risk):
                candidate_duplicate_risks.append(duplicate_risk)
    negative_candidate_metrics = [value for value in candidate_metric_values if value < -args.metric_eps]
    add_check(
        checks,
        "Closure",
        "Candidate branch metric scalar",
        "PASS" if candidate_metric_values and not negative_candidate_metrics else "WARN",
        (
            f"candidate_metric={fmt_range(candidate_metric_values)}, "
            f"negative_count={len(negative_candidate_metrics)}"
        ),
    )

    positive_duplicate_risks = [value for value in candidate_duplicate_risks if value > 1e-6]
    if args.expect_duplicate_risk:
        duplicate_status = "PASS" if positive_duplicate_risks else "FAIL"
    else:
        duplicate_status = "PASS" if candidate_duplicate_risks else "INFO"
    add_check(
        checks,
        "Score",
        "Predictive duplicate risk",
        duplicate_status,
        (
            f"candidate_duplicate_risk={fmt_range(candidate_duplicate_risks)}, "
            f"positive_count={len(positive_duplicate_risks)}"
        ),
    )

    normal_dot_values = finite_values(
        parse_float(row, "tangent_escape_rmp_normal_dot_tangent")
        if math.isfinite(parse_float(row, "tangent_escape_rmp_normal_dot_tangent"))
        else vector_dot(row, "tangent_escape_rmp_normal", "tangent_escape_rmp_tangent")
        for row in active_rows
    )
    max_normal_dot = max(abs(value) for value in normal_dot_values) if normal_dot_values else float("nan")
    max_candidate_dot = max(candidate_normal_dots) if candidate_normal_dots else float("nan")
    max_mode_normal_dot = (
        max(candidate_mode_normal_dots) if candidate_mode_normal_dots else float("nan")
    )
    stable_leaf_observed = any(
        math.isfinite(value) and int(round(value)) == 4 for value in mode_ids
    )
    add_check(
        checks,
        "Closure",
        "Fixed task tangent orthogonality",
        "PASS" if (
            math.isfinite(max_mode_normal_dot) and
            max_mode_normal_dot <= args.fixed_direction_tol
        ) else ("INFO" if not stable_leaf_observed else "FAIL"),
        f"max |mode-entry normal dot tangent|={fmt(max_mode_normal_dot)}",
    )
    add_check(
        checks,
        "Closure",
        "Current obstacle tangent drift",
        "PASS" if math.isfinite(max_normal_dot) and max_normal_dot <= args.normal_dot_tol else "WARN",
        (
            f"max |normal dot selected tangent|={fmt(max_normal_dot)}, "
            f"max candidate={fmt(max_candidate_dot)}, threshold={fmt(args.normal_dot_tol)}"
        ),
    )

    tangent_norms = finite_values(vector_norm(row, "tangent_escape_rmp_tangent") for row in active_rows)
    normal_norms = finite_values(vector_norm(row, "tangent_escape_rmp_normal") for row in active_rows)
    bad_tangent_norms = [
        value for value in tangent_norms if abs(value - 1.0) > args.unit_vector_tol
    ]
    bad_normal_norms = [
        value for value in normal_norms if abs(value - 1.0) > args.unit_vector_tol
    ]
    add_check(
        checks,
        "Closure",
        "Unit normal/tangent vectors",
        "PASS" if tangent_norms and normal_norms and not bad_tangent_norms and not bad_normal_norms else "WARN",
        f"tangent_norm={fmt_range(tangent_norms)}, normal_norm={fmt_range(normal_norms)}",
    )

    accel_normal_values = finite_values(
        parse_float(row, "tangent_escape_rmp_desired_accel_dot_normal_m_s2")
        for row in active_rows
    )
    max_accel_normal = max(abs(value) for value in accel_normal_values) if accel_normal_values else float("nan")
    add_check(
        checks,
        "Closure",
        "Tangent acceleration",
        "PASS" if math.isfinite(max_accel_normal) and max_accel_normal <= args.accel_normal_tol else "WARN",
        f"max |desired_accel dot normal|={fmt(max_accel_normal)} m/s^2",
    )

    branch_weight_sums = finite_values(parse_float(row, "tangent_escape_rmp_branch_weight_sum") for row in active_rows)
    bad_weight_sums = [value for value in branch_weight_sums if abs(value - 1.0) > args.weight_sum_tol]
    add_check(
        checks,
        "Closure",
        "Softmax branch weights",
        "PASS" if branch_weight_sums and not bad_weight_sums else "WARN",
        f"branch_weight_sum={fmt_range(branch_weight_sums)}, bad_count={len(bad_weight_sums)}",
    )

    stable_rows = [
        (index, row) for index, row in active
        if math.isfinite(parse_float(row, "tangent_escape_rmp_leaf_mode_id")) and
        int(round(parse_float(row, "tangent_escape_rmp_leaf_mode_id"))) == 4
    ]
    if stable_rows:
        latched_flags = finite_values(
            parse_float(row, "tangent_escape_rmp_weights_latched")
            for _, row in stable_rows
        )
        bounded_flags = finite_values(
            parse_float(row, "tangent_escape_rmp_bounded_potential")
            for _, row in stable_rows
        )
        generations = finite_values(
            parse_float(row, "tangent_escape_rmp_mode_generation")
            for _, row in stable_rows
        )
        add_check(
            checks,
            "Fixed-mode stability",
            "Latched score mode",
            "PASS" if (
                len(latched_flags) == len(stable_rows) and
                min(latched_flags) >= 0.5 and
                generations and min(generations) >= 1.0
            ) else "FAIL",
            (
                f"latched={fmt_range(latched_flags)}, "
                f"generation={fmt_range(generations)}"
            ),
        )
        add_check(
            checks,
            "Fixed-mode stability",
            "Bounded GDS potential enabled",
            "PASS" if (
                len(bounded_flags) == len(stable_rows) and
                min(bounded_flags) >= 0.5
            ) else "FAIL",
            f"bounded_potential={fmt_range(bounded_flags)}",
        )

        fixed_groups: Dict[Tuple[int, int, int], List[Dict[str, float]]] = {}
        all_stable_candidates: List[Dict[str, float]] = []
        for _, row in stable_rows:
            cp = parse_float(row, "tangent_escape_rmp_control_point_index")
            for candidate in parse_candidates(row):
                generation = candidate.get("modeGeneration", float("nan"))
                slot = candidate.get("slot", float("nan"))
                if not all(math.isfinite(value) for value in (cp, generation, slot)):
                    continue
                key = (int(round(cp)), int(round(generation)), int(round(slot)))
                fixed_groups.setdefault(key, []).append(candidate)
                all_stable_candidates.append(candidate)

        max_weight_drift = 0.0
        max_metric_drift = 0.0
        max_score_drift = 0.0
        max_direction_drift = 0.0
        repeated_groups = 0
        for samples in fixed_groups.values():
            if len(samples) < 2:
                continue
            repeated_groups += 1
            for field, current_max_name in (
                ("weight", "weight"),
                ("metricScalar", "metric"),
                ("score", "score"),
            ):
                values = finite_values(sample.get(field, float("nan")) for sample in samples)
                drift = max(values) - min(values) if values else float("inf")
                if current_max_name == "weight":
                    max_weight_drift = max(max_weight_drift, drift)
                elif current_max_name == "metric":
                    max_metric_drift = max(max_metric_drift, drift)
                else:
                    max_score_drift = max(max_score_drift, drift)
            reference = [samples[0].get(f"direction{axis}", float("nan")) for axis in "XYZ"]
            for sample in samples[1:]:
                direction = [sample.get(f"direction{axis}", float("nan")) for axis in "XYZ"]
                if all(math.isfinite(value) for value in reference + direction):
                    max_direction_drift = max(
                        max_direction_drift,
                        norm([direction[axis] - reference[axis] for axis in range(3)]),
                    )
                else:
                    max_direction_drift = float("inf")
        fixed_mode_ok = (
            repeated_groups > 0 and
            max_weight_drift <= args.fixed_weight_tol and
            max_metric_drift <= args.fixed_metric_tol and
            max_score_drift <= args.fixed_score_tol and
            max_direction_drift <= args.fixed_direction_tol
        )
        add_check(
            checks,
            "Fixed-mode stability",
            "Fixed branch parameters",
            "PASS" if fixed_mode_ok else "FAIL",
            (
                f"repeated_groups={repeated_groups}, max drift: weight={fmt(max_weight_drift)}, "
                f"metric={fmt(max_metric_drift)}, score={fmt(max_score_drift)}, "
                f"direction={fmt(max_direction_drift)}"
            ),
        )

        potentials = finite_values(
            candidate.get("potentialEnergy", float("nan"))
            for candidate in all_stable_candidates
        )
        kinetics = finite_values(
            candidate.get("kineticEnergy", float("nan"))
            for candidate in all_stable_candidates
        )
        lyapunovs = finite_values(
            candidate.get("lyapunovEnergy", float("nan"))
            for candidate in all_stable_candidates
        )
        damping_vdots = finite_values(
            candidate.get("dampingVdot", float("nan"))
            for candidate in all_stable_candidates
        )
        energy_residuals = []
        damping_residuals = []
        for candidate in all_stable_candidates:
            potential = candidate.get("potentialEnergy", float("nan"))
            kinetic = candidate.get("kineticEnergy", float("nan"))
            value = candidate.get("lyapunovEnergy", float("nan"))
            damping_vdot = candidate.get("dampingVdot", float("nan"))
            metric_value = candidate.get("metricScalar", float("nan"))
            velocity_value = candidate.get("scalarVelocity", float("nan"))
            accel_boost = candidate.get("accelBoost", float("nan"))
            if all(math.isfinite(item) for item in (potential, kinetic, value)):
                energy_residuals.append(value - potential - kinetic)
            if all(math.isfinite(item) for item in (
                damping_vdot,
                metric_value,
                velocity_value,
                accel_boost,
            )):
                predicted = (
                    -metric_value * args.damping_gain * accel_boost *
                    velocity_value * velocity_value
                )
                damping_residuals.append(damping_vdot - predicted)
        energy_ok = (
            potentials and kinetics and lyapunovs and damping_vdots and
            min(potentials) >= -args.energy_eps and
            min(kinetics) >= -args.energy_eps and
            min(lyapunovs) >= -args.energy_eps and
            max(damping_vdots) <= args.energy_eps and
            max((abs(value) for value in energy_residuals), default=float("inf")) <= args.energy_eps and
            max((abs(value) for value in damping_residuals), default=float("inf")) <= args.energy_eps
        )
        add_check(
            checks,
            "Fixed-mode stability",
            "GDS energy and damping identity",
            "PASS" if energy_ok else "FAIL",
            (
                f"potential={fmt_range(potentials)}, kinetic={fmt_range(kinetics)}, "
                f"V={fmt_range(lyapunovs)}, damping_dV={fmt_range(damping_vdots)}, "
                f"max_energy_residual={fmt(max((abs(value) for value in energy_residuals), default=float('nan')))}, "
                f"max_damping_residual={fmt(max((abs(value) for value in damping_residuals), default=float('nan')))}"
            ),
        )

        force_residuals = []
        for _, row in stable_rows:
            selected_slot = parse_float(row, "tangent_escape_rmp_selected_candidate_index")
            desired = parse_float(row, "tangent_escape_rmp_desired_tangent_accel_m_s2")
            if not (math.isfinite(selected_slot) and math.isfinite(desired)):
                continue
            selected = next(
                (
                    candidate for candidate in parse_candidates(row)
                    if math.isfinite(candidate.get("slot", float("nan"))) and
                    int(round(candidate["slot"])) == int(round(selected_slot))
                ),
                None,
            )
            if selected is None:
                continue
            error = selected.get("scalarError", float("nan"))
            velocity_value = selected.get("scalarVelocity", float("nan"))
            boost = selected.get("accelBoost", float("nan"))
            if not all(math.isfinite(value) for value in (error, velocity_value, boost)):
                continue
            position_gain = args.position_gain * boost
            damping_gain = args.damping_gain * boost
            acceleration_limit = args.max_accel * boost
            spring = position_gain * error
            if acceleration_limit > 0.0:
                spring = acceleration_limit * math.tanh(
                    position_gain * error / acceleration_limit
                )
            predicted = spring - damping_gain * velocity_value
            force_residuals.append(desired - predicted)
        max_force_residual = max((abs(value) for value in force_residuals), default=float("inf"))
        add_check(
            checks,
            "Fixed-mode stability",
            "Bounded force identity",
            "PASS" if force_residuals and max_force_residual <= args.bounded_force_tol else "FAIL",
            (
                f"samples={len(force_residuals)}, max |logged-predicted|="
                f"{fmt(max_force_residual)} m/s^2"
            ),
        )

        transition_jumps = []
        for previous, current in zip(stable_rows, stable_rows[1:]):
            previous_index, previous_row = previous
            current_index, current_row = current
            if current_index != previous_index + 1:
                continue
            previous_cp = parse_float(previous_row, "tangent_escape_rmp_control_point_index")
            current_cp = parse_float(current_row, "tangent_escape_rmp_control_point_index")
            previous_generation = parse_float(previous_row, "tangent_escape_rmp_mode_generation")
            current_generation = parse_float(current_row, "tangent_escape_rmp_mode_generation")
            if not all(math.isfinite(value) for value in (
                previous_cp,
                current_cp,
                previous_generation,
                current_generation,
            )):
                continue
            if (
                int(round(previous_cp)) == int(round(current_cp)) and
                int(round(previous_generation)) != int(round(current_generation))
            ):
                previous_v = lyapunov(previous_row, args.position_gain)
                current_v = lyapunov(current_row, args.position_gain)
                if math.isfinite(previous_v) and math.isfinite(current_v):
                    transition_jumps.append(current_v - previous_v)
        positive_transition_jumps = [
            value for value in transition_jumps if value > args.transition_energy_tol
        ]
        transition_status = "PASS" if not positive_transition_jumps else "WARN"
        add_check(
            checks,
            "Stage-4 supervisor",
            "Hybrid energy jumps",
            transition_status,
            (
                f"transitions={len(transition_jumps)}, positive={len(positive_transition_jumps)}, "
                f"jump={fmt_range(transition_jumps)}. A positive reset jump prevents a global "
                "hybrid Lyapunov claim without an additional switching proof."
            ),
        )
    else:
        add_check(
            checks,
            "Fixed-mode stability",
            "Latched score mode",
            "INFO",
            "stable_hybrid_gds (leaf mode id 4) was not used.",
        )

    gds_active = [
        (index, row) for index, row in active
        if math.isfinite(parse_float(row, "tangent_escape_rmp_leaf_mode_id")) and
        int(round(parse_float(row, "tangent_escape_rmp_leaf_mode_id"))) in {2, 3, 4}
    ]
    fixed_pairs, positive_pairs, min_vdot, max_vdot = fixed_mode_vdot_stats(
        gds_active,
        args.position_gain,
    )
    positive_ratio = positive_pairs / fixed_pairs if fixed_pairs > 0 else float("nan")
    if fixed_pairs == 0:
        stability_status = "INFO" if collision_scaled_rows else "WARN"
    elif positive_ratio <= args.positive_vdot_ratio_warn:
        stability_status = "PASS"
    else:
        stability_status = "WARN"
    add_check(
        checks,
        "Fixed-mode stability",
        "Lyapunov finite-difference",
        stability_status,
        (
            f"fixed_mode_pairs={fixed_pairs}, positive={positive_pairs}, "
            f"positive_ratio={fmt(positive_ratio)}, dVdt_range={fmt(min_vdot)}..{fmt(max_vdot)}"
        ),
    )

    damping_predictions = []
    clamp_residuals = []
    for row in active_rows:
        metric = parse_float(row, "tangent_escape_rmp_effective_metric_scalar")
        velocity = parse_float(row, "tangent_escape_rmp_scalar_velocity_m_s")
        error = parse_float(row, "tangent_escape_rmp_scalar_error_m")
        desired = parse_float(row, "tangent_escape_rmp_desired_tangent_accel_m_s2")
        if all(math.isfinite(value) for value in (metric, velocity)):
            damping_predictions.append(-metric * args.damping_gain * velocity * velocity)
        mode_id = parse_float(row, "tangent_escape_rmp_leaf_mode_id")
        if (
            all(math.isfinite(value) for value in (velocity, error, desired)) and
            (not math.isfinite(mode_id) or int(round(mode_id)) not in {4, 5})
        ):
            unclamped = args.position_gain * error - args.damping_gain * velocity
            clamp_residuals.append(desired - unclamped)
    positive_damping = [value for value in damping_predictions if value > 1e-9]
    clamp_count = sum(1 for value in clamp_residuals if abs(value) > args.accel_clamp_residual_tol)
    add_check(
        checks,
        "Fixed-mode stability",
        "Damping sign",
        "PASS" if damping_predictions and not positive_damping else "FAIL",
        f"damping_prediction={fmt_range(damping_predictions)}, positive_count={len(positive_damping)}",
    )
    add_check(
        checks,
        "Fixed-mode stability",
        "Acceleration clamp influence",
        "INFO" if clamp_count == 0 else "WARN",
        f"clamp_residual_count={clamp_count}/{len(clamp_residuals)}, residual={fmt_range(clamp_residuals)}",
    )

    saturation_threshold = args.qdd_limit * args.saturation_ratio
    saturation_rows = []
    for index, row in enumerate(rows):
        values = qdd_vector(row)
        if all(math.isfinite(value) for value in values) and max(abs(value) for value in values) >= saturation_threshold:
            saturation_rows.append(index)
    active_saturation = [index for index in saturation_rows if index in active_indices]
    jerk, jerk_index = max_jerk(rows)
    add_check(
        checks,
        "Fixed-mode stability",
        "Joint acceleration saturation",
        "PASS" if not active_saturation else "WARN",
        (
            f"active_saturation_rows={len(active_saturation)}, all_saturation_rows={len(saturation_rows)}, "
            f"threshold={fmt(saturation_threshold)} rad/s^2"
        ),
    )
    add_check(
        checks,
        "Fixed-mode stability",
        "Joint acceleration jerk",
        "PASS" if not math.isfinite(jerk) or jerk <= args.jerk_warn else "WARN",
        f"max_qdd_jerk={fmt(jerk)} rad/s^3 at row={jerk_index}",
    )

    clearances = finite_values(parse_float(row, "tangent_escape_rmp_clearance_m") for row in active_rows)
    min_clearance = min(clearances) if clearances else float("nan")
    add_check(
        checks,
        "Safety / invariance",
        "Empirical active clearance",
        "PASS" if math.isfinite(min_clearance) and min_clearance >= args.active_clearance_min else "WARN",
        f"active clearance={fmt_range(clearances)}, threshold={fmt(args.active_clearance_min)} m",
    )

    external_clearances = finite_values(parse_float(row, "min_external_clearance_m") for row in rows)
    body_clearances = finite_values(parse_float(row, "min_body_clearance_m") for row in rows)
    if external_clearances:
        add_check(
            checks,
            "Safety / invariance",
            "Global external clearance",
            "PASS" if min(external_clearances) >= args.global_clearance_min else "WARN",
            f"min_external_clearance={fmt_range(external_clearances)}, threshold={fmt(args.global_clearance_min)} m",
        )
    else:
        add_check(checks, "Safety / invariance", "Global external clearance", "INFO", "No global external clearance field.")
    if body_clearances:
        add_check(
            checks,
            "Safety / invariance",
            "Robot body clearance",
            "PASS" if min(body_clearances) >= args.global_clearance_min else "WARN",
            f"min_body_clearance={fmt_range(body_clearances)}, threshold={fmt(args.global_clearance_min)} m",
        )

    add_check(
        checks,
        "Safety / invariance",
        "Formal forward invariance",
        "INFO",
        (
            "No CBF or hard safety constraint is checked here. This report gives empirical "
            "clearance evidence, not a formal invariance proof."
        ),
    )

    stage4_missing = [column for column in STAGE4_COLUMNS if column not in fieldnames]
    stage4_required = args.stage4_profile != "auto"
    add_check(
        checks,
        "Stage-4 supervisor",
        "Diagnostic columns",
        "FAIL" if stage4_required and stage4_missing else ("INFO" if stage4_missing else "PASS"),
        "missing=" + ",".join(stage4_missing) if stage4_missing else "all Stage-4 fields are present",
    )

    supervisor_modes_all = finite_values(
        parse_float(row, "tangent_escape_rmp_supervisor_mode_id") for row in rows
    )
    supervisor_modes = finite_values(
        parse_float(row, "tangent_escape_rmp_supervisor_mode_id") for row in active_rows
    )
    mode_set = {int(round(value)) for value in supervisor_modes_all}
    supervisor_observed = bool(mode_set.intersection({1, 2, 3}))
    mode_indices = {1: [], 2: [], 3: []}
    for index, row in enumerate(rows):
        value = parse_float(row, "tangent_escape_rmp_supervisor_mode_id")
        if not math.isfinite(value):
            continue
        mode = int(round(value))
        if mode in mode_indices:
            mode_indices[mode].append(index)
    mode_1_indices = mode_indices[1]
    mode_2_indices = mode_indices[2]
    mode_3_indices = mode_indices[3]
    if stage4_required:
        supervisor_status = "PASS" if supervisor_observed else "FAIL"
    else:
        supervisor_status = "PASS" if supervisor_observed else "INFO"
    add_check(
        checks,
        "Stage-4 supervisor",
        "Supervisor activation",
        supervisor_status,
        f"observed modes={sorted(mode_set) if mode_set else 'n/a'} (0=off, 1=preventive, 2=stuck, 3=recovery)",
    )

    hold_index_rows = [
        (index, row) for index, row in enumerate(rows)
        if parse_float(row, "tangent_escape_rmp_hold_active") >= 0.5
    ]
    hold_rows = len(hold_index_rows)
    hold_bonus_values = finite_values(
        parse_float(row, "tangent_escape_rmp_selected_hold_bonus")
        for _, row in hold_index_rows
    )
    hold_switches = 0
    premature_hold_switches = 0
    previous_hold = None
    for index, row in hold_index_rows:
        cp = parse_float(row, "tangent_escape_rmp_control_point_index")
        slot = parse_float(row, "tangent_escape_rmp_selected_candidate_index")
        key = (
            int(round(cp)) if math.isfinite(cp) else -1,
            int(round(slot)) if math.isfinite(slot) else -1,
        )
        if previous_hold is not None:
            previous_index, previous_key, previous_row = previous_hold
            if index == previous_index + 1 and key != previous_key:
                hold_switches += 1
                previous_age = parse_float(
                    previous_row,
                    "tangent_escape_rmp_branch_age_s",
                )
                message_dt = parse_float(row, "tangent_escape_rmp_dt_s")
                if not math.isfinite(message_dt):
                    previous_time = row_time(previous_row)
                    current_time = row_time(row)
                    message_dt = current_time - previous_time
                hold_elapsed = previous_age + max(message_dt, 0.0)
                if (
                    not math.isfinite(hold_elapsed) or
                    hold_elapsed < args.branch_hold_duration - args.branch_hold_time_tolerance
                ):
                    premature_hold_switches += 1
        previous_hold = (index, key, row)
    hold_positive = any(value > 1e-6 for value in hold_bonus_values)
    hold_ok = hold_rows > 0 and premature_hold_switches == 0 and hold_positive
    add_check(
        checks,
        "Stage-4 supervisor",
        "Branch hold",
        ("PASS" if hold_ok else "FAIL") if stage4_required else ("PASS" if hold_ok else "INFO"),
        (
            f"hold_rows={hold_rows}, switches={hold_switches}, "
            f"premature_switches={premature_hold_switches}, "
            f"hold_bonus={fmt_range(hold_bonus_values)}"
        ),
    )

    stuck_index_rows = [
        (index, row) for index, row in enumerate(rows)
        if parse_float(row, "tangent_escape_rmp_stuck_active") >= 0.5
    ]
    stuck_rows = len(stuck_index_rows)
    stuck_timers = finite_values(
        parse_float(row, "tangent_escape_rmp_stuck_timer_s") for _, row in stuck_index_rows
    )
    if args.stage4_profile == "forced_stuck":
        stuck_status = "PASS" if stuck_rows > 0 and mode_2_indices else "FAIL"
    elif args.stage4_profile == "normal":
        stuck_status = "PASS" if stuck_rows == 0 else "WARN"
    else:
        stuck_status = "INFO" if stuck_rows == 0 else "PASS"
    add_check(
        checks,
        "Stage-4 supervisor",
        "Stuck detector",
        stuck_status,
        f"mode2_rows={len(mode_2_indices)}, stuck_rows={stuck_rows}, timer={fmt_range(stuck_timers)}",
    )

    stuck_metric_boosts = finite_values(
        parse_float(row, "tangent_escape_rmp_metric_boost") for _, row in stuck_index_rows
    )
    stuck_accel_boosts = finite_values(
        parse_float(row, "tangent_escape_rmp_accel_boost") for _, row in stuck_index_rows
    )
    boost_observed = (
        any(value > 1.0 + 1e-6 for value in stuck_metric_boosts) or
        any(value > 1.0 + 1e-6 for value in stuck_accel_boosts)
    )
    if args.stage4_profile == "forced_stuck":
        boost_status = "PASS" if boost_observed else "FAIL"
    elif args.stage4_profile == "normal":
        boost_status = "PASS" if stuck_rows == 0 or boost_observed else "WARN"
    else:
        boost_status = "PASS" if boost_observed else "INFO"
    add_check(
        checks,
        "Stage-4 supervisor",
        "Stuck gain boost",
        boost_status,
        (
            f"metric_boost={fmt_range(stuck_metric_boosts)}, "
            f"accel_boost={fmt_range(stuck_accel_boosts)}"
            if stuck_rows > 0 else "not applicable: no stuck rows"
        ),
    )

    blocked = finite_values(
        parse_float(row, "tangent_escape_rmp_max_blocked_memory") for row in rows
    )
    max_blocked = max(blocked) if blocked else float("nan")
    candidate_blocked_rows = 0
    avoided_blocked_rows = 0
    max_candidate_penalty = 0.0
    for row in rows:
        selected = parse_float(row, "tangent_escape_rmp_selected_candidate_index")
        penalized_slots = set()
        for candidate in parse_candidates(row):
            penalty = candidate.get("blockedPenalty", float("nan"))
            slot = candidate.get("slot", float("nan"))
            if math.isfinite(penalty):
                max_candidate_penalty = max(max_candidate_penalty, penalty)
            if math.isfinite(penalty) and penalty > 1e-6 and math.isfinite(slot):
                penalized_slots.add(int(round(slot)))
        if penalized_slots:
            candidate_blocked_rows += 1
            if math.isfinite(selected) and int(round(selected)) not in penalized_slots:
                avoided_blocked_rows += 1
    memory_observed = math.isfinite(max_blocked) and max_blocked > 1e-3
    penalty_observed = max_candidate_penalty > 1e-6
    if args.stage4_profile == "forced_stuck":
        memory_status = "PASS" if memory_observed and penalty_observed else "FAIL"
    elif args.stage4_profile == "normal":
        memory_status = "INFO" if memory_observed else "PASS"
    else:
        memory_status = "PASS" if memory_observed and penalty_observed else "INFO"
    add_check(
        checks,
        "Stage-4 supervisor",
        "Blocked-branch memory",
        memory_status,
        (
            f"memory={fmt_range(blocked)}, max_candidate_penalty={fmt(max_candidate_penalty)}, "
            f"penalized_rows={candidate_blocked_rows}, avoided_rows={avoided_blocked_rows}"
        ),
    )

    recovery_status = "PASS" if mode_3_indices else ("WARN" if stage4_required else "INFO")
    add_check(
        checks,
        "Stage-4 supervisor",
        "Recovery mode",
        recovery_status,
        f"mode3_rows={len(mode_3_indices)}",
    )
    if args.stage4_profile == "forced_stuck":
        ordered = bool(
            mode_1_indices and mode_2_indices and mode_3_indices and
            mode_1_indices[0] < mode_2_indices[0] < mode_3_indices[-1]
        )
        add_check(
            checks,
            "Stage-4 supervisor",
            "Mode transition order",
            "PASS" if ordered else "FAIL",
            (
                f"first preventive={mode_1_indices[0] if mode_1_indices else 'n/a'}, "
                f"first stuck={mode_2_indices[0] if mode_2_indices else 'n/a'}, "
                f"last recovery={mode_3_indices[-1] if mode_3_indices else 'n/a'}"
            ),
        )

    summary["supervisor_modes"] = (
        ",".join(str(value) for value in sorted(mode_set)) if mode_set else "n/a"
    )
    summary["hold_rows"] = str(hold_rows)
    summary["stuck_rows"] = str(stuck_rows)
    summary["blocked_memory"] = fmt_range(blocked)
    return checks, summary


def status_rank(status: str) -> int:
    return {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}.get(status, 3)


def section_status(checks: Sequence[Check], section: str) -> str:
    section_checks = [check for check in checks if check.section == section]
    if not section_checks:
        return "INFO"
    worst = max(section_checks, key=lambda check: status_rank(check.status))
    return worst.status


def print_report(path: Path, checks: Sequence[Check], summary: Dict[str, str]) -> None:
    print(f"Input CSV: {path}")
    print(
        "Rows: {rows}, duration: {duration_s} s, active rows: {active_rows}, "
        "active intervals: {active_intervals}, active duration: {active_duration_s} s".format(**summary)
    )
    print(
        "Supervisor: modes={supervisor_modes}, hold_rows={hold_rows}, "
        "stuck_rows={stuck_rows}, blocked_memory={blocked_memory}".format(**summary)
    )
    print("")
    for section in (
        "Performance",
        "Score",
        "Closure",
        "Collision-scaled policy",
        "Fixed-mode stability",
        "Safety / invariance",
        "Stage-4 supervisor",
    ):
        print(f"{section}: {section_status(checks, section)}")
        for check in checks:
            if check.section == section:
                print(f"  [{check.status}] {check.name}: {check.detail}")
        print("")


def save_html(path: Path, csv_path: Path, checks: Sequence[Check], summary: Dict[str, str]) -> None:
    sections = (
        "Performance",
        "Score",
        "Closure",
        "Collision-scaled policy",
        "Fixed-mode stability",
        "Safety / invariance",
        "Stage-4 supervisor",
    )
    section_cards = "\n".join(
        f'<div class="card {section_status(checks, section).lower()}">'
        f"<div class=\"label\">{html.escape(section)}</div>"
        f"<div class=\"value\">{html.escape(section_status(checks, section))}</div>"
        "</div>"
        for section in sections
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(check.section)}</td>"
        f"<td>{html.escape(check.name)}</td>"
        f"<td><span class=\"pill {html.escape(check.status.lower())}\">{html.escape(check.status)}</span></td>"
        f"<td>{html.escape(check.detail)}</td>"
        "</tr>"
        for check in checks
    )
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in summary.items()
    )
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tangent Escape RMP Validation</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 10px; }}
    .path {{ color: #5f6b7a; margin-bottom: 20px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 12px; background: #f8fafc; }}
    .card.pass {{ border-color: #4caf50; }}
    .card.info {{ border-color: #607d8b; }}
    .card.warn {{ border-color: #ff9800; }}
    .card.fail {{ border-color: #e53935; }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #667085; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .pill {{ display: inline-block; min-width: 46px; text-align: center; border-radius: 999px; padding: 2px 8px; color: white; font-size: 12px; font-weight: 700; }}
    .pill.pass {{ background: #2e7d32; }}
    .pill.info {{ background: #455a64; }}
    .pill.warn {{ background: #ef6c00; }}
    .pill.fail {{ background: #c62828; }}
    .note {{ color: #5f6b7a; max-width: 980px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Tangent Escape RMP Validation</h1>
  <div class="path">{html.escape(str(csv_path))}</div>
  <div class="cards">{section_cards}</div>
  <h2>Summary</h2>
  <table>{summary_rows}</table>
  <h2>Checks</h2>
  <table>
    <thead><tr><th>Section</th><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Interpretation</h2>
  <p class="note">
    Closure checks are structural checks from the logged RMP leaf data. For stable_hybrid_gds,
    fixed-mode checks verify latched branch parameters and the bounded potential-damping identity.
    Finite-difference Escape energy is diagnostic because the other RMP leaves can exchange energy
    with this leaf. Hybrid mode resets are reported separately and require a switching argument for
    a global Lyapunov claim. Safety/invariance remains an empirical clearance check, not a formal
    forward-invariance proof.
  </p>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Tangent Escape RMP performance, closure, stability, and empirical safety."
    )
    parser.add_argument("csv", nargs="?", help="RMPFlow trace CSV. Defaults to latest compatible CSV.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory searched for latest CSV.")
    parser.add_argument("--save", help="Optional HTML report path.")
    parser.add_argument("--position-gain", type=float, default=16.0)
    parser.add_argument("--damping-gain", type=float, default=4.0)
    parser.add_argument("--max-accel", type=float, default=0.6)
    parser.add_argument(
        "--collision-accel-scale",
        type=float,
        default=None,
        help="Expected collision acceleration scale for leaf mode 5; defaults to the logged value.",
    )
    parser.add_argument("--collision-formula-tol", type=float, default=1e-6)
    parser.add_argument("--scalar-progress-ratio", type=float, default=0.8)
    parser.add_argument("--normal-dot-tol", type=float, default=0.05)
    parser.add_argument("--accel-normal-tol", type=float, default=0.05)
    parser.add_argument("--unit-vector-tol", type=float, default=0.05)
    parser.add_argument("--weight-sum-tol", type=float, default=0.02)
    parser.add_argument("--metric-eps", type=float, default=1e-9)
    parser.add_argument("--fixed-weight-tol", type=float, default=1e-6)
    parser.add_argument("--fixed-metric-tol", type=float, default=1e-5)
    parser.add_argument("--fixed-score-tol", type=float, default=1e-6)
    parser.add_argument("--fixed-direction-tol", type=float, default=1e-6)
    parser.add_argument("--energy-eps", type=float, default=1e-6)
    parser.add_argument("--bounded-force-tol", type=float, default=1e-5)
    parser.add_argument("--transition-energy-tol", type=float, default=1e-6)
    parser.add_argument("--positive-vdot-ratio-warn", type=float, default=0.2)
    parser.add_argument("--accel-clamp-residual-tol", type=float, default=1e-3)
    parser.add_argument("--qdd-limit", type=float, default=10.0)
    parser.add_argument("--saturation-ratio", type=float, default=0.95)
    parser.add_argument("--jerk-warn", type=float, default=100.0)
    parser.add_argument("--active-clearance-min", type=float, default=0.02)
    parser.add_argument("--global-clearance-min", type=float, default=0.0)
    parser.add_argument("--branch-hold-duration", type=float, default=0.6)
    parser.add_argument("--branch-hold-time-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--expect-duplicate-risk",
        action="store_true",
        help="Fail validation unless at least one candidate has positive predictive duplicate risk.",
    )
    parser.add_argument(
        "--stage4-profile",
        choices=("auto", "normal", "forced_stuck"),
        default="auto",
        help=(
            "Stage-4 expectation: auto reports what was observed, normal flags false stuck "
            "detections, and forced_stuck requires all supervisor transitions."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any check fails.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="With --strict, also return non-zero when any check warns.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser() if args.csv else latest_compatible_csv(Path(args.log_dir).expanduser())
    rows, fieldnames = load_rows(csv_path)
    checks, summary = summarize(rows, fieldnames, args)
    print_report(csv_path, checks, summary)
    if args.save:
        save_path = Path(args.save).expanduser()
    else:
        save_path = csv_path.with_name(f"{csv_path.stem}_rmp_validation.html")
    save_html(save_path, csv_path, checks, summary)
    print(f"Saved report: {save_path}")
    if args.strict:
        rejected = {"FAIL", "WARN"} if args.fail_on_warn else {"FAIL"}
        if any(check.status in rejected for check in checks):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
